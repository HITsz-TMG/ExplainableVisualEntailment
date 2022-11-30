# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import sys
sys.path.append('../')
import argparse
import base64
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import torch
import csv
from toolz.sandbox import unzip
from cytoolz import concat
import json
from torch.nn.utils.rnn import pad_sequence
from utils.tsv_file import TSVFile
import base64
import cv2
import pickle
from transformers import BertTokenizerFast,GPT2Tokenizer

class CaptionTensorizer_GCN_global(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=100,
            max_seq_a_length=40,
            is_train=True,synonym_file=None):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.

        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.synonym_dict=self.read_synonym(synonym_file)
        self.max_synonym=5

    def read_synonym(self,synonym_file):
        with open(synonym_file,'r') as f:
            synonym_dict=json.load(f)
        return synonym_dict

    def tensorize_example(self, text_a, img_feat, global_feat,labels,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):

        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len=len(tokens)
        input_mask = [1] * seq_a_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids=torch.tensor(input_ids).cuda()
        padding_a_len = self.max_seq_a_len - len(tokens)
        padding_tensor=torch.zeros(padding_a_len,dtype=torch.long,device='cuda')
        input_ids=torch.cat((input_ids,padding_tensor),0)
        segment_ids=torch.tensor(segment_ids,device='cuda')
        segment_ids = torch.cat((segment_ids, padding_tensor), 0)
        input_mask=torch.tensor(input_mask,device='cuda')
        input_mask = torch.cat((input_mask, padding_tensor), 0)



        region_label_ids = []
        syn_labels=[]
        for ids,label in enumerate(labels):
            tmp_list = [label]
            if label in self.synonym_dict.keys():
                tmp_list.extend(self.synonym_dict[label])
                # if len(tmp_list)>self.max_synonym:
                #     tmp_list = random.sample(tmp_list, self.max_synonym)  # 从list中随机获取5个元素，作为一个片断返回
            tmp_str=' '.join(word for word in tmp_list)
            syn_labels_ids = self.tokenizer.encode(tmp_str, add_special_tokens=False)
            region_label_ids.append(torch.tensor([1] * len(syn_labels_ids), dtype=torch.long).cuda())
            syn_labels_ids = torch.tensor(syn_labels_ids, dtype=torch.long).cuda()
            syn_labels.append(syn_labels_ids)




        assert len(input_ids) == self.max_seq_a_len
        assert len(segment_ids) == self.max_seq_a_len
        assert len(input_mask) == self.max_seq_a_len

        # image features
        img_len = img_feat.shape[0]
        syn_labels_ids = pad_sequence(syn_labels, batch_first=True, padding_value=0)
        region_label_ids = pad_sequence(region_label_ids, batch_first=True, padding_value=0)
        if img_len > self.max_img_seq_len-1:
            img_feat = img_feat[0: self.max_img_seq_len-1, ]
            syn_labels_ids=syn_labels_ids[0: self.max_img_seq_len-1, ]
            region_label_ids=region_label_ids[0: self.max_img_seq_len-1, ]

            img_feat = torch.cat((global_feat, img_feat), dim=0)
            pad_label = torch.zeros((1, syn_labels_ids.size(1)), dtype=torch.long).cuda()
            syn_labels_ids = torch.cat((pad_label, syn_labels_ids), 0)
            region_label_ids = torch.cat((pad_label, region_label_ids), 0)

            img_len = img_feat.shape[0]
            padd_mask=torch.ones(img_len,dtype=torch.long,device='cuda')
            input_mask = torch.cat((input_mask,padd_mask),0)
        else:
            padd_len = self.max_img_seq_len-1 - img_len
            img_feat = torch.cat((global_feat, img_feat), dim=0)
            padding_matrix = torch.zeros((padd_len,img_feat.shape[1])).cuda()
            img_feat = torch.cat((img_feat, padding_matrix), 0)

            pad_label = torch.zeros((1, syn_labels_ids.size(1)), dtype=torch.long).cuda()
            syn_labels_ids = torch.cat((pad_label, syn_labels_ids), 0)
            region_label_ids = torch.cat((pad_label, region_label_ids), 0)

            padd_mask = torch.ones(img_len+1, dtype=torch.long, device='cuda')
            input_mask = torch.cat((input_mask, padd_mask), 0)
            padd_mask = torch.zeros(padd_len, dtype=torch.long, device='cuda')
            input_mask = torch.cat((input_mask, padd_mask), 0)

            padding_matrix=torch.zeros((padd_len,syn_labels_ids.size(1)),dtype=torch.long).cuda()
            syn_labels_ids = torch.cat((syn_labels_ids, padding_matrix), 0)
            region_label_ids=torch.cat((region_label_ids, padding_matrix), 0)

        return input_ids, segment_ids,input_mask,img_feat,syn_labels_ids,region_label_ids

class SNLI_GCN_relation_align_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,synonym_file,relation_file,whole_feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.img_dir='/'.join(self.feat_dir.split('/')[:-1])+'/flickr30k_images/flickr30k_images'
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.relation=json.load(open(relation_file,'r'))
        self.whole_feat=pickle.load(open(whole_feat_file,'rb'))
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        #因为加入了全局visual节点,所以需要-1
        self.tensorizer = CaptionTensorizer_GCN_global(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train,synonym_file=synonym_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.relation_threshold=0.01
        self.max_hypo_len=max_hypo_len
        self.max_seq_len=max_seq_length
        self.cls_template='The answer is'

    def get_valid_tsv(self):
        # based on the order of file size
        if self.label_tsv:
            return self.label_tsv
        if self.feat_tsv:
            return self.feat_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.seek(i)[0] for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.seek(i)[0] : i for i in range(tsv.num_rows())}

    def read_csv_annot(self,SNLI_annot_file,img_dict):
        f = open(SNLI_annot_file, 'r')
        csvreader = csv.reader(f)
        result = {}
        idx = 0
        for item in csvreader:
            # 忽略第一行
            try:
                if csvreader.line_num == 1:
                    continue
                # if csvreader.line_num>10741:
                result[idx] = {}
                result[idx]['pairID'] = item[1]
                result[idx]['Flickr30kID'] = item[2]
                result[idx]['hypothesis'] = item[3]
                result[idx]['gold_label'] = item[4]
                result[idx]['explanation'] = item[5]
                idx += 1
            except:
                None
                # 因为会自动读到最后一行空行
        return result


    def read_csv_cap(self,flickr_caption_file):
        f=open(flickr_caption_file,'r')
        csvreader=csv.reader(f)
        result = {}
        for item in csvreader:
            # 忽略第一行
            if csvreader.line_num==1:
                continue
            #第一个cap
            if csvreader.line_num %5==2:
                infos=item[0].split('|')
                cap=infos[-1]
                if len(item)>1:
                    caps=','.join(i for i in item[1:])
                    cap=cap+','+caps
                result[infos[0]] = [cap]
            else:
                infos = item[0].split('|')
                cap = infos[-1]
                if len(item) > 1:
                    caps = ','.join(i for i in item[1:])
                    cap = cap + ',' + caps
                result[infos[0]].append(cap)
        return result

    def read_json_annot(self,SNLI_obj_file):
        f=open(SNLI_obj_file,'r')
        data=f.readlines()
        result=json.loads(data[0])
        return result

    def get_image_key(self, idx):
        return self.image_keys[idx]

    def get_image_features(self, img_idx):
        feat_info=self.feat_tsv.seek(img_idx)
        num_boxes = int(feat_info[1])
        features = np.frombuffer(base64.b64decode(feat_info[2]), np.float32
                ).reshape((num_boxes, -1))
        return torch.Tensor(features)

    def get_od_labels(self, img_idx):

        num=0
        label_info = json.loads(self.label_tsv.seek(img_idx)[1])
        od_labels=[]
        gt_boxes=[]
        for l in label_info['objects']:
            if l['conf']>self.nms_threshold:
                od_labels.append(l['class'])
                gt_boxes.append(l['rect'])
                num+=1
            else:
                break
        img_w=label_info['image_w']
        img_h=label_info['image_h']
        return od_labels,num,gt_boxes,img_w,img_h

    def get_len(self):
        return len(self.SNLI_annot_dict)

    def get_feat(self, i):
        example = self.SNLI_annot_dict[i]
        flickrID=example['Flickr30kID'].split('.')[0]

        if flickrID in self.image2id.keys():
            img_idx = self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num,gt_boxes,img_w,img_h = self.get_od_labels(img_idx)
            features=features.cuda()
            features = features[:num]
            gt_boxes = torch.tensor(gt_boxes).cuda()
        else:
            path = os.path.join(self.feat_dir, 'box_feat', flickrID)
            features = torch.load(path)
            features = features.cuda()
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir, 'pred', file_name)
            with open(path, 'r') as f:
                data = f.read()
            data = json.loads(data)
            od_labels = []
            gt_rel_boxes = []
            gt_boxes = []
            for item in data:
                x1, y1, x2, y2 = item['rect']
                gt_boxes.append(item['rect'])
                w = x2 - x1
                h = y2 - y1
                gt_rel_boxes.append([x1, y1, w, h])
                tmp_label = item['class']
                od_labels.append(tmp_label)
            gt_rel_boxes = torch.tensor(gt_rel_boxes).cuda()
            gt_boxes = torch.tensor(gt_boxes).cuda()
            cat_tensor = torch.zeros((features.size(0), 6)).cuda()
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
            img_shape=cv2.imread(os.path.join(self.img_dir,example['Flickr30kID'])).shape
            img_w=img_shape[1]
            img_h = img_shape[0]
        hypo = example['hypothesis']
        label = torch.tensor(self.label2id[example['gold_label']]).cuda()

        global_feat = self.whole_feat[flickrID]
        cat_tensor = torch.zeros((1, 6), dtype=torch.float32).cuda()
        cat_tensor[0, 2] = 1.
        cat_tensor[0, 2] = 1.
        rel_rect = torch.tensor([0, 0, img_w, img_h]).cuda()
        cat_tensor[0, 4] = torch.div(rel_rect[2], rel_rect[3])
        cat_tensor[0, 5] = torch.div(rel_rect[3], rel_rect[2])
        global_feat = torch.cat((global_feat, cat_tensor), -1)

        input_ids, segment_ids,input_mask,img_feat,syn_labels_ids,region_label_ids = self.tensorizer.tensorize_example(hypo,features,global_feat,od_labels)
        expl=example['explanation']
        expl = self.gpt_toker.encode(expl)
        gpt_ids=expl
        gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
        attn_mask=[1]*len(gpt_ids)
        gpt_ids=torch.tensor(gpt_ids).cuda()
        attn_mask=torch.tensor(attn_mask).cuda()
        od_labels = self.bert_toker.encode((' ').join(item for item in od_labels), add_special_tokens=False)
        od_labels=torch.tensor(od_labels)
        #判定box是否相交
        #把global信息拼接
        gt_boxes=torch.cat((rel_rect.unsqueeze(0),gt_boxes),0)
        if gt_boxes.size(0)>img_feat.size(0):
            gt_boxes=gt_boxes[:img_feat.size(0)]
        nodes, dim = gt_boxes.shape

        A = gt_boxes.size(0)
        B = gt_boxes.size(0)
        max_xy = torch.min(gt_boxes[:, 2:].unsqueeze(1).expand(A, B, 2),
                           gt_boxes[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(gt_boxes[:, :2].unsqueeze(1).expand(A, B, 2),
                           gt_boxes[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]

        #计算整个面积
        area_a = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
                  (gt_boxes[:, 3] - gt_boxes[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]

        inter_mask =(inter/area_a>0.5)

        relation_mask=torch.eye(nodes,dtype=torch.long).cuda()
        relation_mask=torch.where(relation_mask ==1, torch.full_like(relation_mask,self.bert_toker.encode("<self>",add_special_tokens=False)[0]), relation_mask)
        #pad为0
        if flickrID in self.relation.keys():
            relation_list=self.relation[flickrID]
            for r in relation_list:
                if r['scores']>self.relation_threshold:
                    subj=r['rel_subj_centers']
                    obj=r['rel_obj_centers']
                    #当时就是feat顺序存的
                    if subj < (img_feat.size(0)-1) and obj <(img_feat.size(0)-1):
                        relation_mask[subj+1,obj+1]=self.bert_toker.encode(r['relation'],add_special_tokens=False)[0]

        pad_len=img_feat.size(0)-inter_mask.size(0)
        if pad_len>0:
            padd_matrix=torch.full((inter_mask.size(0),pad_len),False).cuda()
            inter_mask=torch.cat((inter_mask,padd_matrix),-1)
            padd_matrix = torch.zeros((relation_mask.size(0), pad_len), dtype=torch.long).cuda()
            relation_mask=torch.cat((relation_mask,padd_matrix),-1)
            padd_matrix = torch.full((pad_len, img_feat.size(0)), False).cuda()
            inter_mask = torch.cat((inter_mask, padd_matrix), 0)
            padd_matrix = torch.zeros((pad_len, img_feat.size(0)), dtype=torch.long).cuda()
            relation_mask = torch.cat((relation_mask, padd_matrix), 0)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,syn_labels_ids,region_label_ids,inter_mask,relation_mask,od_labels)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,syn_labels_ids,region_label_ids,inter_mask,relation_mask,od_labels) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        od_labels = pad_sequence(od_labels, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        inter_mask = torch.stack(inter_mask, dim=0)
        relation_mask = torch.stack(relation_mask, dim=0)
        max_label_len=max([item.size(1) for item in region_label_ids])
        region_label_ids_padd=[]
        syn_labels_ids_padd=[]
        for idx,item in enumerate(region_label_ids):
            padd_matrix=torch.zeros((item.size(0),max_label_len-item.size(1)),dtype=torch.long).cuda()
            item=torch.cat((item,padd_matrix),dim=-1)
            region_label_ids_padd.append(item)
            label_padd=torch.cat((syn_labels_ids[idx],padd_matrix),dim=-1)
            syn_labels_ids_padd.append(label_padd)
        region_label_ids_padd = torch.stack(region_label_ids_padd, dim=0)
        syn_labels_ids_padd = torch.stack(syn_labels_ids_padd, dim=0)

        # region_label_ids=self.get_gather_index(region_label_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,
                'region_label_ids':region_label_ids_padd,'syn_labels_ids':syn_labels_ids_padd,'inter_mask':inter_mask,
                'relation_mask':relation_mask,'od_labels':od_labels
            }

        return batch
    def get_gather_index(self,region_label_ids):

        obj_len=max([len(item) for item in region_label_ids])
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_label_ids),obj_len,1024).clone()
        for i, region in enumerate(region_label_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--SNLI_annot_file_train", default="/mnt/inspurfs/user-fs/yangqian/e-ViL-main/data/esnlive_train.csv", type=str)
    parser.add_argument("--SNLI_annot_file_dev", default="/mnt/inspurfs/user-fs/yangqian/e-ViL-main/data/esnlive_dev.csv", type=str)
    parser.add_argument("--SNLI_annot_file_test", default="/mnt/inspurfs/user-fs/yangqian/e-ViL-main/data/esnlive_test.csv", type=str)

    parser.add_argument("--flickr_caption_file", default="/mnt/inspurfs/user-fs/yangqian/e-ViL-main/flickr30k_images/results.csv", type=str)
    parser.add_argument("--feat_file", default="/mnt/inspurfs/user-fs/yangqian/e-ViL-main/OSCAR_feat/features.tsv", type=str)
    parser.add_argument("--synonym_file", default="/mnt/inspurfs/user-fs/yangqian/e-ViL-main/data/concept_synonyms.json",type=str)
    parser.add_argument("--relation_file", default="/mnt/inspurfs/user-fs/yangqian/e-ViL-main/relations.json",type=str)
    parser.add_argument("--whole_feat_file", default="/mnt/inspurfs/user-fs/yangqian/e-ViL-main/whole_feat.pkl",type=str)
    parser.add_argument("--data_dir", default='/mnt/inspurfs/user-fs/yangqian/Oscar/datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--gpt_model_name_or_path", default='/mnt/inspurfs/user-fs/yangqian/GPT2', type=str, required=False,
                        help="Path to GPT model.")
    parser.add_argument("--model_name_or_path", default='/mnt/inspurfs/user-fs/yangqian/Oscar/pretrained_models/image_captioning/pretrained_large/pretrained_large/checkpoint-1410000/', type=str, required=False,
                        help="Path to pre-trained model or model type.")

    args = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_model_name_or_path,bos_token='[CLS]',eos_token='[SEP]',pad_token='[PAD]')

    train_dataset = SNLI_GCN_relation_align_Dataset(tokenizer, gpt_tokenizer, args.SNLI_annot_file_train,
                                                           args.flickr_caption_file, args.feat_file, args.synonym_file,
                                                           args.relation_file, args.whole_feat_file,
                                                           max_hypo_len=args.max_hypo_len,
                                                           max_seq_length=args.max_seq_length)
    total_num=train_dataset.get_len()
    pbar=tqdm(ntotal=total_num)
    total_example=[]
    for i in range(total_num):
        tmp_example=train_dataset.get_feat(i)
        total_example[tmp_example['img_id']]=tmp_example