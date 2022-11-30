"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VCR dataset
"""
from torch.utils.data import Dataset
import copy
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
from Data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb,
                             TxtLmdb, get_ids_and_lens, pad_tensors,
                             get_gather_index)
import random
import numpy as np
def dict_slice(adict):
    keys = adict.keys()
    dict_slice = {}
    for k in keys:
        dict_slice[k] = adict[k]
        if len(dict_slice)==100:
            break
    return dict_slice

def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch

class VcrTxtTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len=120, task="qa,qar"):
        # assert task == "qa" or task == "qar" or task == "qa,qar",\
        #     "VCR only support the following tasks: 'qa', 'qar' or 'qa,qar'"
        self.task = task
        if task == "qa,qar":
            id2len_task = "qar"
        else:
            id2len_task = task
        if max_txt_len == -1:
            self.id2len = json.load(
                open(f'{db_dir}/id2len_{id2len_task}.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(
                    open(f'{db_dir}/id2len_{id2len_task}.json')
                    ).items()
                if len_ <= max_txt_len
            }
        #self.id2len=dict_slice(self.id2len)

        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

class VcrDetectFeatTxtTokDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db_gt=None, img_db=None):
        # assert not (img_db_gt is None and img_db is None),\
        #     "img_db_gt and img_db cannot all be None"
        # assert isinstance(txt_db, VcrTxtTokLmdb)
        assert img_db_gt is None or isinstance(img_db_gt, DetectFeatLmdb)
        assert img_db is None or isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.img_db_gt = img_db_gt
        self.ls = img_db_gt
        self.task = self.txt_db.task
        txt_lens, self.ids = get_ids_and_lens(txt_db)

        txt2img = txt_db.txt2img

        if self.img_db and self.img_db_gt:
            self.lens = [tl+self.img_db_gt.name2nbb[txt2img[id_][0]] +
                         self.img_db.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        elif self.img_db:
            self.lens = [tl+self.img_db.name2nbb[txt2img[id_][1]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        elif self.img_db_gt:
            self.lens = [tl+self.img_db_gt.name2nbb[txt2img[id_][0]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        else:
            self.lens = [tl  for tl in txt_lens]

    def _get_img_feat(self, fname_gt, fname):
        if self.img_db and self.img_db_gt:
            img_feat_gt, bb_gt = self.img_db_gt[fname_gt]
            img_bb_gt = torch.cat([bb_gt, bb_gt[:, 4:5]*bb_gt[:, 5:]], dim=-1)

            img_feat, bb = self.img_db[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)

            img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
            img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
            num_bb = img_feat.size(0)
        elif self.img_db:
            img_feat, bb = self.img_db[fname]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        elif self.img_db_gt:
            img_feat, bb = self.img_db_gt[fname_gt]
            img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
            num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb

class Tensorizer(object):
    def __init__(self, tokenizer, max_qa=50):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.

        """
        self.tokenizer = tokenizer

        self.max_qa=max_qa

    def _truncate_seq_pair(self,tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def tensorize_example(self, ques,ans,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):


        ques = self.tokenizer.tokenize(ques)
        ans=self.tokenizer.tokenize(ans)
        self._truncate_seq_pair(ans, ques, self.max_qa - 3)

        tokens = ques + [self.tokenizer.sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens += ans +  [self.tokenizer.sep_token]
        segment_ids += [sequence_b_segment_id] * (len(ans) + 1)

        tokens = [self.tokenizer.cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_len = len(input_ids)

        padding_len = self.max_qa- seq_len
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)


        input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).cuda()
        input_mask = torch.tensor(input_mask, dtype=torch.long).cuda()
        return input_ids, segment_ids,input_mask

class Expl_VcrDataset(Dataset):
    def __init__(self, tokenizer,gpt_tokenizer, txt_db, img_db_gt, img_db):
        super().__init__()
        # assert self.task != "qa,qar",\
        #     "loading training dataset with each task separately"
        self.max_qa = 50
        self.max_img = 50

        self.tokenizer = tokenizer
        self.txt_db = txt_db
        self.img_db = img_db
        self.img_db_gt = img_db_gt

        self.txt2img = txt_db.txt2img
        self.img2txt=txt_db.img2txts
        self.ques_ids=list(self.txt2img.keys())
        self.img_ids=list(self.img2txt.keys())
        self.enc_token=tokenizer
        self.dec_token = gpt_tokenizer
        self.tensorizer = Tensorizer(self.enc_token, max_qa = self.max_qa)
        self.classes=[]
        with open('/raid/yq/Oscar/oscar/Data/utils/uniter_objects_vocab.txt') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())


    def __len__(self):
        return len(self.ques_ids)

    def __getitem__(self, i):
        ques_ids = self.ques_ids[i]
        img_id = self.txt2img[ques_ids]
        example=self.txt_db[ques_ids]
        img_feat, img_pos_feat, num_bb,labels = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])
        od_labels=example['objects'].copy()

        for i in range(len(labels)):
            od_labels.append(self.classes[labels[i]])
        region_label_ids = []
        syn_labels = []
        for ids,label in enumerate(od_labels):
            syn_labels_ids = self.enc_token.encode(label, add_special_tokens=False)
            region_label_ids.append(torch.tensor([1] * len(syn_labels_ids), dtype=torch.long).cuda())
            syn_labels_ids = torch.tensor(syn_labels_ids, dtype=torch.long).cuda()
            syn_labels.append(syn_labels_ids)
        #label只有groung label
        cat_tensor = torch.zeros((img_feat.size(0), 6), dtype=img_feat.dtype)
        cat_tensor[:, 0] = torch.div(img_pos_feat[:, 0], img_pos_feat[:, 4])
        cat_tensor[:, 1] = torch.div(img_pos_feat[:, 1], img_pos_feat[:, 5])
        cat_tensor[:, 2] = torch.div(img_pos_feat[:, 2], img_pos_feat[:, 4])
        cat_tensor[:, 3] = torch.div(img_pos_feat[:, 3], img_pos_feat[:, 5])
        cat_tensor[:, 4] = torch.div(img_pos_feat[:, 2]-img_pos_feat[:, 0], img_pos_feat[:, 3]-img_pos_feat[:, 1])
        cat_tensor[:, 5] = torch.div(img_pos_feat[:, 3]-img_pos_feat[:, 1], img_pos_feat[:, 2]-img_pos_feat[:, 0])
        img_feat = torch.cat((img_feat, cat_tensor), dim=-1).cuda()

        # image features
        img_len = img_feat.shape[0]
        syn_labels_ids = pad_sequence(syn_labels, batch_first=True, padding_value=0)
        region_label_ids = pad_sequence(region_label_ids, batch_first=True, padding_value=0)
        if img_len > self.max_img:
            img_feat = img_feat[0: self.max_img, ]
            syn_labels_ids=syn_labels_ids[0: self.max_img, ]
            region_label_ids=region_label_ids[0: self.max_img, ]
            img_len = img_feat.shape[0]
            img_mask=torch.ones(img_len,dtype=torch.long,device='cuda').cuda()
        else:
            padd_len = self.max_img - img_len
            padding_matrix = torch.zeros((padd_len,img_feat.shape[1])).cuda()
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            img_mask = torch.ones(img_len, dtype=torch.long, device='cuda').cuda()
            padd_mask = torch.zeros(padd_len, dtype=torch.long, device='cuda').cuda()
            img_mask = torch.cat((img_mask, padd_mask), 0).cuda()
            padding_matrix=torch.zeros((padd_len,syn_labels_ids.size(1)),dtype=torch.long).cuda()
            syn_labels_ids = torch.cat((syn_labels_ids, padding_matrix), 0)
            region_label_ids=torch.cat((region_label_ids, padding_matrix), 0)

        outputs = []
        ques=example['raw_q']
        label = example['qa_target']
        expl = example['raw_rs'][example['rationale_label']]
        expl = self.dec_token.encode(expl)
        gpt_ids = expl
        gpt_ids = [self.dec_token.bos_token_id] + gpt_ids + [self.dec_token.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids).cuda()
        attn_mask = torch.tensor(attn_mask).cuda()
        for index, ans_raw in enumerate(example['raw_as']):
            if index == label:
                target = torch.tensor([1]).long().cuda()
            else:
                target = torch.tensor([0]).long().cuda()
            # input_text=ques+ self.enc_token.sep_token+ans_raw
            input_ids, segment_ids, input_mask = self.tensorizer.tensorize_example(ques,ans_raw)
            input_mask=torch.cat((input_mask,img_mask),-1)
            # outputs.append((input_ids, segment_ids,input_mask,img_feat,target,gpt_ids,attn_mask,syn_labels_ids,region_label_ids))
            outputs.append((input_ids, segment_ids,input_mask,img_feat,target,syn_labels_ids,region_label_ids))
        return tuple(outputs),ques_ids,gpt_ids,attn_mask

    def _get_img_feat(self, fname_gt, fname):
        img_feat_gt, bb_gt,gt_labels = self.img_db_gt[fname_gt]
        img_bb_gt = torch.cat([bb_gt, bb_gt[:, 4:5]*bb_gt[:, 5:]], dim=-1)
        img_feat, bb ,labels= self.img_db[fname]
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        img_feat = torch.cat([img_feat_gt, img_feat], dim=0)
        img_bb = torch.cat([img_bb_gt, img_bb], dim=0)
        num_bb = img_feat.size(0)
        # labels=torch.cat((gt_labels,labels),-1)
        return img_feat,img_bb,num_bb,labels

    def collate(self,inputs):
        # (input_ids, segment_ids,input_mask,img_feat,target,gpt_ids,attn_mask,syn_labels_ids,region_label_ids) = map(list, unzip(concat(inputs)))
        (input_ids, segment_ids,input_mask,img_feat,target,syn_labels_ids,region_label_ids) = map(list, unzip(concat(outs for outs, _, _, _ in inputs)))
        gpt_ids = pad_sequence([t for _, _, t,_ in inputs], batch_first=True, padding_value=self.dec_token.pad_token_id)
        attn_mask = pad_sequence([t for _, _, _, t in inputs], batch_first=True, padding_value=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        target = torch.stack(target, dim=0)
        max_label_len = max([item.size(1) for item in region_label_ids])
        region_label_ids_padd = []
        syn_labels_ids_padd = []
        for idx, item in enumerate(region_label_ids):
            padd_matrix = torch.zeros((item.size(0), max_label_len - item.size(1)), dtype=torch.long).cuda()
            item = torch.cat((item, padd_matrix), dim=-1)
            region_label_ids_padd.append(item)
            label_padd = torch.cat((syn_labels_ids[idx], padd_matrix), dim=-1)
            syn_labels_ids_padd.append(label_padd)
        region_label_ids_padd = torch.stack(region_label_ids_padd, dim=0)
        syn_labels_ids_padd = torch.stack(syn_labels_ids_padd, dim=0)
        batch= {'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'target':target,'expl_ids':gpt_ids,'attn_mask':attn_mask,
                'region_label_ids':region_label_ids_padd,'syn_labels_ids':syn_labels_ids_padd
            }

        return batch