import random
import os
from torch.utils.data import Dataset
import torch
import csv
from toolz.sandbox import unzip
from cytoolz import concat
import json
import numpy as np
# from torch.nn.utils.rnn import pad_sequence
from .data import (get_ids_and_lens, pad_tensors,
                   get_gather_index)
from oscar.utils.tsv_file import TSVFile
import base64
import cv2

# from tools.demo.visual_utils import draw_bb, draw_rel

def put_text(im, text, bottomleft=(0,100), color=(255,255,255),
        font_scale=0.5, font_thickness=1):
    # function borrowed from quickdetection
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, text, bottomleft, font, font_scale, color,
            thickness=font_thickness)
    return cv2.getTextSize(text, font, font_scale, font_thickness)[0]

def get_font_info(img_size):
    font = cv2.FONT_HERSHEY_SIMPLEX  # default font
    ref = (img_size[0] + img_size[1]) / 2
    font_scale = ref / 1000
    font_thickness = int(max(ref / 400, 1))
    return font, font_scale, font_thickness

def draw_bb(im, all_rect, all_label, probs=None, color=None,
            draw_label=True):
    '''
    function borrowed from quickdetection.
    all_rect: in xyxy mode
    all_label: list of class names
    probs: list of confidence scores, will show if given
    '''
    font, font_scale, font_thickness = get_font_info(im.shape[:2])

    dist_label = set(all_label)
    if color is None:
        gold_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), ]
        color = {}
        for l in dist_label:
            if l in color:
                continue
            if len(gold_colors) > 0:
                color[l] = gold_colors.pop()
    for i, l in enumerate(dist_label):
        if l in color:
            continue
        color[l] = (random.random() * 255., random.random() * 255, random.random() * 255)

    if type(all_rect) is list:
        assert len(all_rect) == len(all_label)
    elif type(all_rect) is np.ndarray:
        assert all_rect.shape[0] == len(all_label)
        assert all_rect.shape[1] == 4
    else:
        assert False

    for i in range(len(all_label)):
        rect = all_rect[i]
        label = all_label[i]
        cv2.rectangle(im, (int(rect[0]), int(rect[1])),
                      (int(rect[2]), int(rect[3])), color[label],
                      thickness=font_thickness)
        if probs is not None:
            if draw_label:
                label_in_image = '{}-{:.2f}'.format(label, probs[i])
            else:
                label_in_image = '{:.2f}'.format(probs[i])
        else:
            if draw_label:
                label_in_image = '{}'.format(label)

        def gen_candidate():
            # above of top left
            yield int(rect[0]) + 2, int(rect[1]) - 4
            # below of bottom left
            yield int(rect[0]) + 2, int(rect[3]) + text_height + 2

        if draw_label or probs is not None:
            (_, text_height), _ = cv2.getTextSize(label_in_image, font,
                                                  font_scale, font_thickness)
            for text_left, text_bottom in gen_candidate():
                if 0 <= text_left < im.shape[1] - 12 and 12 < text_bottom < im.shape[0]:
                    put_text(im, label_in_image, (text_left, text_bottom), color[label],
                             font_scale, font_thickness)
                    break

def pad_sequence(sequences, batch_first=False, padding_value=0):
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    # max_len+=1
    #增加一位
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70,
            max_seq_a_length=40,
            is_train=True):
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
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len,
            self.max_seq_len), dtype=torch.long))

    def tensorize_example(self,expl, text_a, img_feat, text_b,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_expl = self.tokenizer.tokenize(expl)
        if len(tokens_expl) > self.max_seq_a_len - 2:
            tokens_expl = tokens_expl[:(self.max_seq_a_len - 2)]

        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len=len(tokens)
        input_mask = [1] * seq_a_len

        padding_a_len = self.max_seq_a_len - len(tokens)
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        input_mask = input_mask + ([0] * padding_a_len)
        # hypo，也就是需要生成的部分
        tokens += [self.tokenizer.cls_token]+tokens_expl + [self.tokenizer.sep_token]
        segment_ids += [sequence_a_segment_id] * (len(tokens_expl) + 2)
        padding_a_len = self.max_seq_a_len*2 - len(tokens)
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        #对于分类，解释需要完全mask
        input_mask = input_mask + ([0] * self.max_seq_a_len)

        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len+self.max_seq_a_len- len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len+self.max_seq_a_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            input_mask = input_mask + ([1] * (len(tokens_b) + 1))

        seq_len = len(tokens)
        # pad on the right for image captioning
        padding_len = self.max_seq_len+self.max_seq_a_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask + ([0] * padding_len)
        assert len(input_ids) == self.max_seq_len+self.max_seq_a_len
        assert len(segment_ids) == self.max_seq_len+self.max_seq_a_len
        assert len(input_mask) == self.max_seq_len + self.max_seq_a_len

        # image features
        img_len = img_feat.shape[0]

        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
            input_mask = input_mask + ([1] * img_len)
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            input_mask = input_mask + ([1] * img_len)
            input_mask = input_mask + ([0] * (self.max_img_seq_len - img_len))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        max_len = self.max_seq_len +self.max_seq_a_len+ self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # h: hypo,e:expl, L: label, R: image region
        h_start, h_end = 0, len(tokens_a)+2
        e_start, e_end =self.max_seq_a_len,self.max_seq_a_len+len(tokens_expl)+2
        l_start, l_end = self.max_seq_a_len*2, seq_len
        r_start, r_end = self.max_seq_len+self.max_seq_a_len, self.max_seq_len+self.max_seq_a_len +img_len
        # triangle mask for expl to expl
        attention_mask[e_start: e_end, e_start: e_end].copy_(self._triangle_mask[0: len(tokens_expl)+2, 0: len(tokens_expl)+2])
        # full attention for L-L, R-R,h-h
        attention_mask[h_start: h_end, h_start: h_end] = 1
        attention_mask[l_start: l_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, r_start: r_end] = 1
        # full attention for E-H,E-L, E-R,
        attention_mask[e_start: e_end, h_start: h_end] = 1
        attention_mask[e_start: e_end, l_start: l_end] = 1
        attention_mask[e_start: e_end, r_start: r_end] = 1
        # full attention for L-R,L-h,h-R:
        attention_mask[l_start: l_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, l_start: l_end] = 1

        attention_mask[l_start: l_end, h_start: h_end] = 1
        attention_mask[h_start: h_end, l_start: l_end] = 1

        attention_mask[h_start: h_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, h_start: h_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_mask=torch.tensor(input_mask,dtype=torch.long)

        return input_ids, segment_ids,attention_mask,input_mask,img_feat

def _pad_ids(ids, max_len):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        return ids + [0] * (max_len - len(ids))

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

class SNLIDataset(Dataset):
    def __init__(self,tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.tokenizer=tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50

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
        f=open(SNLI_annot_file,'r')
        csvreader=csv.reader(f)
        result = {}
        idx=0
        for item in csvreader:
            # 忽略第一行
            try:
                if csvreader.line_num == 1:
                    continue
                if item[2].split('.')[0] in img_dict.keys():
                    result[idx] = {}
                    result[idx]['pairID']=item[1]
                    result[idx]['Flickr30kID'] = item[2]
                    result[idx]['hypothesis'] = item[3]
                    result[idx]['gold_label']=item[4]
                    result[idx]['explanation'] = item[5]
                    idx+=1
            except:
                None
                #因为会自动读到最后一行空行
        return result

    def read_csv_cap(self,flickr_caption_file):
        f=open(flickr_caption_file,'r')
        csvreader=csv.reader(f)
        result = {}
        for item in csvreader:
            # 忽略第一行
            if csvreader.line_num %5==2:
                infos=item[0].split('|')
                cap=infos[-1]
                if len(item)>1:
                    caps=','.join(i for i in item[1:])
                    cap=cap+','+caps
                result[infos[0]]=cap
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        flickrID=example['Flickr30kID'].split('.')[0]
        img_idx=self.image2id[flickrID]
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        explain=example['explanation']
        features = self.get_image_features(img_idx)
        od_labels,num= self.get_od_labels(img_idx)
        features=features[:num]
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,attention_mask,input_mask,img_feat = self.tensorizer.tensorize_example(explain,hypo,features,od_labels)
        return (torch.tensor(int(flickrID)),input_ids, segment_ids,attention_mask,input_mask,img_feat,label)

class CaptionTensorizer_cls(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=100,
            max_seq_a_length=40,
            is_train=True):
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


    def tensorize_example(self, text_a, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):


        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len=len(tokens)
        input_mask = [1] * seq_a_len

        padding_a_len = self.max_seq_a_len - len(tokens)
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        input_mask = input_mask + ([0] * padding_a_len)


        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len- len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            input_mask = input_mask + ([1] * (len(tokens_b) + 1))

        seq_len = len(tokens)
        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask + ([0] * padding_len)
        assert len(input_ids) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len

        # image features
        img_len = img_feat.shape[0]

        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
            input_mask = input_mask + ([1] * img_len)
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            input_mask = input_mask + ([1] * img_len)
            input_mask = input_mask + ([0] * (self.max_img_seq_len - img_len))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        return input_ids, segment_ids,input_mask,img_feat

class SNLIDataset_cls(Dataset):
    def __init__(self,tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,flickr_intent_file,flickr_after_file,max_img_seq_length=50, max_seq_length=100, max_seq_a_length=40,
            is_train=True):
        self.flickr_intent_dict=self.read_json_pred(flickr_intent_file)
        self.flickr_after_dict = self.read_json_pred(flickr_after_file)
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.tokenizer=tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_cls(self.tokenizer, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.det2person_dic=self.det2person_trans()

    def det2person_trans(self):
        dic={}
        for i in range(100):
            dic['<|det%s|>' % (str(i))]=i
        return dic

    def read_json_pred(self,file_path):
        dic={}
        with open(file_path,'r') as f:
            data=f.readline()
        list=json.loads(data)
        for item in list:
            dic[item['id']]=item
        return dic

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
        f=open(SNLI_annot_file,'r')
        csvreader=csv.reader(f)
        result = {}
        idx=0
        for item in csvreader:
            # 忽略第一行
            try:
                if csvreader.line_num == 1:
                    continue
                if item[2].split('.')[0] in img_dict.keys():
                    result[idx] = {}
                    result[idx]['pairID']=item[1]
                    result[idx]['Flickr30kID'] = item[2]
                    result[idx]['hypothesis'] = item[3]
                    result[idx]['gold_label']=item[4]
                    result[idx]['explanation'] = item[5]
                    idx+=1
            except:
                None
                #因为会自动读到最后一行空行
        return result

    def read_csv_cap(self,flickr_caption_file):
        f=open(flickr_caption_file,'r')
        csvreader=csv.reader(f)
        result = {}
        for item in csvreader:
            # 忽略第一行
            if csvreader.line_num %5==2:
                infos=item[0].split('|')
                cap=infos[-1]
                if len(item)>1:
                    caps=','.join(i for i in item[1:])
                    cap=cap+','+caps
                result[infos[0]]=cap
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        flickrID=example['Flickr30kID'].split('.')[0]
        img_idx=self.image2id[flickrID]
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        features = self.get_image_features(img_idx)
        od_labels,num= self.get_od_labels(img_idx)
        features=features[:num]
        # caption=self.flickr_caption_dict[flickrID+'.jpg']
        # sup_infp = caption + od_labels

        sup_list=self.flickr_intent_dict[flickrID+'.jpg']['generations'][:2]
        sup_list+=self.flickr_after_dict[flickrID+'.jpg']['generations'][:2]
        sup_str='.'.join(sup_list)
        sup_infp=sup_str+od_labels
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat = self.tensorizer.tensorize_example(hypo,features,sup_infp)
        return (torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label)

class SNLIGPTDataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.max_expl_len=50

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
        f=open(SNLI_annot_file,'r')
        csvreader=csv.reader(f)
        result = {}
        idx=0
        for item in csvreader:
            # 忽略第一行
            try:
                if csvreader.line_num == 1:
                    continue
                if item[2].split('.')[0] in img_dict.keys():
                    result[idx] = {}
                    result[idx]['pairID']=item[1]
                    result[idx]['Flickr30kID'] = item[2]
                    result[idx]['hypothesis'] = item[3]
                    result[idx]['gold_label']=item[4]
                    result[idx]['explanation'] = item[5]
                    idx+=1
            except:
                None
                #因为会自动读到最后一行空行
        return result

    def read_csv_cap(self,flickr_caption_file):
        f=open(flickr_caption_file,'r')
        csvreader=csv.reader(f)
        result = {}
        for item in csvreader:
            # 忽略第一行
            if csvreader.line_num %5==2:
                infos=item[0].split('|')
                cap=infos[-1]
                if len(item)>1:
                    caps=','.join(i for i in item[1:])
                    cap=cap+','+caps
                result[infos[0]]=cap
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        flickrID=example['Flickr30kID'].split('.')[0]
        img_idx=self.image2id[flickrID]
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        features = self.get_image_features(img_idx)
        od_labels,num= self.get_od_labels(img_idx)
        features=features[:num]
        caption=self.flickr_caption_dict[flickrID+'.jpg']
        sup_infp=caption+od_labels
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat = self.tensorizer.tensorize_example(hypo,features,od_labels)
        expl=example['explanation']
        expl_ids=self.gpt_toker(expl)['input_ids']
        if len(expl_ids)>self.max_expl_len-2:
            expl_ids=expl_ids[:self.max_expl_len-2]
        expl_ids=[self.gpt_toker.bos_token_id]+expl_ids+[self.gpt_toker.eos_token_id]
        pad_list=[self.gpt_toker.pad_token_id]*(self.max_expl_len-len(expl_ids))
        attn_mask=[1]*len(expl_ids)
        attn_mask += [0] * (self.max_expl_len - len(expl_ids))
        expl_ids+=pad_list
        expl_ids=torch.tensor(expl_ids)
        attn_mask=torch.tensor(attn_mask)
        return (torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,expl_ids,attn_mask)

class SNLIGPT_gen_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,max_caption_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.max_caption_len = max_caption_len
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

    # def read_csv_annot(self,SNLI_annot_file,img_dict):
    #
    #     save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
    #     with open(os.path.join(save_path, 'reelation_train.json'), 'r') as f:
    #         result = json.load(f)
    #     return result
    def read_csv_annot(self,SNLI_annot_file,img_dict):
        f=open(SNLI_annot_file,'r')
        csvreader=csv.reader(f)
        result = {}
        idx=0
        for item in csvreader:
            # 忽略第一行
            try:
                if csvreader.line_num == 1:
                    continue
                # if item[2].split('.')[0] in img_dict.keys():
                result[idx] = {}
                result[idx]['pairID']=item[1]
                result[idx]['Flickr30kID'] = item[2]
                result[idx]['hypothesis'] = item[3]
                result[idx]['gold_label']=item[4]
                result[idx]['explanation'] = item[5]
                idx+=1
            except:
                None
                #因为会自动读到最后一行空行
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)


        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        #将假设、object、图片拼接，用于判断假设是否正确
        # input_ids, segment_ids,input_mask,img_feat = self.tensorizer.tensorize_example(hypo,features,od_labels)
        input_ids, segment_ids,input_mask,img_feat = self.tensorizer.tensorize_example(hypo,features)

        expl=example['explanation']
        expl = self.gpt_toker(expl,add_special_tokens=False)['input_ids']
        gpt_ids=expl
        gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]

        attn_mask=[1]*len(gpt_ids)
        gpt_ids=torch.tensor(gpt_ids)
        attn_mask=torch.tensor(attn_mask)

        # outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask))
        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)

        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask}
        batch = move_to_cuda(batch)
        return batch

class CaptionTensorizer_no_img(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=100,
            max_seq_a_length=40,
            is_train=True):
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


    def tensorize_example(self, text_a,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):


        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len=len(tokens)
        input_mask = [1] * seq_a_len

        padding_a_len = self.max_seq_a_len - len(tokens)
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        input_mask = input_mask + ([0] * padding_a_len)




        seq_len = len(tokens)
        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask + ([0] * padding_len)
        assert len(input_ids) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len

        # image features
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        return input_ids, segment_ids,input_mask

class SNLIGPT_gen_no_img_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,max_caption_len=40,
            is_train=True):
        self.is_train=is_train
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_no_img(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.is_train = is_train

    def get_valid_tsv(self):
        # based on the order of file size
        if self.label_tsv:
            return self.label_tsv
        if self.feat_tsv:
            return self.feat_tsv


    def read_csv_annot(self,SNLI_annot_file,img_dict):
        f=open(SNLI_annot_file,'r')
        csvreader=csv.reader(f)
        result = {}
        idx=0
        for item in csvreader:
            # 忽略第一行
            try:
                if csvreader.line_num == 1:
                    continue
                # if item[2].split('.')[0] in img_dict.keys():
                result[idx] = {}
                result[idx]['pairID']=item[1]
                result[idx]['Flickr30kID'] = item[2]
                result[idx]['hypothesis'] = item[3]
                result[idx]['gold_label']=item[4]
                result[idx]['explanation'] = item[5]
                idx+=1
            except:
                None
                #因为会自动读到最后一行空行
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        flickrID=example['Flickr30kID'].split('.')[0]
        hypo=example['hypothesis']
        input_ids, segment_ids,input_mask = self.tensorizer.tensorize_example(hypo)
        expl=example['explanation']
        expl = self.gpt_toker(expl,add_special_tokens=False)['input_ids']
        gpt_ids=expl
        gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
        attn_mask=[1]*len(gpt_ids)
        gpt_ids=torch.tensor(gpt_ids)
        attn_mask=torch.tensor(attn_mask)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,gpt_ids,attn_mask)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,gpt_ids,attn_mask) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'expl_ids':gpt_ids,'attn_mask':attn_mask}
        batch = move_to_cuda(batch)
        return batch

class SNLIGPT_gen_eval_all_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,max_caption_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.max_caption_len = max_caption_len
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        pairID=example['pairID']
        cap_index=pairID.split('#')[1][0]
        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])

        od_labels=self.bert_toker.encode(od_labels,add_special_tokens=False)
        #将假设、object、图片拼接，用于判断假设是否正确
        # input_ids, segment_ids,input_mask,img_feat = self.tensorizer.tensorize_example(hypo,features,od_labels)
        input_ids, segment_ids,input_mask,img_feat = self.tensorizer.tensorize_example(hypo,features)

        expl=example['explanation']
        expl = self.gpt_toker(expl,add_special_tokens=False)['input_ids']
        gpt_ids=expl
        gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]


        attn_mask=[1]*len(gpt_ids)
        gpt_ids=torch.tensor(gpt_ids)
        attn_mask=torch.tensor(attn_mask)
        od_labels=torch.tensor(od_labels)
        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,od_labels)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,od_labels) = map(list, unzip(concat(inputs)))
        od_labels = pad_sequence(od_labels, batch_first=True, padding_value=0)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'od_labels':od_labels}
        batch = move_to_cuda(batch)
        return batch

class SNLIGPT_gen_sep_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,max_caption_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.max_caption_len = max_caption_len
        self.max_expl_len=50
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

        save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
        with open(os.path.join(save_path, 'reelation_train.json'), 'r') as f:
            result = json.load(f)
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        examples=self.SNLI_annot_dict[str(i)]
        outputs=[]
        for example in examples:
            pairID=example['pairID']
            cap_index=pairID.split('#')[1][0]
            flickrID=example['Flickr30kID'].split('.')[0]
            img_idx=self.image2id[flickrID]
            hypo=example['hypothesis']
            label=torch.tensor(self.label2id[example['gold_label']])
            features = self.get_image_features(img_idx)
            od_labels,num= self.get_od_labels(img_idx)
            features=features[:num]
            # sup_infp=caption+od_labels
            #将假设、object、图片拼接，用于判断假设是否正确
            input_ids, segment_ids,input_mask,img_feat = self.tensorizer.tensorize_example(hypo,features,od_labels)
            caption ="<|b_cap|> "+ self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)]+ " <|e_cap|>"
            expl="<|b_exp|> "+example['explanation']+' <|e_exp|>'+" <classifier>"

            caption = self.gpt_toker(caption,add_special_tokens=False)['input_ids']
            expl = self.gpt_toker(expl,add_special_tokens=False)['input_ids']
            gpt_ids=expl
            # if len(gpt_ids)>self.max_expl_len-2:
            #     gpt_ids=gpt_ids[:self.max_expl_len-2]

            gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
            # pad_len=self.max_expl_len-len(gpt_ids)
            attn_mask = [1] * len(gpt_ids)
            # gpt_ids+=[self.gpt_toker.pad_token_id]*pad_len
            # attn_mask+=[0] *pad_len
            if len(caption)>self.max_caption_len-2:
                caption=caption[:self.max_caption_len-2]
            caption=[self.gpt_toker.bos_token_id]+caption+[self.gpt_toker.eos_token_id]
            attn_mask_cap = [1] * len(caption)
            attn_mask_cap += [0] * (self.max_caption_len - len(caption))
            caption += [self.gpt_toker.pad_token_id] * (self.max_caption_len - len(caption))


            gpt_ids=torch.tensor(gpt_ids)
            attn_mask=torch.tensor(attn_mask)
            caption = torch.tensor(caption)
            attn_mask_cap = torch.tensor(attn_mask_cap)
            outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,caption,attn_mask_cap))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,caption,attn_mask_cap) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        attn_mask_cap = pad_sequence(attn_mask_cap, batch_first=True, padding_value=0)
        caption = pad_sequence(caption, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'cap_ids':caption,'attn_mask_cap':attn_mask_cap}
        batch = move_to_cuda(batch)
        return batch

class SNLIGPT_gen_sep_eval_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,max_caption_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.max_caption_len = max_caption_len
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
                if item[2].split('.')[0] in img_dict.keys():
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        pairID=example['pairID']
        cap_index=pairID.split('#')[1][0]
        flickrID=example['Flickr30kID'].split('.')[0]
        img_idx=self.image2id[flickrID]
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        features = self.get_image_features(img_idx)
        od_labels,num= self.get_od_labels(img_idx)
        features=features[:num]
        # sup_infp=caption+od_labels
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat = self.tensorizer.tensorize_example(hypo,features,od_labels)
        caption ="<|b_cap|> "+ self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)]+ " <|e_cap|>"
        expl="<|b_exp|> "+example['explanation']+' <|e_exp|>'+" <classifier>"

        caption = self.gpt_toker(caption,add_special_tokens=False)['input_ids']
        expl = self.gpt_toker(expl,add_special_tokens=False)['input_ids']
        gpt_ids=expl
        gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
        if len(caption)>self.max_caption_len-2:
            caption=caption[:self.max_caption_len-2]
        caption=[self.gpt_toker.bos_token_id]+caption+[self.gpt_toker.eos_token_id]
        attn_mask_cap = [1] * len(caption)
        attn_mask_cap += [0] * (self.max_caption_len - len(caption))
        caption += [self.gpt_toker.pad_token_id] * (self.max_caption_len - len(caption))

        attn_mask=[1]*len(gpt_ids)
        gpt_ids=torch.tensor(gpt_ids)
        attn_mask=torch.tensor(attn_mask)
        caption = torch.tensor(caption)
        attn_mask_cap = torch.tensor(attn_mask_cap)
        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,caption,attn_mask_cap)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,caption,attn_mask_cap) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        attn_mask_cap = pad_sequence(attn_mask_cap, batch_first=True, padding_value=0)
        caption = pad_sequence(caption, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'cap_ids':caption,'attn_mask_cap':attn_mask_cap}
        batch = move_to_cuda(batch)
        return batch

class CaptionTensorizer_add_cls(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=100,
            max_seq_a_length=40,
            is_train=True):
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

    def tensorize_example(self, text_a, img_feat, text_b,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):


        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len=len(tokens)
        input_mask = [1] * seq_a_len
        region_id=[0]* seq_a_len
        padding_a_len = self.max_seq_a_len - len(tokens)
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        input_mask = input_mask + ([0] * padding_a_len)


        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len- len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            input_mask = input_mask + ([1] * (len(tokens_b) + 1))
        region_b_ids=self.tokenizer(text_b).encodings[0].word_ids[1:-1][:50]
        inter_list = list(set(tokens_a).intersection(set(tokens_b)))
        for i in range(len(tokens_a)):
            if tokens_a[i] in inter_list:
                index=tokens_b.index(tokens_a[i])
                #注意region前面有cls,且图片从0开始编号
                region_id[i+1]=region_b_ids[index]+1


        seq_len = len(tokens)
        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask + ([0] * padding_len)
        assert len(input_ids) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len

        # image features
        img_len = img_feat.shape[0]

        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
            input_mask = input_mask + ([1] * img_len)
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            input_mask = input_mask + ([1] * img_len)
            input_mask = input_mask + ([0] * (self.max_img_seq_len - img_len))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        region_id = torch.tensor(region_id, dtype=torch.long)
        region_b_ids = torch.tensor(region_b_ids, dtype=torch.long)
        return input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids

class CaptionTensorizer_obj_contras(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=100,
            max_seq_a_length=40,
            is_train=True):
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

    def tensorize_example(self, text_a, img_feat, text_b,gpt_pad_token_id,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):


        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len=len(tokens)
        input_mask = [1] * seq_a_len
        region_id=[gpt_pad_token_id]* self.max_seq_a_len
        padding_a_len = self.max_seq_a_len - len(tokens)
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        input_mask = input_mask + ([0] * padding_a_len)


        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len- len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            input_mask = input_mask + ([1] * (len(tokens_b) + 1))
        region_b_ids=self.tokenizer(text_b).encodings[0].word_ids[1:-1][:50]
        inter_list = list(set(tokens_a).intersection(set(tokens_b)))
        for i in range(len(tokens_a)):
            if tokens_a[i] in inter_list:
                index=tokens_b.index(tokens_a[i])
                #注意region前面有cls
                region_id[i+1]=region_b_ids[index]


        seq_len = len(tokens)
        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask + ([0] * padding_len)
        region_b_ids+=[gpt_pad_token_id] * (self.max_seq_len-self.max_seq_a_len-len(region_b_ids))
        assert len(input_ids) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len

        # image features
        img_len = img_feat.shape[0]

        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
            input_mask = input_mask + ([1] * img_len)
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            input_mask = input_mask + ([1] * img_len)
            input_mask = input_mask + ([0] * (self.max_img_seq_len - img_len))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        region_id = torch.tensor(region_id, dtype=torch.long)
        region_b_ids = torch.tensor(region_b_ids, dtype=torch.long)
        return input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids

class CaptionTensorizer_align(object):
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

    def read_synonym(self,synonym_file):
        with open(synonym_file,'r') as f:
            synonym_dict=json.load(f)
        return synonym_dict

    def tensorize_example(self, text_a, img_feat, text_b,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):


        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len=len(tokens)
        input_mask = [1] * seq_a_len
        region_id=[0]* seq_a_len
        padding_a_len = self.max_seq_a_len - len(tokens)
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        input_mask = input_mask + ([0] * padding_a_len)


        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len- self.max_seq_a_len - 1:
                tokens_b = tokens_b[: (self.max_seq_len - self.max_seq_a_len - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            input_mask = input_mask + ([1] * (len(tokens_b) + 1))
        #用来做obj label和img的对齐
        region_b_ids=self.tokenizer(text_b).encodings[0].word_ids[1:-1][:(self.max_seq_len - self.max_seq_a_len - 1)]
        # #未加入同义词替换
        # inter_list = list(set(tokens_a).intersection(set(tokens_b)))
        # for i in range(len(tokens_a)):
        #     if tokens_a[i] in inter_list:
        #         index=tokens_b.index(tokens_a[i])
        #         #注意region前面有cls
        #         region_id[i+1]=region_b_ids[index]
        #加入同义词替换
        for i in range(len(tokens_a)):
            if tokens_a[i] in self.synonym_dict.keys():
                syn_list = self.synonym_dict[tokens_a[i]]
            else:
                syn_list=[tokens_a[i]]
            inter_list = list(set(syn_list).intersection(set(tokens_b)))
            if len(inter_list)>0:
                #只要有对应的，就取第一个对应上的img object
                index =region_b_ids[tokens_b.index(inter_list[0])]
                # 注意region前面有cls,且图片从0开始编号
                region_id[i + 1] = index + 1

        seq_len = len(tokens)
        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask + ([0] * padding_len)
        assert len(input_ids) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len

        # image features
        img_len = img_feat.shape[0]

        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
            input_mask = input_mask + ([1] * img_len)
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            input_mask = input_mask + ([1] * img_len)
            input_mask = input_mask + ([0] * (self.max_img_seq_len - img_len))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        region_id = torch.tensor(region_id, dtype=torch.long)
        region_b_ids = torch.tensor(region_b_ids, dtype=torch.long)

        return input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids

class CaptionTensorizer_align_obj_contras(object):
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

    def read_synonym(self,synonym_file):
        with open(synonym_file,'r') as f:
            synonym_dict=json.load(f)
        return synonym_dict

    def tensorize_example(self, text_a, img_feat, text_b,object,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):


        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len=len(tokens)
        input_mask = [1] * seq_a_len
        region_id=[0]* seq_a_len
        padding_a_len = self.max_seq_a_len - len(tokens)
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        input_mask = input_mask + ([0] * padding_a_len)


        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len- self.max_seq_a_len - 1:
                tokens_b = tokens_b[: (self.max_seq_len - self.max_seq_a_len - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            input_mask = input_mask + ([1] * (len(tokens_b) + 1))
        #用来做obj label和img的对齐
        region_b_ids=self.tokenizer(text_b).encodings[0].word_ids[1:-1][:(self.max_seq_len - self.max_seq_a_len - 1)]

        #在hypothesis中定位object
        hypo_object_mask=torch.zeros(self.max_seq_a_len,dtype=torch.long).cuda()
        try:
            hypo_object_index = tokens_a.index(object)+1
            #+1是因为CLS
        except:
            #没有的话就取CLS
            hypo_object_index=0
        hypo_object_mask[hypo_object_index]=1
        object=object.lower()
        syn_list=self.synonym_dict[object]
        inter_list = list(set(syn_list).intersection(set(tokens_b)))
        img_object_mask = torch.zeros(self.max_img_seq_len, dtype=torch.long).cuda()
        if len(inter_list)>0:
            img_object_index = tokens_b.index(inter_list[0])
        else:
            img_object_index=0
        img_object_mask[region_b_ids[img_object_index]]=1
        for i in range(len(tokens_a)):
            if tokens_a[i] in self.synonym_dict.keys():
                syn_list = self.synonym_dict[tokens_a[i]]
            else:
                syn_list=[tokens_a[i]]
            inter_list = list(set(syn_list).intersection(set(tokens_b)))
            if len(inter_list)>0:
                #只要有对应的，就取第一个对应上的img object
                index =region_b_ids[tokens_b.index(inter_list[0])]
                # 注意region前面有cls,且图片从0开始编号
                region_id[i + 1] = index + 1

        seq_len = len(tokens)
        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask + ([0] * padding_len)
        assert len(input_ids) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len

        # image features
        img_len = img_feat.shape[0]

        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
            input_mask = input_mask + ([1] * img_len)
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            input_mask = input_mask + ([1] * img_len)
            input_mask = input_mask + ([0] * (self.max_img_seq_len - img_len))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        region_id = torch.tensor(region_id, dtype=torch.long)
        region_b_ids = torch.tensor(region_b_ids, dtype=torch.long)
        # hypo_object_index=torch.tensor(hypo_object_index, dtype=torch.long)
        # img_object_index=torch.tensor(img_object_index, dtype=torch.long)
        return input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids,hypo_object_mask,img_object_mask

class SNLIGPT_gen_add_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=90, max_seq_a_length=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50

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
        f=open(SNLI_annot_file,'r')
        csvreader=csv.reader(f)
        result = {}
        idx=0
        for item in csvreader:
            # 忽略第一行
            try:
                if csvreader.line_num == 1:
                    continue
                if item[2].split('.')[0] in img_dict.keys():
                    result[idx] = {}
                    result[idx]['pairID']=item[1]
                    result[idx]['Flickr30kID'] = item[2]
                    result[idx]['hypothesis'] = item[3]
                    result[idx]['gold_label']=item[4]
                    result[idx]['explanation'] = item[5]
                    idx+=1
            except:
                None
                #因为会自动读到最后一行空行
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        pairID=example['pairID']
        cap_index=pairID.split('#')[1][0]
        flickrID=example['Flickr30kID'].split('.')[0]
        img_idx=self.image2id[flickrID]
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        features = self.get_image_features(img_idx)
        od_labels,num= self.get_od_labels(img_idx)
        features=features[:num]
        # sup_infp=caption+od_labels
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat,region_id = self.tensorizer.tensorize_example(hypo,features,od_labels)
        caption ="<|b_cap|> "+ self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)]+ " <|e_cap|>"
        expl="<|b_exp|> "+example['explanation']+' <|e_exp|>'
        cls="<|b_cls|> "+self.cls_template+' '+example['gold_label']+'.'+' <|e_cls|>'
        caption = self.gpt_toker(caption)['input_ids']
        gpt_martrix_cap = [1] * len(caption)
        gpt_martrix_cls = [0] * len(caption)
        gpt_martrix_expl = [0] * len(caption)
        cls=self.gpt_toker(cls)['input_ids']
        gpt_martrix_cap += [0] * len(cls)
        gpt_martrix_cls += [1] * len(cls)
        gpt_martrix_expl += [0] * len(cls)
        expl = self.gpt_toker(expl)['input_ids']
        gpt_ids=caption+cls+expl
        gpt_martrix_cap += [0] * len(expl)
        gpt_martrix_cls += [0] * len(expl)
        gpt_martrix_expl += [1] * len(expl)
        gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
        gpt_martrix_cap = [0] + gpt_martrix_cap + [0]
        gpt_martrix_cls=[0]+gpt_martrix_cls+[0]
        gpt_martrix_expl = [0] + gpt_martrix_expl + [1]
        attn_mask=[1]*len(gpt_ids)
        gpt_ids=torch.tensor(gpt_ids)
        attn_mask=torch.tensor(attn_mask)
        gpt_martrix_cap = torch.tensor(gpt_martrix_cap)
        gpt_martrix_cls=torch.tensor(gpt_martrix_cls)
        gpt_martrix_expl = torch.tensor(gpt_martrix_expl)
        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,gpt_martrix_cls,gpt_martrix_expl,gpt_martrix_cap,region_id)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,gpt_martrix_cls,gpt_martrix_expl,gpt_martrix_cap,region_id) = map(list, unzip(concat(inputs)))
        gpt_martrix_cap = pad_sequence(gpt_martrix_cap, batch_first=True, padding_value=0)
        gpt_martrix_cls = pad_sequence(gpt_martrix_cls, batch_first=True, padding_value=0)
        gpt_martrix_expl = pad_sequence(gpt_martrix_expl, batch_first=True, padding_value=0)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'gpt_martrix_expl':gpt_martrix_expl,'gpt_martrix_cls':gpt_martrix_cls,'gpt_martrix_cap':gpt_martrix_cap,'region_id':region_id
            }
        batch = move_to_cuda(batch)
        return batch

class SNLIGPT_gen_wo_cap_align_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
        with open(os.path.join(save_path, 'reelation_train.json'), 'r') as f:
            result = json.load(f)
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        examples = self.SNLI_annot_dict[str(i)]
        outputs = []
        for example in examples:
            pairID=example['pairID']

            flickrID=example['Flickr30kID'].split('.')[0]
            img_idx=self.image2id[flickrID]
            hypo=example['hypothesis']
            label=torch.tensor(self.label2id[example['gold_label']])
            features = self.get_image_features(img_idx)
            od_labels,num= self.get_od_labels(img_idx)
            features=features[:num]
            #将假设、object、图片拼接，用于判断假设是否正确
            input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
            expl=example['explanation']
            expl = self.gpt_toker(expl)['input_ids']
            gpt_ids=expl
            gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
            attn_mask=[1]*len(gpt_ids)
            gpt_ids=torch.tensor(gpt_ids)
            attn_mask=torch.tensor(attn_mask)
            outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch
    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_gen_wo_cap_align_eval_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]


        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
        expl=example['explanation']
        expl = self.gpt_toker(expl)['input_ids']
        gpt_ids =expl
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids)
        attn_mask = torch.tensor(attn_mask)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_gen_align_objects_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,synonym_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_align(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train,synonym_file=synonym_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example = self.SNLI_annot_dict[i]
        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx = self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir, 'box_feat', flickrID)
            features = torch.load(path)
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
            od_labels = ' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo = example['hypothesis']
        label = torch.tensor(self.label2id[example['gold_label']])
        input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
        expl=example['explanation']
        expl = self.gpt_toker(expl)['input_ids']
        gpt_ids=expl
        gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
        attn_mask=[1]*len(gpt_ids)
        gpt_ids=torch.tensor(gpt_ids)
        attn_mask=torch.tensor(attn_mask)
        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch
    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_gen_align_object_eval_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,synonym_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_align(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,synonym_file=synonym_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
        expl=example['explanation']
        expl = self.gpt_toker(expl)['input_ids']
        gpt_ids =expl
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids)
        attn_mask = torch.tensor(attn_mask)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_gen_align_object_contras_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,synonym_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_align_obj_contras(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train,synonym_file=synonym_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
        with open(os.path.join(save_path, 'object_contras.json'), 'r') as f:
            result = json.load(f)
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        object=self.SNLI_annot_dict[i]['object']
        examples = self.SNLI_annot_dict[i]['examples']
        outputs = []
        for example in examples:
            flickrID=example['Flickr30kID'].split('.')[0]
            if flickrID in self.image2id.keys():
                img_idx = self.image2id[flickrID]
                features = self.get_image_features(img_idx)
                od_labels, num = self.get_od_labels(img_idx)
                features = features[:num]
            else:
                path = os.path.join(self.feat_dir, 'box_feat', flickrID)
                features = torch.load(path)
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
                od_labels = ' '.join(od_labels)
                gt_rel_boxes = torch.tensor(gt_rel_boxes)
                gt_boxes = torch.tensor(gt_boxes)
                cat_tensor = torch.zeros((features.size(0), 6))
                cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
                cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
                cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
                cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
                cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
                cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
                features = torch.cat((features, cat_tensor), dim=-1)
            hypo = example['hypothesis']
            label = torch.tensor(self.label2id[example['gold_label']])
            input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids,hypo_object_mask,img_object_mask = self.tensorizer.tensorize_example(hypo,features,od_labels,object)
            expl=example['explanation']
            expl = self.gpt_toker(expl)['input_ids']
            gpt_ids=expl
            gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
            attn_mask=[1]*len(gpt_ids)
            gpt_ids=torch.tensor(gpt_ids)
            attn_mask=torch.tensor(attn_mask)
            outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids,hypo_object_mask,img_object_mask))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids,hypo_object_mask,img_object_mask) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        hypo_object_mask=torch.stack(hypo_object_mask,dim=0)
        img_object_mask=torch.stack(img_object_mask,dim=0)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids,'img_object_mask':img_object_mask,
                'hypo_object_mask':hypo_object_mask
            }
        batch = move_to_cuda(batch)
        return batch
    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_gen_align_object_contras_eval_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,synonym_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]

        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
        expl=example['explanation']
        expl = self.gpt_toker(expl)['input_ids']
        gpt_ids =expl
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids)
        attn_mask = torch.tensor(attn_mask)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_gen_cap_align_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
        with open(os.path.join(save_path, 'reelation_train.json'), 'r') as f:
            result = json.load(f)
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
        od_labels = None
        num=0
        rects=[]
        labels=[]
        scores=[]
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
                    rects.append(l['rect'])
                    labels.append(l['class'])
                    scores.append(l['conf'])
            od_labels=od_labels[:-1]
        return od_labels,num,rects,labels,scores

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        examples = self.SNLI_annot_dict[str(i)]
        outputs = []
        for example in examples:
            pairID=example['pairID']
            cap_index=pairID.split('#')[1][0]
            flickrID=example['Flickr30kID'].split('.')[0]
            img_idx=self.image2id[flickrID]
            hypo=example['hypothesis']
            label=torch.tensor(self.label2id['<'+example['gold_label']+'>'])
            features = self.get_image_features(img_idx)
            od_labels,num,rects,labels,scores= self.get_od_labels(img_idx)
            # cv2_img=cv2.imread(os.path.join('/raid/yq/e-ViL-main/flickr30k_images/flickr30k_images',example['Flickr30kID']))
            # draw_bb(cv2_img, rects, labels, scores)
            # cv2.imwrite(os.path.join('/raid/yq/e-ViL-main/flickr30k_images/bounding_box',example['Flickr30kID']), cv2_img)
            features=features[:num]
            #将假设、object、图片拼接，用于判断假设是否正确
            input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
            caption = self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)] +"<cap_exp>"
            expl=example['explanation']

            caption = self.gpt_toker(caption)['input_ids']
            expl = self.gpt_toker(expl)['input_ids']
            gpt_ids = caption +expl
            gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
            attn_mask=[1]*len(gpt_ids)
            gpt_ids=torch.tensor(gpt_ids)
            attn_mask=torch.tensor(attn_mask)
            outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids}
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_gen_cap_align_eval_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]

        pairID = example['pairID']
        cap_index = pairID.split('#')[1][0]
        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id['<'+example['gold_label']+">"])

        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
        caption = self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)] + "<cap_exp>"
        expl=example['explanation']
        caption = self.gpt_toker(caption)['input_ids']
        expl = self.gpt_toker(expl)['input_ids']
        gpt_ids = caption+expl
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids)
        attn_mask = torch.tensor(attn_mask)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_token_align_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
        with open(os.path.join(save_path, 'reelation_train.json'), 'r') as f:
            result = json.load(f)
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        examples = self.SNLI_annot_dict[str(i)]
        outputs = []
        for example in examples:
            pairID=example['pairID']
            cap_index=pairID.split('#')[1][0]
            flickrID = example['Flickr30kID'].split('.')[0]

            flickrID=example['Flickr30kID'].split('.')[0]
            img_idx=self.image2id[flickrID]
            hypo=example['hypothesis']
            label=torch.tensor(self.label2id['<'+example['gold_label']+'>'])
            features = self.get_image_features(img_idx)
            od_labels,num= self.get_od_labels(img_idx)
            features=features[:num]
            #将假设、object、图片拼接，用于判断假设是否正确
            input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
            caption =  self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)]
            expl=example['explanation']
            cls='<answer>'

            caption = self.gpt_toker(caption)['input_ids']
            gpt_martrix_cap = [1] * len(caption)
            gpt_martrix_cls = [0] * len(caption)
            gpt_martrix_expl = [0] * len(caption)
            cls = self.gpt_toker(cls)['input_ids']
            gpt_martrix_cap += [0] * len(cls)
            gpt_martrix_cls += [1] * len(cls)
            gpt_martrix_expl += [0] * len(cls)
            expl = self.gpt_toker(expl)['input_ids']
            gpt_ids = caption + cls + expl
            gpt_martrix_cap += [0] * len(expl)
            gpt_martrix_cls += [0] * len(expl)
            gpt_martrix_expl += [1] * len(expl)
            gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
            gpt_martrix_cap = [0] + gpt_martrix_cap + [0]
            gpt_martrix_cls=[0]+gpt_martrix_cls+[0]
            gpt_martrix_expl = [0] + gpt_martrix_expl + [1]
            attn_mask=[1]*len(gpt_ids)
            gpt_ids=torch.tensor(gpt_ids)
            attn_mask=torch.tensor(attn_mask)
            gpt_martrix_cap = torch.tensor(gpt_martrix_cap)
            gpt_martrix_cls=torch.tensor(gpt_martrix_cls)
            gpt_martrix_expl = torch.tensor(gpt_martrix_expl)
            outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,gpt_martrix_cls,gpt_martrix_expl,gpt_martrix_cap,region_id,region_b_ids))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,gpt_martrix_cls,gpt_martrix_expl,gpt_martrix_cap,region_id,region_b_ids) = map(list, unzip(concat(inputs)))
        gpt_martrix_cls = pad_sequence(gpt_martrix_cls, batch_first=True, padding_value=0)
        gpt_martrix_expl = pad_sequence(gpt_martrix_expl, batch_first=True, padding_value=0)
        gpt_martrix_cap = pad_sequence(gpt_martrix_cap, batch_first=True, padding_value=0)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'gpt_martrix_expl':gpt_martrix_expl,'gpt_martrix_cls':gpt_martrix_cls,'gpt_martrix_cap':gpt_martrix_cap,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_token_align_eval_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]

        pairID = example['pairID']
        cap_index = pairID.split('#')[1][0]
        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id['<'+example['gold_label']+">"])

        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
        caption =  self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)]
        expl=example['explanation']
        cls ='<answer>'
        caption = self.gpt_toker(caption)['input_ids']
        cls = self.gpt_toker(cls)['input_ids']
        expl = self.gpt_toker(expl)['input_ids']
        gpt_ids = caption+cls + expl

        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]

        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids)
        attn_mask = torch.tensor(attn_mask)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_token_first_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.max_hypo_len=max_hypo_len
        self.max_seq_len=max_seq_length

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
        save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
        with open(os.path.join(save_path, 'reelation_train.json'), 'r') as f:
            result = json.load(f)
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        examples = self.SNLI_annot_dict[str(i)]
        outputs = []
        for example in examples:
            pairID=example['pairID']
            cap_index=pairID.split('#')[1][0]
            # flickrID = example['Flickr30kID'].split('.')[0]

            flickrID=example['Flickr30kID'].split('.')[0]
            img_idx=self.image2id[flickrID]
            hypo=example['hypothesis']
            label=torch.tensor(self.label2id['<'+example['gold_label']+'>'])
            features = self.get_image_features(img_idx)
            od_labels,num= self.get_od_labels(img_idx)
            features=features[:num]
            #将假设、object、图片拼接，用于判断假设是否正确
            input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
            caption =  self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)]+"<cap_exp>"
            expl=example['explanation']
            cls='<answer>'
            cls = self.gpt_toker(cls)['input_ids']
            caption = self.gpt_toker(caption)['input_ids']
            expl = self.gpt_toker(expl)['input_ids']
            gpt_ids =  cls+caption + expl
            gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
            attn_mask=[1]*len(gpt_ids)
            gpt_ids=torch.tensor(gpt_ids)
            attn_mask=torch.tensor(attn_mask)
            outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_token_first_eval_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]

        pairID = example['pairID']
        cap_index = pairID.split('#')[1][0]
        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id['<'+example['gold_label']+">"])

        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
        caption =self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)]+"<cap_exp>"
        expl=example['explanation']
        cls ='<answer>'
        caption = self.gpt_toker(caption)['input_ids']
        cls = self.gpt_toker(cls)['input_ids']
        expl = self.gpt_toker(expl)['input_ids']
        gpt_ids = cls+caption + expl

        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]

        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids)
        attn_mask = torch.tensor(attn_mask)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_token_first_wo_cap_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_obj_contras(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.max_hypo_len=max_hypo_len
        self.max_seq_len=max_seq_length

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
        save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
        with open(os.path.join(save_path, 'reelation_train.json'), 'r') as f:
            result = json.load(f)
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        examples = self.SNLI_annot_dict[str(i)]
        outputs = []
        for example in examples:
            pairID=example['pairID']
            # cap_index=pairID.split('#')[1][0]
            # flickrID = example['Flickr30kID'].split('.')[0]

            flickrID=example['Flickr30kID'].split('.')[0]
            img_idx=self.image2id[flickrID]
            hypo=example['hypothesis']
            label=torch.tensor(self.label2id['<'+example['gold_label']+'>'])
            features = self.get_image_features(img_idx)
            od_labels,num= self.get_od_labels(img_idx)
            features=features[:num]
            #将假设、object、图片拼接，用于判断假设是否正确
            input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels,self.gpt_toker.pad_token_id)
            # pad_len=self.max_seq_len-self.max_hypo_len-region_b_ids.size(0)
            # pad_tensor=torch.full((1,pad_len),self.gpt_toker.pad_token_id).squeeze()
            # region_b_ids=torch.cat((region_b_ids,pad_tensor),dim=0)
            expl=example['explanation']
            cls='<answer>'
            cls = self.gpt_toker(cls)['input_ids']

            expl = self.gpt_toker(expl)['input_ids']
            gpt_ids =  cls+ expl
            gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
            attn_mask=[1]*len(gpt_ids)
            gpt_ids=torch.tensor(gpt_ids)
            attn_mask=torch.tensor(attn_mask)
            outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        # region_b_ids=self.get_gather_index(region_b_ids)
        region_b_ids = torch.stack(region_b_ids, dim=0)
        region_id = torch.stack(region_id, dim=0)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_token_first_wo_cap_eval_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]

        pairID = example['pairID']
        cap_index = pairID.split('#')[1][0]
        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id['<'+example['gold_label']+">"])

        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)

        expl=example['explanation']
        cls ='<answer>'

        cls = self.gpt_toker(cls)['input_ids']
        expl = self.gpt_toker(expl)['input_ids']
        gpt_ids = cls+ expl

        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]

        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids)
        attn_mask = torch.tensor(attn_mask)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_token_only_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.max_hypo_len=max_hypo_len
        self.max_seq_len=max_seq_length

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
        save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
        with open(os.path.join(save_path, 'reelation_train.json'), 'r') as f:
            result = json.load(f)
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        examples = self.SNLI_annot_dict[str(i)]
        outputs = []
        for example in examples:
            pairID=example['pairID']
            cap_index=pairID.split('#')[1][0]
            # flickrID = example['Flickr30kID'].split('.')[0]

            flickrID=example['Flickr30kID'].split('.')[0]
            img_idx=self.image2id[flickrID]
            hypo=example['hypothesis']
            label=torch.tensor(self.label2id['<'+example['gold_label']+'>'])
            features = self.get_image_features(img_idx)
            od_labels,num= self.get_od_labels(img_idx)
            features=features[:num]
            #将假设、object、图片拼接，用于判断假设是否正确
            input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
            # caption =  self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)]+"<cap_exp>"
            # expl=example['explanation']
            cls='<answer>'
            cls = self.gpt_toker(cls)['input_ids']
            # caption = self.gpt_toker(caption)['input_ids']
            # expl = self.gpt_toker(expl)['input_ids']
            gpt_ids =  cls
            gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
            attn_mask=[1]*len(gpt_ids)
            gpt_ids=torch.tensor(gpt_ids)
            attn_mask=torch.tensor(attn_mask)
            outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_token_only_eval_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
            is_train=True):
        self.is_train=is_train
        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_dir='/'.join(feat_file.split('/')[:-1])
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'<neutral>':0,
                       '<contradiction>':1,
                       '<entailment>':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_hypo_len,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]

        pairID = example['pairID']
        cap_index = pairID.split('#')[1][0]
        flickrID=example['Flickr30kID'].split('.')[0]
        if flickrID in self.image2id.keys():
            img_idx=self.image2id[flickrID]
            features = self.get_image_features(img_idx)
            od_labels, num = self.get_od_labels(img_idx)
            features = features[:num]
        else:
            path = os.path.join(self.feat_dir,'box_feat', flickrID)
            features = torch.load(path)
            file_name = flickrID + '.json'
            path = os.path.join(self.feat_dir,'pred', file_name)
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
            od_labels=' '.join(od_labels)
            gt_rel_boxes = torch.tensor(gt_rel_boxes)
            gt_boxes = torch.tensor(gt_boxes)
            cat_tensor = torch.zeros((features.size(0), 6))
            cat_tensor[:, 0] = torch.div(gt_boxes[:, 0], gt_rel_boxes[:, 2])
            cat_tensor[:, 1] = torch.div(gt_boxes[:, 1], gt_rel_boxes[:, 3])
            cat_tensor[:, 2] = torch.div(gt_boxes[:, 2], gt_rel_boxes[:, 2])
            cat_tensor[:, 3] = torch.div(gt_boxes[:, 3], gt_rel_boxes[:, 3])
            cat_tensor[:, 4] = torch.div(gt_rel_boxes[:, 2], gt_rel_boxes[:, 3])
            cat_tensor[:, 5] = torch.div(gt_rel_boxes[:, 3], gt_rel_boxes[:, 2])
            features = torch.cat((features, cat_tensor), dim=-1)
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id['<'+example['gold_label']+">"])
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
        cls ='<answer>'
        cls = self.gpt_toker(cls)['input_ids']
        gpt_ids = cls
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids)
        attn_mask = torch.tensor(attn_mask)

        return [(torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids)]

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,region_id,region_b_ids) = map(list, unzip(concat(inputs)))

        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        obj_len=self.max_seq_len-self.max_hypo_len
        gather_index = torch.arange(0, obj_len, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),obj_len,1024).clone()
        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class SNLIGPT_gen_add_align_Dataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=90, max_seq_a_length=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_add_cls(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50

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
        save_path = '/'.join(SNLI_annot_file.split('/')[:-1])
        with open(os.path.join(save_path, 'reelation_train.json'), 'r') as f:
            result = json.load(f)
        return result
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        examples = self.SNLI_annot_dict[str(i)]
        outputs = []
        for example in examples:
            pairID=example['pairID']
            cap_index=pairID.split('#')[1][0]
            flickrID=example['Flickr30kID'].split('.')[0]
            img_idx=self.image2id[flickrID]
            hypo=example['hypothesis']
            label=torch.tensor(self.label2id[example['gold_label']])
            features = self.get_image_features(img_idx)
            od_labels,num= self.get_od_labels(img_idx)
            features=features[:num]
            #将假设、object、图片拼接，用于判断假设是否正确
            input_ids, segment_ids,input_mask,img_feat,region_id,region_b_ids = self.tensorizer.tensorize_example(hypo,features,od_labels)
            caption ="<|b_cap|> "+ self.flickr_caption_dict[flickrID + '.jpg'][int(cap_index)]+ " <|e_cap|>"
            expl="<|b_exp|> "+example['explanation']+' <|e_exp|>'
            cls="<|b_cls|> "+self.cls_template+' '+example['gold_label']+'.'+' <|e_cls|>'
            caption = self.gpt_toker(caption)['input_ids']
            gpt_martrix_cap = [1] * len(caption)
            gpt_martrix_cls = [0] * len(caption)
            gpt_martrix_expl = [0] * len(caption)
            cls=self.gpt_toker(cls)['input_ids']
            gpt_martrix_cap += [0] * len(cls)
            gpt_martrix_cls += [1] * len(cls)
            gpt_martrix_expl += [0] * len(cls)
            expl = self.gpt_toker(expl)['input_ids']
            gpt_ids=caption+cls+expl
            gpt_martrix_cap += [0] * len(expl)
            gpt_martrix_cls += [0] * len(expl)
            gpt_martrix_expl += [1] * len(expl)
            gpt_ids=[self.gpt_toker.bos_token_id]+gpt_ids+[self.gpt_toker.eos_token_id]
            gpt_martrix_cap = [0] + gpt_martrix_cap + [0]
            gpt_martrix_cls=[0]+gpt_martrix_cls+[0]
            gpt_martrix_expl = [0] + gpt_martrix_expl + [0]
            attn_mask=[1]*len(gpt_ids)
            gpt_ids=torch.tensor(gpt_ids)
            attn_mask=torch.tensor(attn_mask)
            gpt_martrix_cap = torch.tensor(gpt_martrix_cap)
            gpt_martrix_cls=torch.tensor(gpt_martrix_cls)
            gpt_martrix_expl = torch.tensor(gpt_martrix_expl)
            outputs.append((torch.tensor(int(flickrID)),input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,gpt_martrix_cls,gpt_martrix_expl,gpt_martrix_cap,region_id,region_b_ids))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self,inputs):
        (img_id,input_ids, segment_ids,input_mask,img_feat,label,gpt_ids,attn_mask,gpt_martrix_cls,gpt_martrix_expl,gpt_martrix_cap,region_id,region_b_ids) = map(list, unzip(concat(inputs)))
        gpt_martrix_cap = pad_sequence(gpt_martrix_cap, batch_first=True, padding_value=0)
        gpt_martrix_cls = pad_sequence(gpt_martrix_cls, batch_first=True, padding_value=0)
        gpt_martrix_expl = pad_sequence(gpt_martrix_expl, batch_first=True, padding_value=0)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        region_b_ids=self.get_gather_index(region_b_ids)
        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'label':label,'expl_ids':gpt_ids,'attn_mask':attn_mask,'gpt_martrix_expl':gpt_martrix_expl,'gpt_martrix_cls':gpt_martrix_cls,'gpt_martrix_cap':gpt_martrix_cap,'region_id':region_id,'region_b_ids':region_b_ids
            }
        batch = move_to_cuda(batch)
        return batch

    def get_gather_index(self,region_b_ids):
        gather_index = torch.arange(0, 50, dtype=torch.long,).unsqueeze(1).unsqueeze(0).expand(len(region_b_ids),50,1024).clone()

        for i, region in enumerate(region_b_ids):
            gather_index[i,:len(region),:] = region.unsqueeze(1).expand(region.size(0),1024)
        return gather_index

class CaptionTensorizer_cls_gen(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=100,
            max_seq_a_length=40,
            is_train=True):
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
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len,
            self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, hypo, img_feat,caption, text_b,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0,sequence_b_segment_id=1):
        tokens_cap = self.tokenizer.tokenize(caption)
        if len(tokens_cap) > self.max_seq_a_len - 2:
            tokens_cap = tokens_cap[:(self.max_seq_a_len - 2)]

        tokens_hypo = self.tokenizer.tokenize(hypo)
        if len(tokens_hypo) > self.max_seq_a_len - 2:
            tokens_hypo = tokens_hypo[:(self.max_seq_a_len - 2)]

        tokens_cls = [self.tokenizer.cls_token] + tokens_hypo + [self.tokenizer.sep_token]
        segment_ids_cls = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_cls) - 1)
        seq_hypo_len=len(tokens_cls)
        input_mask = [1] * seq_hypo_len
        padding_a_len = self.max_seq_a_len - len(tokens_cls)
        tokens_cls += [self.tokenizer.pad_token] * padding_a_len
        segment_ids_cls += ([pad_token_segment_id] * padding_a_len)
        input_mask = input_mask + ([0] * padding_a_len)

        tokens_gen = [self.tokenizer.cls_token] + tokens_cap + [self.tokenizer.sep_token]
        segment_ids_gen = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_gen) - 1)
        seq_cap_len = len(tokens_gen)
        padding_a_len = self.max_seq_a_len -seq_cap_len
        tokens_gen += [self.tokenizer.pad_token] * padding_a_len
        segment_ids_gen += ([pad_token_segment_id] * padding_a_len)


        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len- len(tokens_gen) - 1:
                tokens_b = tokens_b[: (self.max_seq_len- len(tokens_gen) - 1)]
            tokens_cls += tokens_b + [self.tokenizer.sep_token]
            segment_ids_cls += [sequence_b_segment_id] * (len(tokens_b) + 1)
            input_mask = input_mask + ([1] * (len(tokens_b) + 1))

            tokens_gen += tokens_b + [self.tokenizer.sep_token]
            segment_ids_gen += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens_cls)
        # pad on the right for image captioning
        padding_len = self.max_seq_len- seq_len
        tokens_cls = tokens_cls + ([self.tokenizer.pad_token] * padding_len)
        segment_ids_cls += ([pad_token_segment_id] * padding_len)
        input_ids_cls = self.tokenizer.convert_tokens_to_ids(tokens_cls)
        input_mask = input_mask + ([0] * padding_len)

        tokens_gen = tokens_gen + ([self.tokenizer.pad_token] * padding_len)
        segment_ids_gen += ([pad_token_segment_id] * padding_len)
        input_ids_gen = self.tokenizer.convert_tokens_to_ids(tokens_gen)

        assert len(input_ids_cls) == self.max_seq_len
        assert len(segment_ids_cls) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
            input_mask = input_mask + ([1] * img_len)
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            input_mask = input_mask + ([1] * img_len)
            input_mask = input_mask + ([0] * (self.max_img_seq_len - img_len))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_cap_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start: c_end, c_start: c_end].copy_(self._triangle_mask[0: seq_cap_len, 0: seq_cap_len])
        # full attention for L-L, R-R
        attention_mask[l_start: l_end, l_start: l_end] = 1
        attention_mask[r_start: r_end, r_start: r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start: c_end, l_start: l_end] = 1
        attention_mask[c_start: c_end, r_start: r_end] = 1
        # full attention for L-R:
        attention_mask[l_start: l_end, r_start: r_end] = 1
        attention_mask[r_start: r_end, l_start: l_end] = 1

        input_ids_cls = torch.tensor(input_ids_cls, dtype=torch.long)
        segment_ids_cls = torch.tensor(segment_ids_cls, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        input_ids_gen = torch.tensor(input_ids_gen, dtype=torch.long)
        segment_ids_gen = torch.tensor(segment_ids_gen, dtype=torch.long)
        attention_mask_cross=attention_mask.clone()
        attention_mask_cross[c_start: c_end, c_start: c_end]=1

        input_mask_enc=torch.cat((input_mask,torch.zeros(self.max_seq_a_len)))
        input_mask_enc[len(input_mask):len(input_mask)+len(tokens_cap)+2]=1
        return input_ids_cls, segment_ids_cls,input_ids_gen,segment_ids_gen,input_mask,img_feat,attention_mask,attention_mask_cross,input_mask_enc

class SNLI_cap_GPTDataset(Dataset):
    def __init__(self,bert_tokenizer,GPT_tokenizer,SNLI_annot_file,flickr_caption_file,feat_file,max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,
            is_train=True,):

        self.flickr_caption_dict=self.read_csv_cap(flickr_caption_file)
        self.bert_toker=bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.feat_tsv = TSVFile(feat_file)
        self.label_tsv =TSVFile(feat_file[:-12]+'predictions.tsv')
        self.label2id={'neutral':0,
                       'contradiction':1,
                       'entailment':2}
        f=open(feat_file[:-12]+'imageid2idx.json','r')
        self.image2id=json.load(f)
        self.SNLI_annot_dict = self.read_csv_annot(SNLI_annot_file,self.image2id)
        self.tensorizer = CaptionTensorizer_cls_gen(self.bert_toker, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.nms_threshold=0.50
        self.max_expl_len=50

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
        f=open(SNLI_annot_file,'r')
        csvreader=csv.reader(f)
        result = {}
        idx=0
        for item in csvreader:
            # 忽略第一行
            try:
                if csvreader.line_num == 1:
                    continue
                if item[2].split('.')[0] in img_dict.keys():
                    result[idx] = {}
                    result[idx]['pairID']=item[1]
                    result[idx]['Flickr30kID'] = item[2]
                    result[idx]['hypothesis'] = item[3]
                    result[idx]['gold_label']=item[4]
                    result[idx]['explanation'] = item[5]
                    idx+=1
            except:
                None
                #因为会自动读到最后一行空行
        return result

    def read_csv_cap(self,flickr_caption_file):
        f=open(flickr_caption_file,'r')
        csvreader=csv.reader(f)
        result = {}
        for item in csvreader:
            # 忽略第一行
            if csvreader.line_num %5==2:
                infos=item[0].split('|')
                cap=infos[-1]
                if len(item)>1:
                    caps=','.join(i for i in item[1:])
                    cap=cap+','+caps
                result[infos[0]]=cap
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
        od_labels = None
        num=0
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1])
            od_labels=''
            for l in label_info['objects']:
                if l['conf']>self.nms_threshold:
                    od_labels+=l['class']+' '
                    num+=1
            od_labels=od_labels[:-1]
        return od_labels,num

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example=self.SNLI_annot_dict[i]
        flickrID=example['Flickr30kID'].split('.')[0]
        img_idx=self.image2id[flickrID]
        hypo=example['hypothesis']
        label=torch.tensor(self.label2id[example['gold_label']])
        features = self.get_image_features(img_idx)
        od_labels,num= self.get_od_labels(img_idx)
        features=features[:num]
        caption=self.flickr_caption_dict[flickrID+'.jpg']
        #将假设、object、图片拼接，用于判断假设是否正确
        input_ids_cls, segment_ids_cls,input_ids_gen,segment_ids_gen,input_mask,img_feat,attention_mask,attention_mask_cross,input_mask_enc= self.tensorizer.tensorize_example(hypo,features,caption,od_labels)
        expl=example['explanation']
        expl_ids=self.gpt_toker(expl)['input_ids']
        if len(expl_ids)>self.max_expl_len-2:
            expl_ids=expl_ids[:self.max_expl_len-2]
        expl_ids=[self.gpt_toker.bos_token_id]+expl_ids+[self.gpt_toker.eos_token_id]
        pad_list=[self.gpt_toker.pad_token_id]*(self.max_expl_len-len(expl_ids))
        attn_mask=[1]*len(expl_ids)
        attn_mask += [0] * (self.max_expl_len - len(expl_ids))
        expl_ids+=pad_list
        expl_ids=torch.tensor(expl_ids)
        attn_mask=torch.tensor(attn_mask)
        return (torch.tensor(int(flickrID)),input_ids_cls, segment_ids_cls,input_ids_gen,segment_ids_gen,input_mask,img_feat,attention_mask,attention_mask_cross,input_mask_enc,label,expl_ids,attn_mask)