import random
import os
from torch.utils.data import Dataset
import torch
import csv
from toolz.sandbox import unzip
from cytoolz import concat
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from .data import (get_ids_and_lens, pad_tensors,
                   get_gather_index)
from oscar.utils.tsv_file import TSVFile
from oscar.utils.misc import (mkdir, set_seed,
        load_from_yaml_file, find_file_path_in_yaml)
import base64
# import spacy
# nlp = spacy.load('en_core_web_sm')


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

    def tensorize_example(self,text_a, img_feat, text_b,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        input_mask = [1] * seq_a_len

        padding_a_len = self.max_seq_a_len - len(tokens)
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)
        input_mask = input_mask + ([0] * padding_a_len)

        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
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
        return input_ids, segment_ids, input_mask, img_feat

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

class VQADataset(Dataset):
    def __init__(self,tokenizer,vqa_train_file,vqa_val_file,vqa_ques_train,vqa_ques_val,explanation_file,train_yaml_file,val_yaml_file,test_yaml_file,max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40,max_caption_len=50,
            is_train=True):


        self.explanation_dict=self.read_explanation(explanation_file,vqa_train_file,vqa_val_file,vqa_ques_train,vqa_ques_val)
        self.root = os.path.dirname(train_yaml_file)

        self.yaml_file = train_yaml_file
        self.train_cfg = load_from_yaml_file(train_yaml_file)
        self.val_cfg = load_from_yaml_file(val_yaml_file)
        self.test_cfg = load_from_yaml_file(test_yaml_file)

        self.train_label_file = find_file_path_in_yaml(self.train_cfg['label'], self.root)
        self.train_feat_file = find_file_path_in_yaml(self.train_cfg['feature'], self.root)
        self.train_caption_file = find_file_path_in_yaml(self.train_cfg.get('caption'), self.root)

        self.val_label_file = find_file_path_in_yaml(self.val_cfg['label'], self.root)
        self.val_feat_file = find_file_path_in_yaml(self.val_cfg['feature'], self.root)
        self.val_caption_file = find_file_path_in_yaml(self.val_cfg.get('caption'), self.root)

        self.test_label_file = find_file_path_in_yaml(self.test_cfg['label'], self.root)
        self.test_feat_file = find_file_path_in_yaml(self.test_cfg['feature'], self.root)
        self.test_caption_file = find_file_path_in_yaml(self.test_cfg.get('caption'), self.root)


        self.train_label_tsv = TSVFile(self.train_label_file)
        self.train_feat_tsv = TSVFile(self.train_feat_file)

        self.val_label_tsv = TSVFile(self.val_label_file)
        self.val_feat_tsv = TSVFile(self.val_feat_file)

        self.test_label_tsv = TSVFile(self.test_label_file)
        self.test_feat_tsv = TSVFile(self.test_feat_file)


        with open(self.train_caption_file, 'r') as f:
            self.train_captions = json.load(f)
        with open(self.val_caption_file, 'r') as f:
            self.val_captions = json.load(f)
        with open(self.test_caption_file, 'r') as f:
            self.test_captions = json.load(f)

        self.tokenizer = tokenizer

        self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length,
                                            max_seq_length, max_seq_a_length,
                                            is_train=is_train)
        self.add_od_labels = True
        self.is_train = is_train
        self.train_image_keys = self.prepare_image_keys('train')
        self.train_key2index = self.prepare_image_key_to_index('train')
        self.train_key2captions = self.prepare_image_key_to_captions('train')

        self.val_image_keys = self.prepare_image_keys('val')
        self.val_key2index = self.prepare_image_key_to_index('val')
        self.val_key2captions = self.prepare_image_key_to_captions('val')

        self.test_image_keys = self.prepare_image_keys('test')
        self.test_key2index = self.prepare_image_key_to_index('test')
        self.test_key2captions = self.prepare_image_key_to_captions('test')

        self.nms_threshold=0.50
        self.max_caption_len=max_caption_len

    def read_explanation(self,file,vqa_train_file,vqa_val_file,vqa_ques_train,vqa_ques_val):
        vqa_annot = []
        with open(vqa_train_file, 'r') as f:
            tmp_data = json.load(f)
            vqa_annot.extend(tmp_data['annotations'])
        with open(vqa_val_file, 'r') as f:
            tmp_data = json.load(f)
            vqa_annot.extend(tmp_data['annotations'])
        vqa_ques = []
        with open(vqa_ques_train, 'r') as f:
            tmp_data = json.load(f)
            vqa_ques.extend(tmp_data['questions'])
        with open(vqa_ques_val, 'r') as f:
            tmp_data = json.load(f)
            vqa_ques.extend(tmp_data['questions'])

        with open(file,'r') as f:
            data=json.load(f)
        qa_exp=[]
        for i in range(len(vqa_annot)):
            qa=vqa_annot[i]
            if str(qa['question_id']) in data.keys():
                tmp_res=qa
                tmp_res['question']=vqa_ques[i]['question']
                tmp_res['explanation']=data[str(qa['question_id'])]
                qa_exp.append(tmp_res)
        return qa_exp


    def get_valid_tsv(self,flag):
        # based on the order of file size
        if flag=='train':
            if self.train_label_tsv:
                return self.train_label_tsv
            if self.train_feat_tsv:
                return self.train_feat_tsv
        elif flag=='val':
            if self.val_label_tsv:
                return self.val_label_tsv
            if self.val_feat_tsv:
                return self.val_feat_tsv
        else:
            if self.test_label_tsv:
                return self.test_label_tsv
            if self.test_feat_tsv:
                return self.test_feat_tsv


    def prepare_image_keys(self,flag):
        tsv = self.get_valid_tsv(flag)
        return [tsv.seek(i)[0] for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self,flag):
        tsv = self.get_valid_tsv(flag)
        return {tsv.seek(i)[0] : i for i in range(tsv.num_rows())}

    def prepare_image_key_to_captions(self,flag):
        if flag == 'train':
            key2captions = {key: [] for key in self.train_image_keys}
            for cap in self.train_captions:
                key2captions[cap['image_id']].append(cap['caption'])
            return key2captions
        elif flag == 'val':
            key2captions = {key: [] for key in self.val_image_keys}
            for cap in self.val_captions:
                key2captions[cap['image_id']].append(cap['caption'])
            return key2captions
        else:
            key2captions = {key: [] for key in self.test_image_keys}
            for cap in self.test_captions:
                key2captions[cap['image_id']].append(cap['caption'])
            return key2captions

    def get_image_index(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            img_key = img_cap_pair['image_id']
            return self.key2index[img_key]
        return idx

    def get_image_key(self, idx):
        img_idx = self.get_image_index(idx)
        return self.image_keys[img_idx]

    def get_image_features(self, img_idx,flag):
        if flag=='train':
            feat_info = json.loads(self.train_feat_tsv.seek(img_idx)[1])
        elif flag=='val':
            feat_info = json.loads(self.val_feat_tsv.seek(img_idx)[1])
        else:
            feat_info = json.loads(self.test_feat_tsv.seek(img_idx)[1])
        num_boxes = feat_info['num_boxes']
        features = np.frombuffer(base64.b64decode(feat_info['features']), np.float32
                ).reshape((num_boxes, -1))
        return torch.Tensor(features)

    def get_caption(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            return img_cap_pair['caption']
        return ""

    def get_od_labels(self, img_idx,flag):
        od_labels = None
        if flag=='train':
            label_info = json.loads(self.train_label_tsv.seek(img_idx)[1])
        elif flag=='val':
            label_info = json.loads(self.val_label_tsv.seek(img_idx)[1])
        else:
            label_info = json.loads(self.test_label_tsv.seek(img_idx)[1])
        od_labels = " ".join([l['class'] for l in label_info])
        return od_labels

    def get_caption_file_in_coco_format(self):
        cap_file = os.path.splitext(self.caption_file)[0] + '_coco_format.json'
        return cap_file

    def get_captions_by_key(self, key,flag):
        if flag=='train':
            return self.train_key2captions[key]
        elif flag=='val':
            return self.val_key2captions[key]
        else:
            return self.test_key2captions[key]

    def __len__(self):
        return len(self.explanation_dict)

    def __getitem__(self, idx):
        example=self.explanation_dict[idx]
        question=example['question']
        image_key=example['image_id']
        if str(image_key) in self.train_key2index.keys():
            img_idx = self.train_key2index[str(image_key)]
            flag='train'
        elif str(image_key) in self.val_key2index.keys():
            img_idx = self.val_key2index[str(image_key)]
            flag='val'
        else:
            img_idx = self.test_key2index[str(image_key)]
            flag='test'
        features = self.get_image_features(img_idx,flag)
        captions = self.get_captions_by_key(str(image_key),flag)
        caption=random.choice(captions)
        od_labels = self.get_od_labels(img_idx,flag)
        input_ids, segment_ids,input_mask,img_feat = self.tensorizer.tensorize_example(question, features, text_b=od_labels)
        caption = "<|b_cap|> " + caption+ " <|e_cap|>"
        ans= "<|b_ans|> " + example['multiple_choice_answer'] + ' <|e_ans|>'
        expl = "<|b_exp|> " + example['explanation'][0] + ' <|e_exp|>'
        caption = self.tokenizer(caption, add_special_tokens=False)['input_ids']
        expl = self.tokenizer(expl, add_special_tokens=False)['input_ids']
        ans = self.tokenizer(ans, add_special_tokens=False)['input_ids']
        gpt_ids = ans+expl
        gpt_ids = [self.tokenizer.bos_token_id] + gpt_ids + [self.tokenizer.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        ans_matrix=[1]*(len(ans)+1)
        ans_matrix+=[0]*(len(gpt_ids)-len(ans_matrix))
        expl_matrix=[1]*(len(expl)+1)
        expl_matrix=[0]*(len(gpt_ids)-len(expl_matrix))+expl_matrix
        if len(caption) > self.max_caption_len - 2:
            caption = caption[:self.max_caption_len - 2]
        caption = [self.tokenizer.bos_token_id] + caption + [self.tokenizer.eos_token_id]
        attn_mask_cap = [1] * len(caption)
        gpt_ids = torch.tensor(gpt_ids)
        attn_mask = torch.tensor(attn_mask)
        caption = torch.tensor(caption)
        attn_mask_cap = torch.tensor(attn_mask_cap)
        ans_matrix = torch.tensor(ans_matrix)
        expl_matrix = torch.tensor(expl_matrix)
        return [(torch.tensor(image_key),input_ids, segment_ids,input_mask,img_feat,gpt_ids,attn_mask,caption,attn_mask_cap,ans_matrix,expl_matrix)]

    def VQAGPT_gen_collate(self,inputs):
        (img_id, input_ids, segment_ids, input_mask, img_feat, gpt_ids, attn_mask, caption, attn_mask_cap,ans_matrix,expl_matrix) = map(
            list, unzip(concat(inputs)))
        ans_matrix = pad_sequence(ans_matrix, batch_first=True, padding_value=0)
        expl_matrix = pad_sequence(expl_matrix, batch_first=True, padding_value=0)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask_cap = pad_sequence(attn_mask_cap, batch_first=True, padding_value=0)
        caption = pad_sequence(caption, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)

        batch= {'img_id':img_id,'input_ids': input_ids, 'token_type_ids': segment_ids,
                'input_mask': input_mask,'img_feat':img_feat,'expl_ids':gpt_ids,'attn_mask':attn_mask,'cap_ids':caption,'attn_mask_cap':attn_mask_cap,'expl_matrix':expl_matrix,'ans_matrix':ans_matrix}
        batch = move_to_cuda(batch)
        return batch