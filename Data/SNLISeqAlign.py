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
import base64
import cv2
import pickle
from tqdm import tqdm


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


class SNLISeqBaseDatasets(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, SNLI_example_file, flickr_feat_file, chunk_mask_file,
                 max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
                 is_train=True, heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.label2id = {'neutral': 0,
                         'contradiction': 1,
                         'entailment': 2}
        self.SNLI_annot_dict = self.read_example(SNLI_example_file)
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_hypo_len = max_hypo_len
        self.max_seq_len = max_seq_length
        self.chunk_mask_dict = pickle.load(open(chunk_mask_file, 'rb'))
        if heat_index is not None:
            # 只保存当前样例
            self.SNLI_annot_dict = self.SNLI_annot_dict[heat_index:heat_index + 1]

    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example = self.SNLI_annot_dict[i]
        image_feat = self.image_feat_dict[example['Flickr30kID']]
        flickrID = example['Flickr30kID'].split('.')[0]
        input_ids = example['input_ids'].cuda()
        od_labels = image_feat['od_labels'].cuda()
        segment_ids = example['segment_ids'].cuda()
        img_feat = image_feat['image_feat'].cuda()
        label = torch.tensor(self.label2id[example['gold_label']]).cuda()
        expl = example['explanation']
        expl = self.gpt_toker.encode(expl)
        gpt_ids = expl
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids).cuda()
        attn_mask = torch.tensor(attn_mask).cuda()
        syn_labels_ids = image_feat['syn_labels_ids'].cuda()
        input_mask = example['input_mask'].cuda()

        offsets = self.chunk_mask_dict[example['pairID']]['offsets']
        chunk_mask = self.chunk_mask_dict[example['pairID']]['mask'].cuda()

        gather_index = []
        for idx, set in enumerate(offsets):
            set = torch.tensor(set).cuda()
            gather_index.extend([idx] * set.size(0))
        gather_index = torch.tensor(gather_index).cuda()
        return [(torch.tensor(int(flickrID)).cuda(), input_ids, segment_ids, input_mask, img_feat, label, gpt_ids,
                 attn_mask, syn_labels_ids, od_labels, chunk_mask, gather_index, offsets)]

    def SNLIGPT_gen_collate(self, inputs):
        (img_id, input_ids, segment_ids, input_mask, img_feat, label, gpt_ids, attn_mask, syn_labels_ids, od_labels,
         chunk_mask, gather_index, offsets) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        od_labels = pad_sequence(od_labels, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        max_label_len = max([item.size(1) for item in syn_labels_ids])
        syn_labels_ids_padd = []
        for idx, item in enumerate(syn_labels_ids):
            padd_matrix = torch.zeros((item.size(0), max_label_len - item.size(1)), dtype=torch.long).cuda()
            item = torch.cat((item, padd_matrix), dim=-1)
            syn_labels_ids_padd.append(item)
        syn_labels_ids_padd = torch.stack(syn_labels_ids_padd, dim=0)

        input_mask_hypo = input_mask[:, :self.max_hypo_len]
        max_hypo = input_mask_hypo.sum(-1).max()
        input_ids = input_ids[:, :max_hypo]
        input_mask_hypo = input_mask_hypo[:, :max_hypo]
        segment_ids = segment_ids[:, :max_hypo]

        input_mask_img = input_mask[:, self.max_hypo_len:]
        max_img = input_mask_img.sum(-1).max()
        img_feat = img_feat[:, :max_img]
        syn_labels_ids_padd = syn_labels_ids_padd[:, :max_img]
        input_mask_img = input_mask_img[:, :max_img]

        input_mask = torch.cat((input_mask_hypo, input_mask_img), -1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)
        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': label, 'expl_ids': gpt_ids,
                 'attn_mask': attn_mask, 'syn_labels_ids': syn_labels_ids_padd, 'od_labels': od_labels,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets
                 }

        return batch


class SNLISeqAlignChunkDataset_v7(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, SNLI_example_file, flickr_feat_file, chunk_mask_file,
                 max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
                 is_train=True, heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.label2id = {'neutral': 0,
                         'contradiction': 1,
                         'entailment': 2}
        self.SNLI_annot_dict = self.read_example(SNLI_example_file)
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_hypo_len = max_hypo_len
        self.max_seq_len = max_seq_length
        self.chunk_mask_dict = pickle.load(open(chunk_mask_file, 'rb'))
        if heat_index is not None:
            # 只保存当前样例
            self.SNLI_annot_dict = self.SNLI_annot_dict[heat_index:heat_index + 1]

    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example = self.SNLI_annot_dict[i]
        image_feat = self.image_feat_dict[example['Flickr30kID']]
        flickrID = example['Flickr30kID'].split('.')[0]
        input_ids = example['input_ids'].cuda()
        od_labels = image_feat['od_labels'].cuda()
        segment_ids = example['segment_ids'].cuda()
        img_feat = image_feat['image_feat'].cuda()
        label = torch.tensor(self.label2id[example['gold_label']]).cuda()
        expl = example['explanation']
        expl = self.gpt_toker.encode(expl)
        gpt_ids = expl
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids).cuda()
        attn_mask = torch.tensor(attn_mask).cuda()
        syn_labels_ids = image_feat['syn_labels_ids'].cuda()
        input_mask = example['input_mask'].cuda()

        offsets = self.chunk_mask_dict[example['pairID']]['offsets']
        chunk_mask = self.chunk_mask_dict[example['pairID']]['mask'].cuda()

        gather_index = []
        for idx, set in enumerate(offsets):
            set = torch.tensor(set).cuda()
            gather_index.extend([idx] * set.size(0))
        gather_index = torch.tensor(gather_index).cuda()
        return [(torch.tensor(int(flickrID)).cuda(), input_ids, segment_ids, input_mask, img_feat, label, gpt_ids,
                 attn_mask, syn_labels_ids, od_labels, chunk_mask, gather_index, offsets)]

    def SNLIGPT_gen_collate(self, inputs):
        (img_id, input_ids, segment_ids, input_mask, img_feat, label, gpt_ids, attn_mask, syn_labels_ids, od_labels,
         chunk_mask, gather_index, offsets) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        od_labels = pad_sequence(od_labels, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        max_label_len = max([item.size(1) for item in syn_labels_ids])
        syn_labels_ids_padd = []
        for idx, item in enumerate(syn_labels_ids):
            padd_matrix = torch.zeros((item.size(0), max_label_len - item.size(1)), dtype=torch.long).cuda()
            item = torch.cat((item, padd_matrix), dim=-1)
            syn_labels_ids_padd.append(item)
        syn_labels_ids_padd = torch.stack(syn_labels_ids_padd, dim=0)

        input_mask_hypo = input_mask[:, :self.max_hypo_len]
        max_hypo = input_mask_hypo.sum(-1).max()
        input_ids = input_ids[:, :max_hypo]
        input_mask_hypo = input_mask_hypo[:, :max_hypo]
        segment_ids = segment_ids[:, :max_hypo]

        input_mask_img = input_mask[:, self.max_hypo_len:]
        max_img = input_mask_img.sum(-1).max()
        img_feat = img_feat[:, :max_img]
        syn_labels_ids_padd = syn_labels_ids_padd[:, :max_img]
        input_mask_img = input_mask_img[:, :max_img]

        input_mask = torch.cat((input_mask_hypo, input_mask_img), -1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)
        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': label, 'expl_ids': gpt_ids,
                 'attn_mask': attn_mask, 'syn_labels_ids': syn_labels_ids_padd, 'od_labels': od_labels,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets
                 }

        return batch


class SNLISeqAlignChunkDataset_v7_prompt(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, SNLI_example_file, flickr_feat_file, chunk_mask_file,
                 max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
                 is_train=True):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.label2id = {'neutral': 0,
                         'contradiction': 1,
                         'entailment': 2}
        self.SNLI_annot_dict = self.read_example(SNLI_example_file)
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_hypo_len = max_hypo_len
        self.max_seq_len = max_seq_length
        self.chunk_mask_dict = pickle.load(open(chunk_mask_file, 'rb'))
        self.b_qu = "<|b_qn|>"
        self.e_qn = "<|e_qn|>"
        self.b_ans = "<|b_ans|>"
        self.e_ans = "<|e_ans|>"
        self.b_rtnl = "<|b_rtnl|>"
        self.e_rtnl = "<|e_rtnl|>"

    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example = self.SNLI_annot_dict[i]
        image_feat = self.image_feat_dict[example['Flickr30kID']]
        flickrID = example['Flickr30kID'].split('.')[0]
        input_ids = example['input_ids'].cuda()
        od_labels = image_feat['od_labels'].cuda()
        segment_ids = example['segment_ids'].cuda()
        img_feat = image_feat['image_feat'].cuda()
        label = torch.tensor(self.label2id[example['gold_label']]).cuda()

        qus = example['hypothesis']
        qus = " ".join(
            [self.b_qu, qus, self.e_qn]
        )
        qus = self.gpt_toker.encode(qus)

        ans = example['gold_label']
        ans = " ".join(
            [self.b_ans, ans, self.e_ans]
        )
        ans = self.gpt_toker.encode(ans)

        expl = example['explanation']
        expl = " ".join(
            [self.b_rtnl, expl, self.e_rtnl]
        )
        expl = self.gpt_toker.encode(expl)

        gpt_ids = qus + ans + expl
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        gpt_labels = [self.gpt_toker.pad_token_id] * (len(qus) + len(ans) + 2) + expl[1:] + [
            self.gpt_toker.pad_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids).cuda()
        gpt_labels = torch.tensor(gpt_labels).cuda()
        attn_mask = torch.tensor(attn_mask).cuda()
        syn_labels_ids = image_feat['syn_labels_ids'].cuda()
        input_mask = example['input_mask'].cuda()

        offsets = self.chunk_mask_dict[example['pairID']]['offsets']
        chunk_mask = self.chunk_mask_dict[example['pairID']]['mask'].cuda()

        gather_index = []
        for idx, offset in enumerate(offsets):
            offset = torch.tensor(offset).cuda()
            gather_index.extend([idx] * offset.size(0))
        gather_index = torch.tensor(gather_index).cuda()

        hypo = self.gpt_toker.encode(example['hypothesis'])
        expl = self.gpt_toker.encode(example['explanation'])
        pre_keywords = list(set(hypo) & set(expl))

        return [(torch.tensor(int(flickrID)).cuda(), input_ids, segment_ids, input_mask, img_feat, label, gpt_ids,
                 attn_mask, syn_labels_ids, od_labels, chunk_mask, gather_index, offsets, gpt_labels, pre_keywords)]

    def SNLIGPT_gen_collate(self, inputs):
        (img_id, input_ids, segment_ids, input_mask, img_feat, label, gpt_ids, attn_mask, syn_labels_ids, od_labels,
         chunk_mask, gather_index, offsets, gpt_labels, pre_keywords) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        od_labels = pad_sequence(od_labels, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        gpt_labels = pad_sequence(gpt_labels, batch_first=True, padding_value=self.gpt_toker.pad_token_id)

        img_id = torch.stack(img_id, dim=0)
        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)
        max_label_len = max([item.size(1) for item in syn_labels_ids])
        syn_labels_ids_padd = []
        for idx, item in enumerate(syn_labels_ids):
            padd_matrix = torch.zeros((item.size(0), max_label_len - item.size(1)), dtype=torch.long).cuda()
            item = torch.cat((item, padd_matrix), dim=-1)
            syn_labels_ids_padd.append(item)
        syn_labels_ids_padd = torch.stack(syn_labels_ids_padd, dim=0)

        input_mask_hypo = input_mask[:, :self.max_hypo_len]
        max_hypo = input_mask_hypo.sum(-1).max()
        input_ids = input_ids[:, :max_hypo]
        input_mask_hypo = input_mask_hypo[:, :max_hypo]
        segment_ids = segment_ids[:, :max_hypo]

        input_mask_img = input_mask[:, self.max_hypo_len:]
        max_img = input_mask_img.sum(-1).max()
        img_feat = img_feat[:, :max_img]
        syn_labels_ids_padd = syn_labels_ids_padd[:, :max_img]
        input_mask_img = input_mask_img[:, :max_img]

        input_mask = torch.cat((input_mask_hypo, input_mask_img), -1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)
        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': label, 'expl_ids': gpt_ids,
                 'attn_mask': attn_mask, 'syn_labels_ids': syn_labels_ids_padd, 'od_labels': od_labels,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'gpt_labels': gpt_labels, 'pre_keywords': pre_keywords
                 }

        return batch
