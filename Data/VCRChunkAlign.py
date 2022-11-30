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


class VCR_ChunkAlign_Dataset(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, VCR_example_file, VCR_chunk_mask_file,
                 flickr_feat_file,
                 max_img_seq_length=50,
                 is_train=True):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.VCR_annot_dict = pickle.load(open(VCR_example_file, 'rb'))
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token

    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.VCR_annot_dict)

    def __getitem__(self, i):
        example = self.VCR_annot_dict[i]
        image_feat = self.image_feat_dict[example['image_id']]
        img_feat = image_feat['features'].cuda()
        img_mask = image_feat['img_mask'].cuda()

        que_tokens = example['sent'].lower()
        que_tokens = self.bert_toker.tokenize(que_tokens)
        outputs = []

        for ans_idx, ans in enumerate(example['answer_choices']):
            ans_tokens = self.bert_toker.tokenize(ans)
            input_tokens = [self.cls] + que_tokens + [self.sep] + ans_tokens + [self.sep]
            input_ids = self.bert_toker.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda()
            mask_len = input_ids.size(0)
            input_mask = torch.ones(mask_len).cuda()

            segment_ids_ques = torch.zeros(len(que_tokens) + 2, dtype=torch.int64).cuda()
            segment_ids_ans = torch.ones(len(ans_tokens) + 1, dtype=torch.int64).cuda()
            segment_ids = torch.cat((segment_ids_ques, segment_ids_ans), 0)
            offsets = self.chunk_mask_dict[i][ans_idx]['offsets']
            chunk_mask = self.chunk_mask_dict[i][ans_idx]['mask'].cuda()

            gather_index = []
            for idx, set in enumerate(offsets):
                set = torch.tensor(set).cuda()
                gather_index.extend([idx] * set.size(0))
            gather_index = torch.tensor(gather_index).cuda()

            if ans_idx == example['answer_label']:
                target = torch.tensor(1).cuda()
            else:
                target = torch.tensor(0).cuda()

            expl = example['explanation']
            expl = self.gpt_toker.encode(expl)
            gpt_ids = [self.gpt_toker.bos_token_id] + expl + [self.gpt_toker.eos_token_id]
            attn_mask = [1] * len(gpt_ids)
            gpt_ids = torch.tensor(gpt_ids).cuda()
            attn_mask = torch.tensor(attn_mask).cuda()

            outputs.append((example['annot_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target,
                            chunk_mask, gather_index, offsets, example['sent'],
                            example['answer_choices'][ans_idx], gpt_ids, attn_mask))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):

        (img_id, input_ids, segment_ids, input_mask, img_feat, img_mask, target, chunk_mask,
         gather_index, offsets, ques, ans, gpt_ids, attn_mask) = map(list, unzip(concat(inputs)))
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)

        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        target = torch.stack(target, dim=0)

        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]

        input_mask = torch.cat((input_mask, img_mask), -1)

        max_hypo = input_ids.size(1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)

        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'ques_str': ques, 'ans_str': ans,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'expl_ids': gpt_ids, 'attn_mask': attn_mask,
                 }

        return batch


class VCR_ChunkAlign_Dataset_align(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, VCR_example_file, VCR_chunk_mask_file,
                 flickr_feat_file,
                 max_img_seq_length=50,
                 is_train=True,heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.VCR_annot_dict = pickle.load(open(VCR_example_file, 'rb'))
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token
        if heat_index is not None:
            # 只保存当前样例
            self.VCR_annot_dict = self.VCR_annot_dict[heat_index:heat_index + 1]


    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.VCR_annot_dict)

    def __getitem__(self, i):
        example = self.VCR_annot_dict[i]
        image_feat = self.image_feat_dict[example['image_id']]
        img_feat = image_feat['features'].cuda()
        img_mask = image_feat['img_mask'].cuda()

        que_tokens = example['sent'].lower()
        que_tokens = self.bert_toker.tokenize(que_tokens)
        outputs = []

        for ans_idx, ans in enumerate(example['answer_choices']):
            ans_tokens = self.bert_toker.tokenize(ans)
            input_tokens = [self.cls] + que_tokens + [self.sep] + ans_tokens + [self.sep]

            region_tokens = [0] * len(input_tokens)
            for token_idx, token in enumerate(input_tokens):
                if '<|det' in token:
                    index = token[5:].split('|')[0]
                    region_tokens[token_idx] = int(index)
            region_tokens = torch.tensor(region_tokens).cuda()
            total_label = region_tokens
            align_pos = torch.where(total_label != 0, torch.ones_like(total_label).cuda(), total_label)

            input_ids = self.bert_toker.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda()
            mask_len = input_ids.size(0)
            input_mask = torch.ones(mask_len).cuda()

            segment_ids_ques = torch.zeros(len(que_tokens) + 2, dtype=torch.int64).cuda()
            segment_ids_ans = torch.ones(len(ans_tokens) + 1, dtype=torch.int64).cuda()
            segment_ids = torch.cat((segment_ids_ques, segment_ids_ans), 0)
            offsets = self.chunk_mask_dict[i][ans_idx]['offsets']
            chunk_mask = self.chunk_mask_dict[i][ans_idx]['mask'].cuda()

            gather_index = []
            for idx, set in enumerate(offsets):
                set = torch.tensor(set).cuda()
                gather_index.extend([idx] * set.size(0))
            gather_index = torch.tensor(gather_index).cuda()

            if ans_idx == example['answer_label']:
                target = torch.tensor(1).cuda()
            else:
                target = torch.tensor(0).cuda()

            outputs.append((example['annot_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target,
                            chunk_mask, gather_index, offsets, example['sent'],
                            example['answer_choices'][ans_idx], total_label, align_pos))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):

        (img_id, input_ids, segment_ids, input_mask, img_feat, img_mask, target, chunk_mask,
         gather_index, offsets, ques, ans, total_label, align_pos) = map(list, unzip(concat(inputs)))
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)

        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        target = torch.stack(target, dim=0)

        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]

        input_mask = torch.cat((input_mask, img_mask), -1)

        max_hypo = input_ids.size(1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)

        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'ques_str': ques, 'ans_str': ans,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'total_label': total_label, 'align_pos': align_pos
                 }

        return batch


class VCR_ChunkAlign_prefix_Dataset(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, VCR_example_file, VCR_chunk_mask_file,
                 flickr_feat_file,
                 max_img_seq_length=50,
                 is_train=True, heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.VCR_annot_dict = pickle.load(open(VCR_example_file, 'rb'))
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        self.b_qu = "<|b_qn|>"
        self.e_qn = "<|e_qn|>"
        self.b_ans = "<|b_ans|>"
        self.e_ans = "<|e_ans|>"
        self.b_rtnl = "<|b_rtnl|>"
        self.e_rtnl = "<|e_rtnl|>"
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token
        if heat_index is not None:
            # 只保存当前样例
            self.VCR_annot_dict = self.VCR_annot_dict[heat_index:heat_index + 1]

    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.VCR_annot_dict)

    def __getitem__(self, i):
        example = self.VCR_annot_dict[i]
        image_feat = self.image_feat_dict[example['image_id']]
        img_feat = image_feat['features'].cuda()
        img_mask = image_feat['img_mask'].cuda()

        que_tokens = example['sent'].lower()
        que_tokens = self.bert_toker.tokenize(que_tokens)
        outputs = []

        golden_ans = example['answer_choices'][example['answer_label']]
        golden_ans_ids = " ".join(
            [self.b_ans, golden_ans, self.e_ans]
        )
        golden_ans_ids = self.gpt_toker.encode(golden_ans_ids)

        qus = example['sent']
        qus = " ".join(
            [self.b_qu, qus, self.e_qn]
        )
        qus = self.gpt_toker.encode(qus)

        for ans_idx, ans in enumerate(example['answer_choices']):
            ans_tokens = self.bert_toker.tokenize(ans)
            input_tokens = [self.cls] + que_tokens + [self.sep] + ans_tokens + [self.sep]
            input_ids = self.bert_toker.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda()
            mask_len = input_ids.size(0)
            input_mask = torch.ones(mask_len).cuda()

            segment_ids_ques = torch.zeros(len(que_tokens) + 2, dtype=torch.int64).cuda()
            segment_ids_ans = torch.ones(len(ans_tokens) + 1, dtype=torch.int64).cuda()
            segment_ids = torch.cat((segment_ids_ques, segment_ids_ans), 0)
            offsets = self.chunk_mask_dict[i][ans_idx]['offsets']
            chunk_mask = self.chunk_mask_dict[i][ans_idx]['mask'].cuda()

            gather_index = []
            for idx, offset in enumerate(offsets):
                offset = torch.tensor(offset).cuda()
                gather_index.extend([idx] * offset.size(0))
            gather_index = torch.tensor(gather_index).cuda()

            if ans_idx == example['answer_label']:
                target = torch.tensor(1).cuda()
            else:
                target = torch.tensor(0).cuda()

            expl = example['explanation']
            expl = " ".join(
                [self.b_rtnl, expl, self.e_rtnl]
            )
            expl = self.gpt_toker.encode(expl)

            gpt_ids = qus + golden_ans_ids + expl
            gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
            gpt_labels = [self.gpt_toker.pad_token_id] * (len(qus) + len(golden_ans_ids) + 2) + expl[1:] + [
                self.gpt_toker.pad_token_id]
            attn_mask = [1] * len(gpt_ids)
            gpt_ids = torch.tensor(gpt_ids).cuda()
            gpt_labels = torch.tensor(gpt_labels).cuda()
            attn_mask = torch.tensor(attn_mask).cuda()
            hypo = self.gpt_toker.encode(example['sent'] + ' ' + ans)
            expl = self.gpt_toker.encode(example['explanation'])
            pre_keywords = list(set(hypo) & set(expl))
            outputs.append(
                (example['annot_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target, gpt_ids,
                 attn_mask, chunk_mask, gather_index, offsets, example['sent'],
                 example['answer_choices'][ans_idx], gpt_labels, pre_keywords))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):

        (img_id, input_ids, segment_ids, input_mask, img_feat, img_mask, target, gpt_ids, attn_mask, chunk_mask,
         gather_index, offsets, ques, ans, gpt_labels, pre_keywords) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
        gpt_labels = pad_sequence(gpt_labels, batch_first=True, padding_value=self.gpt_toker.pad_token_id)

        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        target = torch.stack(target, dim=0)

        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]

        input_mask = torch.cat((input_mask, img_mask), -1)

        max_hypo = input_ids.size(1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)

        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'expl_ids': gpt_ids,
                 'attn_mask': attn_mask, 'ques_str': ques, 'ans_str': ans, 'gpt_labels': gpt_labels,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'pre_keywords': pre_keywords
                 }

        return batch
