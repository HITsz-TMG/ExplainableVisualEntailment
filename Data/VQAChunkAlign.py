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


class VQA_ChunkAlign_Dataset(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, vqa_example_file, vqa_chunk_mask_file, vqa_x_ans2label,
                 flickr_feat_file,
                 max_img_seq_length=50,
                 is_train=True, heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.ans2label = pickle.load(open(vqa_x_ans2label, 'rb'))
        self.chunk_mask_dict = pickle.load(open(vqa_chunk_mask_file, 'rb'))
        self.vqa_x_annot_dict = json.load(open(vqa_example_file))
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.num_answers = len(self.ans2label)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        if heat_index is not None:
            # 只保存当前样例
            self.vqa_x_annot_dict = self.vqa_x_annot_dict[heat_index:heat_index + 1]

    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.vqa_x_annot_dict)

    def __getitem__(self, i):
        example = self.vqa_x_annot_dict[i]
        image_feat = self.image_feat_dict[example['img_id']]
        od_labels = image_feat['od_labels']
        img_feat = image_feat['img_feat'].cuda()
        if img_feat.size(0) > self.max_img_seq_length:
            img_feat = img_feat[:self.max_img_seq_length]
        img_mask = image_feat['img_mask'].cuda()

        input_ids = example['sent'].lower()
        input_ids = self.bert_toker.encode(input_ids)
        input_ids = torch.tensor(input_ids).cuda()
        segment_ids = torch.zeros(input_ids.size(0), dtype=torch.int64).cuda()
        input_mask = torch.ones(input_ids.size(0), dtype=torch.int64).cuda()

        offsets = self.chunk_mask_dict[i]['offsets']
        chunk_mask = self.chunk_mask_dict[i]['mask'].cuda()

        gather_index = []
        for idx, set in enumerate(offsets):
            set = torch.tensor(set).cuda()
            gather_index.extend([idx] * set.size(0))
        gather_index = torch.tensor(gather_index).cuda()

        target = torch.zeros(self.num_answers).cuda()
        for ans, score in example['label'].items():
            target[self.ans2label[ans]] = score

        expl_list = example['explanation']
        if self.is_train:
            # 训练时只考虑第一条解释
            expl = expl_list[0]
            expl = self.gpt_toker.encode(expl)
            gpt_ids = expl
            gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
            attn_mask = [1] * len(gpt_ids)
            gpt_ids = torch.tensor(gpt_ids).cuda()
            attn_mask = torch.tensor(attn_mask).cuda()
            return [(example['img_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target, gpt_ids,
                     attn_mask, od_labels, chunk_mask, gather_index, offsets)]
        else:
            return [(example['img_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target, expl_list,
                     od_labels, chunk_mask, gather_index, offsets)]

    def SNLIGPT_gen_collate(self, inputs):
        if self.is_train:
            (img_id, input_ids, segment_ids, input_mask, img_feat, img_mask, target, gpt_ids, attn_mask,
             od_labels, chunk_mask, gather_index, offsets) = map(
                list, unzip(
                    concat(inputs)))
            attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
            input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
            # img_id = torch.stack(img_id, dim=0)
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
                     'attn_mask': attn_mask, 'od_labels': od_labels,
                     'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets
                     }
        else:
            (img_id, input_ids, segment_ids, input_mask, img_feat, img_mask, target, expl_list,
             od_labels, chunk_mask, gather_index, offsets) = map(
                list, unzip(
                    concat(inputs)))
            input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

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
                     'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'expl_list': expl_list,
                     'od_labels': od_labels, 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index,
                     'offsets': offsets}
        return batch


class VQA_ChunkAlign_prefix_Dataset(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, vqa_example_file, vqa_chunk_mask_file, vqa_x_ans2label,
                 flickr_feat_file,
                 max_img_seq_length=50,
                 is_train=True):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.ans2label = pickle.load(open(vqa_x_ans2label, 'rb'))
        self.chunk_mask_dict = pickle.load(open(vqa_chunk_mask_file, 'rb'))
        self.vqa_x_annot_dict = json.load(open(vqa_example_file))
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.num_answers = len(self.ans2label)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
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
        return len(self.vqa_x_annot_dict)

    def __getitem__(self, i):
        example = self.vqa_x_annot_dict[i]
        image_feat = self.image_feat_dict[example['img_id']]
        od_labels = image_feat['od_labels']
        img_feat = image_feat['img_feat'].cuda()
        if img_feat.size(0) > self.max_img_seq_length:
            img_feat = img_feat[:self.max_img_seq_length]
        img_mask = image_feat['img_mask'].cuda()

        input_ids = example['sent'].lower()
        input_ids = self.bert_toker.encode(input_ids)
        input_ids = torch.tensor(input_ids).cuda()
        segment_ids = torch.zeros(input_ids.size(0), dtype=torch.int64).cuda()
        input_mask = torch.ones(input_ids.size(0), dtype=torch.int64).cuda()

        offsets = self.chunk_mask_dict[i]['offsets']
        chunk_mask = self.chunk_mask_dict[i]['mask'].cuda()

        gather_index = []
        for idx, offset in enumerate(offsets):
            offset = torch.tensor(offset).cuda()
            gather_index.extend([idx] * offset.size(0))
        gather_index = torch.tensor(gather_index).cuda()

        target = torch.zeros(self.num_answers).cuda()
        ans_list = []
        for ans, score in example['label'].items():
            ans_list.append(ans)
            target[self.ans2label[ans]] = score

        expl_list = example['explanation']

        max_pre = 0
        chosen_expl = expl_list[0]
        for expl in expl_list:
            hypo_id = self.gpt_toker.encode(example['sent'])
            expl_id = self.gpt_toker.encode(expl)
            pre_keywords = list(set(hypo_id) & set(expl_id))
            if len(pre_keywords) > max_pre:
                max_pre = len(pre_keywords)
                chosen_expl = expl

        expl = " ".join(
            [self.b_rtnl, chosen_expl, self.e_rtnl]
        )
        expl = self.gpt_toker.encode(expl)

        qus = example['sent']
        qus = " ".join(
            [self.b_qu, qus, self.e_qn]
        )
        qus = self.gpt_toker.encode(qus)

        if len(ans_list) > 0:
            ans = ans_list[0]
            ans = " ".join(
                [self.b_ans, ans, self.e_ans]
            )
            ans = self.gpt_toker.encode(ans)
            gpt_ids = qus + ans + expl
            gpt_labels = [self.gpt_toker.pad_token_id] * (len(qus) + len(ans) + 1) + expl[1:]
        else:
            gpt_ids = qus + expl
            gpt_labels = [self.gpt_toker.pad_token_id] * (len(qus) + 1) + expl[1:]

        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        gpt_labels = [self.gpt_toker.pad_token_id] + gpt_labels + [self.gpt_toker.pad_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids).cuda()
        gpt_labels = torch.tensor(gpt_labels).cuda()
        attn_mask = torch.tensor(attn_mask).cuda()

        if self.is_train:
            # 训练时只考虑第一条解释

            return [(example['img_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target, gpt_ids,
                     attn_mask, od_labels, chunk_mask, gather_index, offsets, gpt_labels, pre_keywords)]
        else:

            return [(example['question_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target, expl_list,
                     gpt_ids, attn_mask, od_labels, chunk_mask, gather_index, offsets)]

    def SNLIGPT_gen_collate(self, inputs):
        if self.is_train:
            (img_id, input_ids, segment_ids, input_mask, img_feat, img_mask, target, gpt_ids, attn_mask,
             od_labels, chunk_mask, gather_index, offsets, gpt_labels, pre_keywords) = map(
                list, unzip(
                    concat(inputs)))
            attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
            input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
            gpt_labels = pad_sequence(gpt_labels, batch_first=True, padding_value=self.gpt_toker.pad_token_id)

            # img_id = torch.stack(img_id, dim=0)
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
                     'attn_mask': attn_mask, 'od_labels': od_labels, 'gpt_labels': gpt_labels,
                     'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                     'pre_keywords': pre_keywords
                     }
        else:
            (q_ids, input_ids, segment_ids, input_mask, img_feat, img_mask, target, expl_list,
             gpt_ids, attn_mask, od_labels, chunk_mask, gather_index, offsets) = map(
                list, unzip(
                    concat(inputs)))
            input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
            attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
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
            batch = {'q_ids': q_ids, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                     'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'expl_list': expl_list,
                     'od_labels': od_labels, 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index,
                     'offsets': offsets, 'gpt_ids': gpt_ids,
                     'attn_mask': attn_mask, }
        return batch


class VQA_ChunkAlign_prefix_ra_Dataset(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, vqa_example_file, vqa_chunk_mask_file, vqa_x_ans2label,
                 flickr_feat_file,
                 max_img_seq_length=50,
                 is_train=True):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.ans2label = pickle.load(open(vqa_x_ans2label, 'rb'))
        self.chunk_mask_dict = pickle.load(open(vqa_chunk_mask_file, 'rb'))
        self.i2t = json.load(
            open('/mnt/inspurfs/user-fs/yangqian/Oscar/pretrained_models/ra/retrieval_attack_data_i2t_vqaX.json', 'r'))
        self.t2i = json.load(
            open('/mnt/inspurfs/user-fs/yangqian/Oscar/pretrained_models/ra/retrieval_attack_data_t2i_vqaX.json', 'r'))

        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.num_answers = len(self.ans2label)
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
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
        return len(self.vqa_x_annot_dict)

    def __getitem__(self, i):
        example = self.vqa_x_annot_dict[i]
        image_feat = self.image_feat_dict[example['img_id']]
        od_labels = image_feat['od_labels']
        img_feat = image_feat['img_feat'].cuda()
        if img_feat.size(0) > self.max_img_seq_length:
            img_feat = img_feat[:self.max_img_seq_length]
        img_mask = image_feat['img_mask'].cuda()

        input_ids = example['sent'].lower()
        input_ids = self.bert_toker.encode(input_ids)
        input_ids = torch.tensor(input_ids).cuda()
        segment_ids = torch.zeros(input_ids.size(0), dtype=torch.int64).cuda()
        input_mask = torch.ones(input_ids.size(0), dtype=torch.int64).cuda()

        offsets = self.chunk_mask_dict[i]['offsets']
        chunk_mask = self.chunk_mask_dict[i]['mask'].cuda()

        gather_index = []
        for idx, offset in enumerate(offsets):
            offset = torch.tensor(offset).cuda()
            gather_index.extend([idx] * offset.size(0))
        gather_index = torch.tensor(gather_index).cuda()

        target = torch.zeros(self.num_answers).cuda()
        ans_list = []
        for ans, score in example['label'].items():
            ans_list.append(ans)
            target[self.ans2label[ans]] = score

        expl_list = example['explanation']

        max_pre = 0
        chosen_expl = expl_list[0]
        for expl in expl_list:
            hypo_id = self.gpt_toker.encode(example['sent'])
            expl_id = self.gpt_toker.encode(expl)
            pre_keywords = list(set(hypo_id) & set(expl_id))
            if len(pre_keywords) > max_pre:
                max_pre = len(pre_keywords)
                chosen_expl = expl

        expl = " ".join(
            [self.b_rtnl, chosen_expl, self.e_rtnl]
        )
        expl = self.gpt_toker.encode(expl)

        qus = example['sent']
        qus = " ".join(
            [self.b_qu, qus, self.e_qn]
        )
        qus = self.gpt_toker.encode(qus)

        if len(ans_list) > 0:
            ans = ans_list[0]
            ans = " ".join(
                [self.b_ans, ans, self.e_ans]
            )
            ans = self.gpt_toker.encode(ans)
            gpt_ids = qus + ans + expl
            gpt_labels = [self.gpt_toker.pad_token_id] * (len(qus) + len(ans) + 1) + expl[1:]
        else:
            gpt_ids = qus + expl
            gpt_labels = [self.gpt_toker.pad_token_id] * (len(qus) + 1) + expl[1:]

        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        gpt_labels = [self.gpt_toker.pad_token_id] + gpt_labels + [self.gpt_toker.pad_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids).cuda()
        gpt_labels = torch.tensor(gpt_labels).cuda()
        attn_mask = torch.tensor(attn_mask).cuda()

        if self.is_train:
            # 训练时只考虑第一条解释

            return [(example['img_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target, gpt_ids,
                     attn_mask, od_labels, chunk_mask, gather_index, offsets, gpt_labels, pre_keywords)]
        else:

            return [(example['question_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target, expl_list,
                     gpt_ids, attn_mask, od_labels, chunk_mask, gather_index, offsets)]

    def SNLIGPT_gen_collate(self, inputs):
        if self.is_train:
            (img_id, input_ids, segment_ids, input_mask, img_feat, img_mask, target, gpt_ids, attn_mask,
             od_labels, chunk_mask, gather_index, offsets, gpt_labels, pre_keywords) = map(
                list, unzip(
                    concat(inputs)))
            attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
            input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
            gpt_labels = pad_sequence(gpt_labels, batch_first=True, padding_value=self.gpt_toker.pad_token_id)

            # img_id = torch.stack(img_id, dim=0)
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
                     'attn_mask': attn_mask, 'od_labels': od_labels, 'gpt_labels': gpt_labels,
                     'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                     'pre_keywords': pre_keywords
                     }
        else:
            (q_ids, input_ids, segment_ids, input_mask, img_feat, img_mask, target, expl_list,
             gpt_ids, attn_mask, od_labels, chunk_mask, gather_index, offsets) = map(
                list, unzip(
                    concat(inputs)))
            input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            gpt_ids = pad_sequence(gpt_ids, batch_first=True, padding_value=self.gpt_toker.pad_token_id)
            attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
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
            batch = {'q_ids': q_ids, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                     'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'expl_list': expl_list,
                     'od_labels': od_labels, 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index,
                     'offsets': offsets, 'gpt_ids': gpt_ids,
                     'attn_mask': attn_mask, }
        return batch
