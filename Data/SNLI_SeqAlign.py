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
import re
from random import randint
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


class SNLI_Dataset(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, SNLI_example_file, max_img_seq_length=50, max_seq_length=140,
                 max_hypo_len=40,
                 is_train=True):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.label2id = {'neutral': 0,
                         'contradiction': 1,
                         'entailment': 2}
        self.SNLI_annot_dict = self.read_example(SNLI_example_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_hypo_len = max_hypo_len
        self.max_seq_len = max_seq_length
        self.SNLI_example_file = SNLI_example_file

    def read_example(self, path):
        if os.path.isdir(path):
            data = []
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    file = os.path.join(root, name)
                    tmp_data = pickle.load(open(file, 'rb'))
                    data.extend(tmp_data)
        elif os.path.isfile(path):
            data = pickle.load(open(path, 'rb'))
        return data

    def del_example(self):
        # 删除一部分数据
        self.SNLI_annot_dict = []

    def read_del_example(self):
        data = []
        for root, dirs, files in os.walk(self.SNLI_example_file, topdown=False):
            for name in files:
                file = os.path.join(root, name)
                tmp_data = pickle.load(open(file, 'rb'))
                data.extend(tmp_data)
        self.SNLI_annot_dict = data

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example = self.SNLI_annot_dict[i]
        flickrID = example['Flickr30kID'].split('.')[0]
        input_ids = example['input_ids'].cuda()
        od_labels = example['od_labels'].cuda()
        segment_ids = example['segment_ids'].cuda()
        input_mask = example['input_mask'].cuda()
        img_feat = example['img_feat'].cuda()
        label = torch.tensor(self.label2id[example['gold_label']]).cuda()
        expl = example['explanation']
        expl = self.gpt_toker.encode(expl)
        gpt_ids = expl
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids).cuda()
        attn_mask = torch.tensor(attn_mask).cuda()
        syn_labels_ids = example['syn_labels_ids'].cuda()
        return [(torch.tensor(int(flickrID)).cuda(), input_ids, segment_ids, input_mask, img_feat, label, gpt_ids,
                 attn_mask, syn_labels_ids, od_labels)]

    def SNLIGPT_gen_collate(self, inputs):
        (img_id, input_ids, segment_ids, input_mask, img_feat, label, gpt_ids, attn_mask, syn_labels_ids,
         od_labels) = map(list, unzip(concat(inputs)))
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
        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': label, 'expl_ids': gpt_ids,
                 'attn_mask': attn_mask,
                 'syn_labels_ids': syn_labels_ids_padd, 'od_labels': od_labels
                 }

        return batch


class SeqAlignPretrainDataset_v2(Dataset):
    def __init__(self, bert_tokenizer, example_file, feat_file):
        self.bert_toker = bert_tokenizer
        self.add_od_labels = True
        self.example_file = pickle.load(open(example_file, 'rb'))
        self.feat_dict = pickle.load(open(feat_file, 'rb'))
        self.text_cls_id = self.bert_toker.encode("[TEXT_CLS]", add_special_tokens=False)
        self.mask_prob = 0.15
        self.max_masked_tokens = 5

    def __len__(self):
        return len(self.example_file)

    def __getitem__(self, i):
        example = self.example_file[i]
        image_key = example['img_key']
        img_feat = self.feat_dict[image_key]['features'].cuda()
        img_mask = self.feat_dict[image_key]['img_mask'].cuda()
        sent = example['sentence']

        input_ids = self.bert_toker.encode(sent, add_special_tokens=False)
        seq_len = len(input_ids)
        masked_pos = torch.zeros(seq_len, dtype=torch.int).cuda()
        # randomly mask words for prediction, ignore [CLS]
        candidate_masked_idx = list(range(0, len(input_ids)))  # only mask text_a
        random.shuffle(candidate_masked_idx)
        num_masked = min(max(round(self.mask_prob * seq_len), 1), self.max_masked_tokens)
        num_masked = int(num_masked)
        masked_idx = candidate_masked_idx[:num_masked]
        masked_idx = sorted(masked_idx)
        masked_token = [input_ids[i] for i in masked_idx]
        masked_input_ids = input_ids.copy()
        for pos in masked_idx:
            if random.random() <= 0.8:
                # 80% chance to be a ['MASK'] token
                masked_input_ids[pos] = self.bert_toker.mask_token_id
            elif random.random() <= 0.5:
                # 10% chance to be a random word ((1-0.8)*0.5)
                index = randint(1, len(self.bert_toker.vocab) - 1)
                masked_input_ids[pos] = index
            else:
                # 10% chance to remain the same (1-0.8-0.1)
                pass
        masked_pos[masked_idx] = 1
        if num_masked < self.max_masked_tokens:
            masked_token = masked_token + ([0] * (self.max_masked_tokens - num_masked))
        masked_token = torch.tensor(masked_token).cuda()
        input_ids = self.text_cls_id + input_ids
        input_ids = [self.bert_toker.cls_token_id] + input_ids + [self.bert_toker.sep_token_id]
        input_ids = torch.tensor(input_ids).cuda()

        masked_input_ids = self.text_cls_id + masked_input_ids
        masked_input_ids = [self.bert_toker.cls_token_id] + masked_input_ids + [self.bert_toker.sep_token_id]
        masked_input_ids = torch.tensor(masked_input_ids).cuda()
        # 两个CLS
        masked_pos = torch.cat((torch.zeros(2).cuda(), masked_pos), 0)

        input_mask = torch.ones(input_ids.size(0)).cuda()
        segment_ids = torch.zeros_like(input_ids)

        ChunkMask_wo_TextCLS = example['ChunkMask'].cuda()
        chunk_mask = torch.cat((torch.ones(1, ChunkMask_wo_TextCLS.size(1)).cuda(), ChunkMask_wo_TextCLS), 0)
        chunk_mask = torch.cat((torch.zeros(chunk_mask.size(0), 1).cuda(), chunk_mask), 1)
        # TextCLS只看见text,不看CLS
        chunk_mask[1, 0] = 0
        chunk_mask[0, 0] = 1
        chunk_mask[-1, 0] = 1

        new_offsets = []
        offsets = example['full_offsets']
        gather_index = []
        for idx, set in enumerate(offsets):
            new_offset = []
            for item in set:
                new_offset.append(item + 1)
            new_offsets.append(new_offset)
            set = torch.tensor(set).cuda()
            gather_index.extend([idx] * set.size(0))
        gather_index = torch.tensor(gather_index).cuda()
        # 保证gather正确
        assert gather_index.size(0) + 3 == input_ids.size(0)

        total_label = torch.zeros(input_ids.size(0)).cuda()
        align_pos = torch.zeros(input_ids.size(0), dtype=torch.int).cuda()
        annot = example['annot']
        offsets = example['offsets']
        for idx, offset in enumerate(offsets):
            align_label = list(annot[idx].values())[0].squeeze(0)
            # gold image
            align_index = torch.nonzero(align_label)
            if align_index.size(0) > 0:
                align_label = align_index[0].item() + 1
                if align_label >= 50:
                    continue
                    # 最大图片长度为50
            else:
                continue
            # TextCLS
            align_pos[offset[0] + 1:offset[-1] + 1 + 1] = 1
            total_label[offset[0] + 1:offset[-1] + 1 + 1] = align_label
        return [(torch.tensor(i).cuda(), input_ids, segment_ids, img_feat, img_mask, total_label,
                 chunk_mask, new_offsets, input_mask, masked_input_ids, masked_pos, masked_token, gather_index,
                 align_pos)]

    def SNLIGPT_gen_collate(self, inputs):
        (id, input_ids, segment_ids, img_feat, img_mask, total_label, chunk_attention_mask,
         offsets, input_mask, masked_input_ids, masked_pos, masked_token, gather_index, align_pos) = map(list, unzip(
            concat(inputs)))

        # id = torch.stack(id, dim=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).unsqueeze(1)
        masked_input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=0).unsqueeze(1)
        input_ids = torch.cat((input_ids, masked_input_ids), 1)
        masked_pos = pad_sequence(masked_pos, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)
        masked_token = torch.stack(masked_token, dim=0)
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]
        input_mask = torch.cat((input_mask, img_mask), -1)

        # [TEXT_CLS]看不见CLS\image
        input_mask = input_mask.unsqueeze(1).repeat(1, input_mask.size(-1), 1)
        input_mask[:, 1, 0] = 0
        input_mask[:, 1, -max_img:] = 0

        max_hypo = input_ids.size(-1)
        chunk_attention_mask_padd = []
        for item in chunk_attention_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((item.size(0), max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), -1)
            chunk_attention_mask_padd.append(item)
        chunk_attention_mask_padd = torch.stack(chunk_attention_mask_padd, 0)

        batch = {'id': id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'img_feat': img_feat, 'total_label': total_label, 'img_mask': img_mask,
                 'chunk_attention_mask': chunk_attention_mask_padd, 'offsets': offsets,
                 'input_mask': input_mask, 'masked_pos': masked_pos, 'masked_token': masked_token,
                 'gather_index': gather_index, 'align_pos': align_pos
                 }

        return batch


class SeqAlignPretrainDataset_v2_test(Dataset):
    def __init__(self, bert_tokenizer, example_file, feat_file):
        self.bert_toker = bert_tokenizer
        self.add_od_labels = True
        self.example_file = pickle.load(open(example_file, 'rb'))
        self.feat_dict = pickle.load(open(feat_file, 'rb'))
        self.text_cls_id = self.bert_toker.encode("[TEXT_CLS]", add_special_tokens=False)
        self.mask_prob = 0.15
        self.max_masked_tokens = 5

    def __len__(self):
        return len(self.example_file)

    def __getitem__(self, i):
        example = self.example_file[i]
        image_key = example['img_key']
        img_feat = self.feat_dict[image_key]['features'].cuda()
        img_mask = self.feat_dict[image_key]['img_mask'].cuda()
        sent = example['sentence']

        input_ids = self.bert_toker.encode(sent, add_special_tokens=False)
        input_ids = self.text_cls_id + input_ids
        input_ids = [self.bert_toker.cls_token_id] + input_ids + [self.bert_toker.sep_token_id]
        input_ids = torch.tensor(input_ids).cuda()

        input_mask = torch.ones(input_ids.size(0)).cuda()
        segment_ids = torch.zeros_like(input_ids)

        ChunkMask_wo_TextCLS = example['ChunkMask'].cuda()
        chunk_mask = torch.cat((torch.ones(1, ChunkMask_wo_TextCLS.size(1)).cuda(), ChunkMask_wo_TextCLS), 0)
        chunk_mask = torch.cat((torch.zeros(chunk_mask.size(0), 1).cuda(), chunk_mask), 1)
        # TextCLS只看见text,不看CLS
        chunk_mask[1, 0] = 0
        chunk_mask[0, 0] = 1
        chunk_mask[-1, 0] = 1

        new_offsets = []
        offsets = example['full_offsets']
        gather_index = []
        for idx, set in enumerate(offsets):
            new_offset = []
            for item in set:
                new_offset.append(item + 1)
            new_offsets.append(new_offset)
            set = torch.tensor(set).cuda()
            gather_index.extend([idx] * set.size(0))
        gather_index = torch.tensor(gather_index).cuda()

        total_label = torch.zeros(input_ids.size(0)).cuda()
        align_pos = torch.zeros(input_ids.size(0), dtype=torch.int).cuda()
        annot = example['annot']
        offsets = example['offsets']
        for idx, offset in enumerate(offsets):
            align_label = list(annot[idx].values())[0].squeeze(0)
            # gold image
            align_index = torch.nonzero(align_label)
            if align_index.size(0) > 0:
                align_label = align_index[0].item() + 1
                if align_label >= 50:
                    continue
                    # 最大图片长度为50
            else:
                continue
            # TextCLS
            align_pos[offset[0] + 1:offset[-1] + 1 + 1] = 1
            total_label[offset[0] + 1:offset[-1] + 1 + 1] = align_label
        return [(torch.tensor(i).cuda(), input_ids, segment_ids, img_feat, img_mask, total_label,
                 chunk_mask, new_offsets, input_mask, gather_index, align_pos)]

    def SNLIGPT_gen_collate(self, inputs):
        (id, input_ids, segment_ids, img_feat, img_mask, total_label, chunk_attention_mask,
         offsets, input_mask, gather_index, align_pos) = map(list, unzip(
            concat(inputs)))

        id = torch.stack(id, dim=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]
        input_mask = torch.cat((input_mask, img_mask), -1)

        # [TEXT_CLS]看不见CLS\image
        input_mask = input_mask.unsqueeze(1).repeat(1, input_mask.size(-1), 1)
        input_mask[:, 1, 0] = 0
        input_mask[:, 1, -max_img:] = 0

        max_hypo = input_ids.size(-1)
        chunk_attention_mask_padd = []
        for item in chunk_attention_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((item.size(0), max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), -1)
            chunk_attention_mask_padd.append(item)
        chunk_attention_mask_padd = torch.stack(chunk_attention_mask_padd, 0)

        batch = {'id': id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'img_feat': img_feat, 'total_label': total_label, 'img_mask': img_mask,
                 'chunk_attention_mask': chunk_attention_mask_padd, 'offsets': offsets,
                 'input_mask': input_mask, 'gather_index': gather_index, 'align_pos': align_pos
                 }

        return batch


class SeqAlignPretrainDataset_v2_align_only(Dataset):
    def __init__(self, bert_tokenizer, example_file, feat_file):
        self.bert_toker = bert_tokenizer
        self.add_od_labels = True
        self.example_file = pickle.load(open(example_file, 'rb'))
        self.feat_dict = pickle.load(open(feat_file, 'rb'))

    def __len__(self):
        return len(self.example_file)

    def __getitem__(self, i):
        example = self.example_file[i]
        image_key = example['img_key']
        img_feat = self.feat_dict[image_key]['features'].cuda()
        img_mask = self.feat_dict[image_key]['img_mask'].cuda()
        sent = example['sentence']

        input_ids = self.bert_toker.encode(sent, add_special_tokens=True)
        input_ids = torch.tensor(input_ids).cuda()
        input_mask = torch.ones(input_ids.size(0)).cuda()
        segment_ids = torch.zeros_like(input_ids)

        chunk_mask = example['ChunkMask'].cuda()
        full_offsets = example['full_offsets']
        gather_index = []
        for idx, set in enumerate(full_offsets):
            set = torch.tensor(set).cuda()
            gather_index.extend([idx] * set.size(0))
        gather_index = torch.tensor(gather_index).cuda()

        total_label = torch.zeros(input_ids.size(0)).cuda()
        align_pos = torch.zeros(input_ids.size(0), dtype=torch.int).cuda()
        annot = example['annot']
        offsets = example['offsets']
        for idx, offset in enumerate(offsets):
            align_label = list(annot[idx].values())[0].squeeze(0)
            # gold image
            align_index = torch.nonzero(align_label)
            if align_index.size(0) > 0:
                align_label = align_index[0].item() + 1
                if align_label >= 50:
                    continue
                    # 最大图片长度为50
            else:
                continue
            align_pos[offset[0]:offset[-1] + 1] = 1
            total_label[offset[0]:offset[-1] + 1] = align_label
        return [(torch.tensor(i).cuda(), input_ids, segment_ids, img_feat, img_mask, total_label,
                 chunk_mask, full_offsets, input_mask, gather_index, align_pos)]

    def SNLIGPT_gen_collate(self, inputs):
        (id, input_ids, segment_ids, img_feat, img_mask, total_label, chunk_attention_mask,
         offsets, input_mask, gather_index, align_pos) = map(list, unzip(
            concat(inputs)))

        id = torch.stack(id, dim=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)

        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]
        input_mask = torch.cat((input_mask, img_mask), -1)
        input_mask = input_mask.unsqueeze(1).repeat(1, input_mask.size(-1), 1)

        max_hypo = input_ids.size(-1)
        chunk_attention_mask_padd = []
        for item in chunk_attention_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((item.size(0), max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), -1)
            chunk_attention_mask_padd.append(item)
        chunk_attention_mask_padd = torch.stack(chunk_attention_mask_padd, 0)

        batch = {'id': id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'img_feat': img_feat, 'total_label': total_label, 'img_mask': img_mask,
                 'chunk_attention_mask': chunk_attention_mask_padd, 'offsets': offsets,
                 'input_mask': input_mask, 'gather_index': gather_index, 'align_pos': align_pos
                 }

        return batch


class SeqAlignPretrainDataset_v2_align_mask_only(Dataset):
    def __init__(self, bert_tokenizer, example_file, feat_file):
        self.bert_toker = bert_tokenizer
        self.add_od_labels = True
        self.example_file = pickle.load(open(example_file, 'rb'))
        self.feat_dict = pickle.load(open(feat_file, 'rb'))
        self.mask_prob = 0.15
        self.max_masked_tokens = 5

    def __len__(self):
        return len(self.example_file)

    def __getitem__(self, i):
        example = self.example_file[i]
        image_key = example['img_key']
        img_feat = self.feat_dict[image_key]['features'].cuda()
        img_mask = self.feat_dict[image_key]['img_mask'].cuda()
        sent = example['sentence']

        input_ids = self.bert_toker.encode(sent, add_special_tokens=True)

        seq_len = len(input_ids)
        masked_pos = torch.zeros(seq_len, dtype=torch.int).cuda()
        # randomly mask words for prediction, ignore [CLS] ans [SEP]
        candidate_masked_idx = list(range(1, len(input_ids) - 1))
        random.shuffle(candidate_masked_idx)
        num_masked = min(max(round(self.mask_prob * seq_len), 1), self.max_masked_tokens)
        num_masked = int(num_masked)
        masked_idx = candidate_masked_idx[:num_masked]
        masked_idx = sorted(masked_idx)
        masked_token = [input_ids[i] for i in masked_idx]
        masked_input_ids = input_ids.copy()
        for pos in masked_idx:
            if random.random() <= 0.8:
                # 80% chance to be a ['MASK'] token
                masked_input_ids[pos] = self.bert_toker.mask_token_id
            elif random.random() <= 0.5:
                # 10% chance to be a random word ((1-0.8)*0.5)
                # 注意是双闭区间
                index = randint(1, len(self.bert_toker.vocab) - 1)
                masked_input_ids[pos] = index
            else:
                # 10% chance to remain the same (1-0.8-0.1)
                pass
        masked_pos[masked_idx] = 1

        # 保证没有 0
        assert 0 not in masked_token
        assert len(masked_idx) == torch.sum(masked_pos)

        if num_masked < self.max_masked_tokens:
            masked_token = masked_token + ([0] * (self.max_masked_tokens - num_masked))
        masked_token = torch.tensor(masked_token).cuda()

        input_ids = torch.tensor(input_ids).cuda()
        masked_input_ids = torch.tensor(masked_input_ids).cuda()
        input_mask = torch.ones(input_ids.size(0)).cuda()
        segment_ids = torch.zeros_like(input_ids)

        chunk_mask = example['ChunkMask'].cuda()
        full_offsets = example['full_offsets']
        gather_index = []
        for idx, set in enumerate(full_offsets):
            set = torch.tensor(set).cuda()
            gather_index.extend([idx] * set.size(0))
        gather_index = torch.tensor(gather_index).cuda()

        total_label = torch.zeros(input_ids.size(0)).cuda()
        align_pos = torch.zeros(input_ids.size(0), dtype=torch.int).cuda()
        annot = example['annot']
        offsets = example['offsets']
        for idx, offset in enumerate(offsets):
            align_label = list(annot[idx].values())[0].squeeze(0)
            # gold image
            align_index = torch.nonzero(align_label)
            if align_index.size(0) > 0:
                align_label = align_index[0].item() + 1
                if align_label >= 50:
                    continue
                    # 最大图片长度为50
            else:
                continue
            align_pos[offset[0]:offset[-1] + 1] = 1
            total_label[offset[0]:offset[-1] + 1] = align_label
        return [(torch.tensor(i).cuda(), input_ids, segment_ids, img_feat, img_mask, total_label,
                 chunk_mask, full_offsets, input_mask, masked_input_ids, masked_pos, masked_token, gather_index,
                 align_pos)]

    def SNLIGPT_gen_collate(self, inputs):
        (id, input_ids, segment_ids, img_feat, img_mask, total_label, chunk_attention_mask,
         offsets, input_mask, masked_input_ids, masked_pos, masked_token, gather_index, align_pos) = map(list, unzip(
            concat(inputs)))

        # id = torch.stack(id, dim=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).unsqueeze(1)
        masked_input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=0).unsqueeze(1)
        input_ids = torch.cat((input_ids, masked_input_ids), 1)
        masked_pos = pad_sequence(masked_pos, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)
        masked_token = torch.stack(masked_token, dim=0)
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)

        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]
        input_mask = torch.cat((input_mask, img_mask), -1)
        input_mask = input_mask.unsqueeze(1).repeat(1, input_mask.size(-1), 1)

        max_hypo = input_ids.size(-1)
        chunk_attention_mask_padd = []
        for item in chunk_attention_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((item.size(0), max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), -1)
            chunk_attention_mask_padd.append(item)
        chunk_attention_mask_padd = torch.stack(chunk_attention_mask_padd, 0)

        batch = {'id': id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'img_feat': img_feat, 'total_label': total_label, 'img_mask': img_mask,
                 'chunk_attention_mask': chunk_attention_mask_padd, 'offsets': offsets,
                 'input_mask': input_mask, 'masked_pos': masked_pos, 'masked_token': masked_token,
                 'gather_index': gather_index, 'align_pos': align_pos
                 }

        return batch


class SeqAlignPretrainDataset_v3(Dataset):
    def __init__(self, bert_tokenizer, example_file, feat_file):
        self.bert_toker = bert_tokenizer
        self.add_od_labels = True
        self.example_file = pickle.load(open(example_file, 'rb'))
        self.feat_dict = pickle.load(open(feat_file, 'rb'))

    def __len__(self):
        return len(self.example_file)

    def __getitem__(self, i):
        example = self.example_file[i]
        image_key = example['img_key']
        img_feat = self.feat_dict[image_key]['features'].cuda()
        img_mask = self.feat_dict[image_key]['img_mask'].cuda()
        sent = example['sentence']

        input_ids = self.bert_toker.encode(sent, add_special_tokens=False)
        input_ids = [self.bert_toker.cls_token_id] + input_ids + [self.bert_toker.sep_token_id]
        input_ids = torch.tensor(input_ids).cuda()
        input_mask = torch.ones(input_ids.size(0)).cuda()
        segment_ids = torch.zeros_like(input_ids)

        total_label = torch.zeros(input_ids.size(0)).cuda()
        align_pos = torch.zeros(input_ids.size(0), dtype=torch.int).cuda()
        annot = example['annot']
        offsets = example['offsets']
        for idx, offset in enumerate(offsets):
            align_label = list(annot[idx].values())[0].squeeze(0)
            # gold image
            align_index = torch.nonzero(align_label)
            if align_index.size(0) > 0:
                align_label = align_index[0].item() + 1
                if align_label >= 50:
                    continue
            else:
                continue
            align_pos[offset[0]:offset[-1] + 1] = 1
            total_label[offset[0]:offset[-1] + 1] = align_label
        return [
            (torch.tensor(i).cuda(), input_ids, segment_ids, img_feat, img_mask, total_label, input_mask, align_pos)]

    def SNLIGPT_gen_collate(self, inputs):
        (id, input_ids, segment_ids, img_feat, img_mask, total_label, input_mask, align_pos) = map(list, unzip(
            concat(inputs)))

        id = torch.stack(id, dim=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]
        input_mask = torch.cat((input_mask, img_mask), -1)

        batch = {'id': id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'img_feat': img_feat, 'total_label': total_label, 'img_mask': img_mask,
                 'input_mask': input_mask, 'align_pos': align_pos
                 }

        return batch


class SNLISeqAlignChunkDataset_v7(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, SNLI_example_file, chunk_mask_file, add_cls,
                 max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
                 is_train=True):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.label2id = {'neutral': 0,
                         'contradiction': 1,
                         'entailment': 2}
        self.SNLI_annot_dict = self.read_example(SNLI_example_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_hypo_len = max_hypo_len
        self.max_seq_len = max_seq_length
        self.SNLI_example_file = SNLI_example_file
        self.chunk_mask_dict = pickle.load(open(chunk_mask_file, 'rb'))
        self.add_cls = add_cls

    def read_example(self, path):
        if os.path.isdir(path):
            data = []
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    file = os.path.join(root, name)
                    tmp_data = pickle.load(open(file, 'rb'))
                    data.extend(tmp_data)
        elif os.path.isfile(path):
            data = pickle.load(open(path, 'rb'))
        return data

    def del_example(self):
        # 删除一部分数据
        self.SNLI_annot_dict = []

    def read_del_example(self):
        data = []
        for root, dirs, files in os.walk(self.SNLI_example_file, topdown=False):
            for name in files:
                file = os.path.join(root, name)
                tmp_data = pickle.load(open(file, 'rb'))
                data.extend(tmp_data)
        self.SNLI_annot_dict = data

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example = self.SNLI_annot_dict[i]
        flickrID = example['Flickr30kID'].split('.')[0]
        input_ids = example['input_ids'].cuda()
        od_labels = example['od_labels'].cuda()
        segment_ids = example['segment_ids'].cuda()
        img_feat = example['img_feat'].cuda()
        label = torch.tensor(self.label2id[example['gold_label']]).cuda()
        expl = example['explanation']
        expl = self.gpt_toker.encode(expl)
        gpt_ids = expl
        gpt_ids = [self.gpt_toker.bos_token_id] + gpt_ids + [self.gpt_toker.eos_token_id]
        attn_mask = [1] * len(gpt_ids)
        gpt_ids = torch.tensor(gpt_ids).cuda()
        attn_mask = torch.tensor(attn_mask).cuda()
        syn_labels_ids = example['syn_labels_ids'].cuda()
        input_mask = example['input_mask'].cuda()

        offsets = self.chunk_mask_dict[example['pairID']]['offsets']
        chunk_mask = self.chunk_mask_dict[example['pairID']]['mask'].cuda()
        gather_index = []
        for idx, set in enumerate(offsets):
            set = torch.tensor(set).cuda()
            gather_index.extend([idx] * set.size(0))
        gather_index = torch.tensor(gather_index).cuda()

        return [(torch.tensor(int(flickrID)).cuda(), input_ids, segment_ids, input_mask, img_feat, label, gpt_ids,
                 attn_mask, syn_labels_ids, od_labels, chunk_mask, offsets, gather_index)]

    def SNLIGPT_gen_collate(self, inputs):
        (img_id, input_ids, segment_ids, input_mask, img_feat, label, gpt_ids, attn_mask, syn_labels_ids, od_labels,
         chunk_mask, offsets, gather_index) = map(list, unzip(concat(inputs)))
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
        # gather_index=pad_sequence(gather_index, batch_first=True, padding_value=0)
        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': label, 'expl_ids': gpt_ids,
                 'attn_mask': attn_mask, 'syn_labels_ids': syn_labels_ids_padd, 'od_labels': od_labels,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets
                 }

        return batch


class SeqAlignPretrainDataset_vcr(Dataset):
    def __init__(self, bert_tokenizer, example_file, VCR_chunk_mask_file, feat_file):
        self.bert_toker = bert_tokenizer
        self.VCR_annot_dict = pickle.load(open(example_file, 'rb'))
        self.image_feat_dict = pickle.load(open(feat_file, 'rb'))
        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token

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

            # toked_txt_region_tokens = example['toked_txt_region_tokens']
            # toked_txt_region_tokens_a = example['toked_txt_region_tokens_a'][ans_idx]
            # region_tokens = [0] + toked_txt_region_tokens + [0] + toked_txt_region_tokens_a + [0]
            # region_tokens = torch.tensor(region_tokens).cuda()

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

            outputs.append((example['img_id'], input_ids, segment_ids, input_mask, img_feat, img_mask,
                            chunk_mask, gather_index, offsets, align_pos, total_label))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):
        (id, input_ids, segment_ids, input_mask, img_feat, img_mask, chunk_attention_mask,
         gather_index, offsets, align_pos, total_label) = map(list, unzip(
            concat(inputs)))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)

        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]
        input_mask = torch.cat((input_mask, img_mask), -1)
        input_mask = input_mask.unsqueeze(1).repeat(1, input_mask.size(-1), 1)

        max_hypo = input_ids.size(-1)
        chunk_attention_mask_padd = []
        for item in chunk_attention_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((item.size(0), max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), -1)
            chunk_attention_mask_padd.append(item)
        chunk_attention_mask_padd = torch.stack(chunk_attention_mask_padd, 0)

        batch = {'id': id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'img_feat': img_feat, 'total_label': total_label, 'img_mask': img_mask,
                 'chunk_attention_mask': chunk_attention_mask_padd, 'offsets': offsets,
                 'input_mask': input_mask, 'gather_index': gather_index, 'align_pos': align_pos
                 }

        return batch
