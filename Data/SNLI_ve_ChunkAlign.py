from torch.utils.data import Dataset
import torch
from toolz.sandbox import unzip
from cytoolz import concat
from torch.nn.utils.rnn import pad_sequence
import pickle
import jsonlines


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


class SNLI_VE_ChunkAlignDataset(Dataset):
    def __init__(self, bert_tokenizer, SNLI_example_file, flickr_feat_file, chunk_mask_file,
                 max_img_seq_length=50, max_seq_length=140, max_hypo_len=40,
                 is_train=True):
        self.bert_toker = bert_tokenizer
        self.label2id = {'neutral': 0,
                         'contradiction': 1,
                         'entailment': 2}
        self.SNLI_annot_dict = self.read_example(SNLI_example_file)
        self.image_feat_dict = self.read_feat(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_hypo_len = max_hypo_len
        self.max_img_seq_length = max_img_seq_length
        self.chunk_mask_dict = pickle.load(open(chunk_mask_file, 'rb'))

    def read_example(self, path):
        data_list = []
        with jsonlines.open(path) as jsonl_file:
            for line in jsonl_file:
                line['hypo'] = line['sentence2']
                data_list.append(line)
        return data_list

    def read_feat(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.SNLI_annot_dict)

    def __getitem__(self, i):
        example = self.SNLI_annot_dict[i]
        image_feat = self.image_feat_dict[example['Flickr30K_ID'] + '.jpg']
        flickrID = example['Flickr30K_ID']
        od_labels = image_feat['od_labels'].cuda()
        img_feat = image_feat['image_feat'].cuda()
        label = torch.tensor(self.label2id[example['gold_label']]).cuda()

        hypo = example['hypo'].lower()
        hypo_tokens = self.bert_toker.tokenize(hypo)
        if len(hypo_tokens) > self.max_hypo_len - 2:
            hypo_tokens = hypo_tokens[:self.max_hypo_len - 2]
        hypo_tokens = [self.bert_toker.cls_token] + hypo_tokens + [self.bert_toker.sep_token]
        input_ids = self.bert_toker.convert_tokens_to_ids(hypo_tokens)
        input_ids = torch.tensor(input_ids).cuda()
        input_mask = torch.ones(input_ids.size(0), dtype=torch.int64).cuda()
        segment_ids = torch.zeros(input_ids.size(0), dtype=torch.int64).cuda()
        image_mask = torch.ones(image_feat['gt_boxes'].size(0) + 1, dtype=torch.int64).cuda()
        if image_mask.size(0) > self.max_img_seq_length:
            image_mask = image_mask[:self.max_img_seq_length]
        offsets = self.chunk_mask_dict[example['pairID']]['offsets']
        chunk_mask = self.chunk_mask_dict[example['pairID']]['mask'].cuda()
        gather_index = []
        for idx, set in enumerate(offsets):
            set = torch.tensor(set).cuda()
            gather_index.extend([idx] * set.size(0))
        gather_index = torch.tensor(gather_index).cuda()
        return [(torch.tensor(int(flickrID)).cuda(), input_ids, segment_ids, input_mask, img_feat, label, image_mask,
                 od_labels, chunk_mask, gather_index, offsets)]

    def SNLIGPT_gen_collate(self, inputs):
        (img_id, input_ids, segment_ids, input_mask, img_feat, label, image_mask, od_labels,
         chunk_mask, gather_index, offsets) = map(list, unzip(concat(inputs)))
        od_labels = pad_sequence(od_labels, batch_first=True, padding_value=0)
        img_id = torch.stack(img_id, dim=0)

        img_feat = torch.stack(img_feat, dim=0)
        label = torch.stack(label, dim=0)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        max_hypo = input_mask.size(1)
        max_img = max([item.size(0) for item in image_mask])
        image_mask = pad_sequence(image_mask, batch_first=True, padding_value=0)
        img_feat = img_feat[:, :max_img]

        input_mask = torch.cat((input_mask, image_mask), -1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)
        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': label,
                 'od_labels': od_labels, 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index,
                 'offsets': offsets
                 }

        return batch
