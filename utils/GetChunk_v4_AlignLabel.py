import os
from torch.utils.data import Dataset
import jsonlines

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
from tqdm import tqdm
import torch
from transformers import BertTokenizerFast
from toolz.sandbox import unzip
from cytoolz import concat
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import BertModelWithHeads

model = BertModelWithHeads.from_pretrained("bert-base-uncased")
adapter_name = model.load_adapter("AdapterHub/bert-base-uncased-pf-conll2000", source="hf")
model.active_adapters = adapter_name
model.cuda()
model.eval()

annot_file = './e-ViL-main/AlignLabelTrain_example.pkl'

tokenizer = BertTokenizerFast.from_pretrained(
    './Oscar/pretrained_models/image_captioning/pretrained_large/pretrained_large/checkpoint-1410000/',
    do_lower_case=True, )

id2label = model.config.id2label
result = {}

AlignData = pickle.load(open(annot_file, 'rb'))


class SNLI_dataset(Dataset):
    def __init__(self):
        self.data = AlignData

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        sentence = example['sentence']
        annot = example['annot']
        input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).cuda()
        mask_len = input_ids.size(0)
        attn_mask = torch.ones(mask_len).cuda()

        return [(input_ids, mask_len, attn_mask, annot, i, sentence)]

    def SNLIGPT_gen_collate(self, inputs):
        (input_ids, mask_len, attn_mask, annot, key, sentence) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        batch = {'mask_len': mask_len, 'attn_mask': attn_mask, 'input_ids': input_ids,
                 'annot': annot, 'key': key, 'sentence': sentence}
        return batch


dataset = SNLI_dataset()
dataloader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False, collate_fn=dataset.SNLIGPT_gen_collate,
                        drop_last=False)
pbar = tqdm(total=len(dataloader))

for step, batch in enumerate(dataloader):
    input_ids = batch['input_ids']
    annots = batch['annot']
    attn_mask = batch['attn_mask']
    logits = model(input_ids, attention_mask=attn_mask).logits
    lens = batch['mask_len']
    class_res = logits.max(dim=-1)[1]
    sentences = batch['sentence']
    sent_index = 0
    for idx, id in enumerate(batch['key']):
        classes = class_res[idx].tolist()
        chunk_offset = []
        token_classs_list = []
        tmp_chunk = []
        mask_len = lens[idx]
        total_mask = torch.eye(mask_len)
        total_mask[0, :mask_len] = 1
        for i in range(1, mask_len - 1):
            token_class = id2label[classes[i]]
            token_classs_list.append(token_class)
            if token_class[0] == 'B':
                if len(tmp_chunk) != 0:
                    chunk_offset.append(tmp_chunk)
                tmp_chunk = [i]
            elif token_class[0] == 'I':
                for index in tmp_chunk:
                    total_mask[index][i] = 1
                    total_mask[i][index] = 1
                tmp_chunk.append(i)
            else:
                #在O不为最后一位的情况下，需要判定O是否在BI之内
                if i!=mask_len-2 and len(tmp_chunk) != 0 and id2label[classes[i+1]][0]=='I' :
                    for index in tmp_chunk:
                        total_mask[index][i] = 1
                        total_mask[i][index] = 1
                    tmp_chunk.append(i)
                else:
                    #O为最后一位，或不在BI之间
                    chunk_offset.append(i)
        if len(tmp_chunk) != 0:
            chunk_offset.append(tmp_chunk)
        # SEP
        total_mask[mask_len - 1, :mask_len] = 1


        begin = 1
        annot = annots[idx]
        annot_ids = []
        offsets = []
        for phrase in annot:
            phrase_ids = torch.tensor(tokenizer.encode(list(phrase.keys())[0], add_special_tokens=False)).cuda()
            for i in range(begin, mask_len - phrase_ids.size(0)):
                if torch.equal(input_ids[idx, i:phrase_ids.size(0) + i], phrase_ids):
                    offsets.append([i, i + phrase_ids.size(0) - 1])
                    begin = i + phrase_ids.size(0)
                    break

        sort_chunk_offset = []
        his_list = []
        for i in range(1, mask_len - 1):
            chunk = torch.nonzero(total_mask[i]).squeeze(-1).tolist()
            if chunk[0] not in his_list:
                sort_chunk_offset.append(chunk)
                his_list.extend(chunk)
        assert len(his_list)==mask_len-2

        AlignData[id]['ChunkMask'] = total_mask
        AlignData[id]['offsets'] = offsets
        AlignData[id]['full_offsets'] = sort_chunk_offset
    pbar.update(1)

pickle.dump(AlignData, open('./e-ViL-main/AlignLabelTrain_example.pkl', 'wb'))
