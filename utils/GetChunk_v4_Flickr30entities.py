import os
from torch.utils.data import Dataset
import jsonlines

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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

annot_file = './e-ViL-main/AlignLabelDev_example.pkl'

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
        outputs = []
        for sent_dict in example['SentAlign']:
            sentence = sent_dict['sentence']
            annot = sent_dict['annot']
            input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).cuda()
            mask_len = input_ids.size(0)
            attn_mask = torch.ones(mask_len).cuda()
            outputs.append(
                (input_ids, mask_len, attn_mask, annot, i, sentence))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):
        (input_ids, mask_len, attn_mask, annot, key, sentence) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        batch = {'mask_len': mask_len, 'attn_mask': attn_mask, 'input_ids': input_ids,
                 'annot': annot, 'key': key, 'sentence': sentence}
        return batch


dataset = SNLI_dataset()
dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False, collate_fn=dataset.SNLIGPT_gen_collate,
                        drop_last=False)
pbar = tqdm(total=len(dataloader))

key_hist = []
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
                # O
                chunk_offset.append(i)
        if len(tmp_chunk) != 0:
            chunk_offset.append(tmp_chunk)
        # SEP
        total_mask[mask_len - 1, :mask_len] = 1

        # 保证entities在同一chunk内
        begin = 1
        annot = annots[idx]
        annot_ids = []

        offsets = []
        for phrase in annot:
            phrase_ids = torch.tensor(tokenizer.encode(list(phrase.keys())[0], add_special_tokens=False)).cuda()
            for i in range(begin, mask_len - phrase_ids.size(0)):
                if torch.equal(input_ids[idx, i:phrase_ids.size(0) + i], phrase_ids):
                    total_mask[i:phrase_ids.size(0) + i, i:phrase_ids.size(0) + i] = 1
                    total_mask[i:phrase_ids.size(0) + i, :i] = 0
                    total_mask[i:phrase_ids.size(0) + i, phrase_ids.size(0) + i:] = 0
                    total_mask[1:i, i:phrase_ids.size(0) + i] = 0
                    total_mask[phrase_ids.size(0) + i:-1, i:phrase_ids.size(0) + i] = 0
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
        if len(his_list) != mask_len - 2:
            print()
        # 找到对应的index
        if id in key_hist:
            sent_index += 1
        else:
            sent_index = 0
            key_hist.append(id)
        AlignData[id]['SentAlign'][sent_index]['ChunkMask'] = total_mask
        AlignData[id]['SentAlign'][sent_index]['offsets'] = offsets
        AlignData[id]['SentAlign'][sent_index]['full_offsets'] = sort_chunk_offset
    pbar.update(1)

pickle.dump(AlignData, open('./e-ViL-main/AlignLabelDev_example.pkl', 'wb'))
