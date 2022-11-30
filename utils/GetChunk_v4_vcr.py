import os
from torch.utils.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pickle
from tqdm import tqdm
import torch
from transformers import BertTokenizerFast
from toolz.sandbox import unzip
from cytoolz import concat
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import BertModelWithHeads
import json

model = BertModelWithHeads.from_pretrained("bert-base-uncased")
adapter_name = model.load_adapter("AdapterHub/bert-base-uncased-pf-conll2000", source="hf")
model.active_adapters = adapter_name
model.cuda()
model.eval()
VCR_example_file_train = './Oscar/datasets/VCR_UNITER_feat/train_example_data.pkl'

tokenizer = BertTokenizerFast.from_pretrained(
    './Oscar/pretrained_models/image_captioning/pretrained_large/pretrained_large/checkpoint-1410000/',
    do_lower_case=True, )
det_tokens = ["<|det%d|>" % i for i in range(45)]
tokenizer.add_special_tokens({"additional_special_tokens": det_tokens})
model.resize_token_embeddings(len(tokenizer))
id2label = model.config.id2label
result = {}


class VCR_dataset(Dataset):
    def __init__(self):
        self.data = pickle.load(open(VCR_example_file_train, 'rb'))
        self.cls = tokenizer.cls_token
        self.sep = tokenizer.sep_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        hypo = example['sent'].lower()
        hypo_tokens = tokenizer.tokenize(hypo, add_special_tokens=True)
        outputs = []

        for ans in example['answer_choices']:
            ans_tokens = tokenizer.tokenize(ans)
            input_tokens = hypo_tokens + ans_tokens + [self.sep]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda()
            mask_len = input_ids.size(0)
            attn_mask = torch.ones(mask_len).cuda()
            ID = i
            outputs.append((input_tokens, ID, mask_len, input_ids, attn_mask))
        return tuple(outputs)

    def VCRGPT_gen_collate(self, inputs):
        (hypo, ID, mask_len, input_ids, attn_mask) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        batch = {'hypo': hypo, 'mask_len': mask_len, 'attn_mask': attn_mask, 'input_ids': input_ids, 'ID': ID}
        return batch


dataset = VCR_dataset()
dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False, collate_fn=dataset.VCRGPT_gen_collate,
                        drop_last=False)
pbar = tqdm(total=len(dataloader))

for step, batch in enumerate(dataloader):
    hypo = batch['hypo']
    input_ids = batch['input_ids']
    attn_mask = batch['attn_mask']
    logits = model(input_ids, attention_mask=attn_mask).logits
    lens = batch['mask_len']
    class_res = logits.max(dim=-1)[1]
    for idx, id in enumerate(batch['ID']):
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
                # 在O不为最后一位的情况下，需要判定O是否在BI之内
                if i != mask_len - 2 and len(tmp_chunk) != 0 and id2label[classes[i + 1]][0] == 'I':
                    for index in tmp_chunk:
                        total_mask[index][i] = 1
                        total_mask[i][index] = 1
                    tmp_chunk.append(i)
                else:
                    # O为最后一位，或不在BI之间
                    chunk_offset.append(i)
        if len(tmp_chunk) != 0:
            chunk_offset.append(tmp_chunk)
        # SEP
        total_mask[mask_len - 1, :mask_len] = 1
        sort_chunk_offset = []
        his_list = []
        for i in range(1, mask_len - 1):
            chunk = torch.nonzero(total_mask[i]).squeeze(-1).tolist()
            if chunk[0] not in his_list:
                sort_chunk_offset.append(chunk)
                his_list.extend(chunk)
        assert len(his_list) == mask_len - 2
        if id in result.keys():
            result[id].append({'mask': total_mask.cpu(),
                               'offsets': sort_chunk_offset})
        else:
            result[id] = [{'mask': total_mask.cpu(),
                           'offsets': sort_chunk_offset}]
    pbar.update(1)

save_file = VCR_example_file_train.split('/')[:-1]
save_file = os.path.join('/'.join(save_file), 'ChunkMaskTrain_v4.pkl')
pickle.dump(result, open(save_file, 'wb'))
