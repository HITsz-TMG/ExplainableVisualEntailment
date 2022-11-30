import os
from torch.utils.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
SNLI_example_file_train = "./e-ViL-main/train_example_data.pkl"

tokenizer = BertTokenizerFast.from_pretrained(
    './Oscar/pretrained_models/image_captioning/pretrained_large/pretrained_large/checkpoint-1410000/',
    do_lower_case=True, )

id2label = model.config.id2label
result = {}


class SNLI_dataset(Dataset):
    def __init__(self):
        self.data = self.read_example(SNLI_example_file_train)

    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        hypo = self.data[i]['hypothesis'].lower()
        tokenize_bert = tokenizer(hypo, add_special_tokens=True)
        input_ids = torch.tensor(tokenize_bert.data['input_ids']).cuda()
        mask_len = input_ids.size(0)
        attn_mask = torch.ones(mask_len).cuda()
        ID = self.data[i]['pairID']
        return [(hypo, ID, mask_len, input_ids, attn_mask)]

    def SNLIGPT_gen_collate(self, inputs):
        (hypo, ID, mask_len, input_ids, attn_mask) = map(list, unzip(concat(inputs)))
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        batch = {'hypo': hypo, 'mask_len': mask_len, 'attn_mask': attn_mask, 'input_ids': input_ids, 'ID': ID}
        return batch


dataset = SNLI_dataset()
dataloader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False, collate_fn=dataset.SNLIGPT_gen_collate,drop_last=False)
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
        mask_len = min(mask_len, 40)
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
        assert len(his_list)==mask_len-2
        result[id] = {'mask': total_mask,
                      'offsets': sort_chunk_offset}
    pbar.update(1)

save_file = SNLI_example_file_train.split('/')[:-1]
save_file = os.path.join('/'.join(save_file), 'ChunkMaskTrain_v4.pkl')
pickle.dump(result, open(save_file, 'wb'))
