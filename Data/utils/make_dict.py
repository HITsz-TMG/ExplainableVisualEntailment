import os
import json
import xlwt

def make_dict():
    file_path = '/raid/yq/UNITER/pretrain/vocab.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        w2i = {}
        for i in range(0, len(lines)):
            w2i[lines[i].split('\n')[0]] = i
    sp_path = '/raid/yq/UNITER/pretrain/txt_db/vcr_val.db/special_tokens.json'
    with open(sp_path, 'r') as f:
        data = f.readlines()
    sp = json.loads(data[0])
    for i in range(len(lines),len(lines)+len(sp)):
        w2i[sp[i-len(lines)]] = i
    i2w = dict(zip(w2i.values(), w2i.keys()))
    return w2i,i2w

def add_sp():
    file_path = './pretrain/vocab.txt'
    sp_path='/raid/yq/UNITER/pretrain/txt_db/vcr_val.db/special_tokens.json'
    with open(sp_path,'r') as f:
        data=f.readlines()
    sp=json.loads(data[0])
    file_path = './pretrain/vocab.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()

def count_len():
    with open('/raid/yq/UNITER/vcr1annots/train.jsonl', 'r') as load_f:
        data = load_f.readlines()
    len_dic={}
    for line in data:
        load_dict = json.loads(line)
        for i in range(4):
            lenth=len(load_dict['rationale_choices'][i])
            if lenth not in len_dic.keys():
                len_dic[lenth] =1
            else:
                len_dic[lenth]+=1
    workbook = xlwt.Workbook()
    sheet=workbook.add_sheet('rs_len_count')
    i=0
    for key, value in len_dic.items():
        sheet.write(i,0,key)
        sheet.write(i, 1, value)
        i+=1
    workbook.save('rs_len_count.xls')

if __name__ == '__main__':
    #count_len()
    #add_sp()
    make_dict()