import os

import numpy as np
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

class Constants:
    OUTPUT_LABELS = ['0', 'B-Claim', 'I-Claim', 'B-Premise', 'I-Premise']
    LABELS_TO_IDS = {v: k for k, v in enumerate(OUTPUT_LABELS)}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataFrame(folder):
    names, texts = [], []
    for f in tqdm(list(os.listdir(folder))):
        names.append(f.replace('.txt', ''))
        texts.append(open(folder + '/' + f, 'r', encoding='utf-8').read())
    df_texts = pd.DataFrame({'id': names, 'text': texts})

    df_texts['text_split'] = df_texts.text.str.split()
    print('Completed tokenizing texts.')
    return df_texts


def ner(df_texts, df_gold):
    all_entities = []
    for _, row in tqdm(df_texts.iterrows(), total=len(df_texts)):
        total = len(row.text_split)
        entities = ['0'] * total
        if '.csv' in row['id']:
            all_entities.append(entities)
            break
        for _, row2 in df_gold[df_gold['id'] == int(row['id'])].iterrows():
            discourse = row2['tipo']
            if discourse == 'MajorClaim':
                discourse = 'Claim'
            if row2['text_start'] != 1:
                text_prev = row.text[:row2['text_start']]
                text_prev_split = text_prev.split()
                l = len(text_prev_split)
                text = row.text[row2['text_start']:row2['text_end']]
                for i in range(len(text.split())):
                    if i == 0:
                        entities[l + i] = f'B-{discourse}'
                    else:
                        entities[l + i] = f'I-{discourse}'
            else:
                text = row.text[row2['text_start']:row2['text_end']]
                for i in range(len(text.split())):
                    if i + 1 == 1:
                        entities[i] = f'B-{discourse}'
                    else:
                        entities[i] = f'I-{discourse}'
        all_entities.append(entities)

    df_texts['entities'] = all_entities
    print('Completed mapping discourse to each token.')
    return df_texts



def preprocess(folder):
    df_texts = get_dataFrame(folder)

    if 'train' in folder:
        df_gold = pd.read_csv(folder + '/train.csv')
    elif 'validate' in folder:
        df_gold = pd.read_csv(folder + '/validate.csv')
    elif 'test' in folder:
        df_gold = pd.read_csv(folder + '/test.csv')
    else:
        raise Exception('Ruta Incorrecta')

    df_texts = ner(df_texts, df_gold)
    return df_texts, df_gold


class DataSet(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sentence = self.data.text_split[index]
        word_labels = self.data.entities[index]

        encoding = self.tokenizer(sentence, is_split_into_words=True,
                                  padding='max_length',
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.max_len)

        labels = [Constants.LABELS_TO_IDS[label] for label in word_labels]

        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                try:
                    encoded_labels[idx] = labels[i]
                    i += 1
                except IndexError:
                    continue

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item




