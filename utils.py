import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import json
import numpy as np
import pickle
import os
import torch
from sklearn.metrics import f1_score, accuracy_score

def convert_lines(df, vocab, bpe, max_seq_length):
    outputs = np.zeros((len(df), max_seq_length))
    
    cls_id = 0 # <s>
    eos_id = 2 # </s>
    pad_id = 1

    pbar = tqdm(df.iterrows(), total=len(df))
    pbar.set_description("BPE encode")
    for idx, row in pbar:
        subwords = '<s> ' + bpe.encode(row.text) + ' </s>'
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ]*(max_seq_length - len(input_ids))
        outputs[idx,:] = np.array(input_ids)
    return outputs

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_f1_score(y_true, y_pred):
    new_y_true = y_true.squeeze().detach().cpu().numpy()
    new_y_pred = y_pred.squeeze().detach().cpu().numpy() > 0.5
    new_y_pred = list(map(lambda x: int(x), new_y_pred))
    return f1_score(new_y_true, new_y_pred, average='binary')


def get_accuracy(y_true, y_pred):
    new_y_true = y_true.squeeze().detach().cpu().numpy()
    new_y_pred = y_pred.squeeze().detach().cpu().numpy() > 0.5
    new_y_pred = list(map(lambda x: int(x), new_y_pred))
    return accuracy_score(new_y_true, new_y_pred)