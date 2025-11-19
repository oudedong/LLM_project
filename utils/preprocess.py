import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers import AutoTokenizer

def normalize_to_length1(x: torch.Tensor, eps: float = 1e-12):
    """
    x: (batch, channels, length)
    벡터의 방향만 남김
    """
    norm = (x.pow(2).sum(dim=1, keepdim=True).sqrt()).clamp_min(eps)
    return x / norm

#데이터 로딩
def load_data_and_tokenize(tok, path):
    ds = load_dataset('csv',data_files=path, split='train') 
    ds = ds.map(lambda batch: tok(batch['full_text'],truncation=True),batched=True)
    ds = ds.remove_columns([c for c in ds.column_names if c not in ['input_ids','attention_mask', 'token_type_ids', 'generated']])
    # ds = ds['train'].train_test_split(test_size=0.1, seed=42)
    return ds
def balanced_train_test_split(ds:Dataset, ratio=0.2, seed=42):
    ds_generated = ds.filter(lambda x: x['generated']==1).train_test_split(test_size=ratio, seed=seed)
    ds_non_generated = ds.filter(lambda x: x['generated']==0).train_test_split(test_size=ratio, seed=seed)
    
    ds_train = concatenate_datasets([ds_generated['train'], ds_non_generated['train']])
    ds_test  = concatenate_datasets([ds_generated['test'], ds_non_generated['test']])
    return ds_train, ds_test

#긴거 자르는 함수