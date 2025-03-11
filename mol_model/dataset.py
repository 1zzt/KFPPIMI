
import pandas as pd
import torch
import os
from datasets import Dataset

import datetime



def get_data(tokenizer):

    
    print('Data loading...\t', datetime.datetime.now())


    corpus_token_path = "/home/mol_model/data/tokens.pt"
    
    
    tokens_list = torch.load(corpus_token_path, map_location='cpu')

    data = {
        'input_ids': tokens_list,
        # 'mol_feature': graph_fea
    }

    print('Data loading completed...\t', datetime.datetime.now())

    dataset = Dataset.from_dict(data)

    return dataset


def get_data_files(train_path):
    if os.path.isdir(train_path):
        return [
            os.path.join(train_path, file_name) for file_name in os.listdir(train_path)
        ]
    elif os.path.isfile(train_path):
        return train_path

    raise ValueError("Please pass in a proper train path")

