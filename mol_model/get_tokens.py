from transformers import LineByLineTextDataset
import torch
from transformers import RobertaTokenizer
from tqdm import tqdm


text_path = "/home/mol_model/data/corpus.txt"

corpusfile_path = "/home/mol_model/data/corpusfile"


# Used when token loading is too slow
BATCH_SIZE = 1000
with open(text_path, 'r') as f:
    lines = f.readlines()
    for i in tqdm(range(0, len(lines), BATCH_SIZE), total=len(lines)//BATCH_SIZE):
        save_path = f"{corpusfile_path}/corpus-{i}.txt"
        with open(save_path, 'w') as f:
            f.writelines(lines[i:i+BATCH_SIZE])



tokenizer = RobertaTokenizer('/home/mol_model/tokenizer/vocab.json',
                             '/home/mol_model/tokenizer/merges.txt')


tokens_list = []


save_path = '/home/mol_model/data/tokens.pt'
for i in tqdm(range(0, len(lines), BATCH_SIZE), total=len(lines)//BATCH_SIZE):
    load_path = f"{corpusfile_path}/corpus-{i}.txt"


    tokens = LineByLineTextDataset( 
        tokenizer = tokenizer,
        file_path = load_path,
        block_size = 512   
        )
    # print(len(tokens))
    for j in range(len(tokens)):
        tokens_list.append(tokens[j]['input_ids'])


torch.save(tokens_list, save_path)
