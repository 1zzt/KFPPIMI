from torch.utils.data import Dataset, DataLoader
import os 
import onto  
import numpy as np 
from tqdm import tqdm 
import torch 
import pickle
from transformers import  AutoModel, AutoTokenizer


def encode_text_v1(caption,tokenizer, max_position_embeddings=256):

    tokens = tokenizer.encode(caption)[:max_position_embeddings] 

    segment_ids =   [0]*(len(tokens[:max_position_embeddings])) 
    input_mask = [1]*len(tokens) 
    n_pad =  max_position_embeddings - len(tokens)  
    tokens.extend([0]*n_pad)  
    segment_ids.extend([0]*n_pad)   
    input_mask.extend([0]*n_pad)   


    return tokens, segment_ids, input_mask


class GOText2Dataset(Dataset):
    def __init__(self, go_onto, go_label_file='/home/Data/go_label.txt', onto_type='go' ):
        self.go_onto = go_onto

        self.vocal_size = len(set(go_onto.ont.keys()))  # 44261
        self.inputs = []
        self.onto_embeddings = []
        self.labels1 = []
        self.labels2 = []

        self.tokenizer = AutoTokenizer.from_pretrained('/home/modelzoo/oubiobert')

        if onto_type == 'go':
            go_id_dict_pth = '/home/Data/go_id_dict'
            go_embed_pth = '/home/Data/struc2vec_go_emb.pkl'
            with open(go_id_dict_pth, 'rb') as fp:
                go_id_dict = pickle.load(fp)
            # 每个GO term编码为128维的向量
            with open(go_embed_pth, 'rb') as fp:
                embeddings_dict = pickle.load(fp)   # 44261
            with open(go_label_file) as f:
                for c, a in tqdm( enumerate(f)):
                    term, neighbors, namespace = a.strip().split('\t')
                    neigh_label = np.zeros(( self.vocal_size, 1 ))  # (44261, 1)
                    neighbors = [ int(t) for t in neighbors.split('!')]
                    neigh_label[ neighbors ] = 1
                    # 获取name和def
                    self.inputs.append( go_onto.ont[term]['name'] + ': ' + go_onto.ont[term]['def'] )
                    # 获取邻居标签，邻居位置为1，其他为0
                    self.labels1.append(  neigh_label )
                    # 获取GO类型标签0,1,2
                    self.labels2.append( int(namespace) )
                    # sturc2vec 向量
                    self.onto_embeddings.append( embeddings_dict[  str(go_id_dict[term])  ]) 
         
               
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # tokens, segment_ids, input_mask = encode_text(self.inputs[index], self.tokenizer)
        tokens, segment_ids, input_mask = encode_text_v1(self.inputs[index], self.tokenizer)

        
        return torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long) ,torch.tensor(input_mask, dtype = torch.long), self.onto_embeddings[index], torch.tensor(self.labels1[index]), torch.tensor([self.labels2[index]])
    
