import numpy as np
from ge import Struc2Vec
import networkx as nx
import pickle

folder_path = '/home/Data/'

G = nx.read_edgelist( folder_path + 'go-terms.edgelist',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)]) #read graph

model = Struc2Vec(G, walk_length=10, num_walks=80, workers=4, verbose=40, )  
model.train(window_size = 5, iter = 3) 
embeddings = model.get_embeddings() 

save_path = folder_path + 'struc2vec_go_emb.pkl'
with open(save_path, 'wb') as fp:
    pickle.dump(embeddings, fp, protocol=pickle.HIGHEST_PROTOCOL)


# print(embeddings['44260'].shape)

