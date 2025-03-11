from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
import os 
import onto  
import numpy as np 
import torch 
from dataset import * 
from model import * 


def train(model, dataloader, optimizer, device, criterion1, criterion2):
    for  (*x, y1, y2) in tqdm(dataloader):
         
        for i in range(len(x)):
            x[i] = x[i].to(device)
        y1 = y1.squeeze(2).to(device)   # torch.Size([b, 44261])
        y2 = y2.squeeze(1).to(device)  # torch.Size([b])
        pred1, pred2 =  model(*x)   # torch.Size([b, 44261])  torch.Size([b, 3])
        pred1 = torch.sigmoid(pred1)    # # torch.Size([b, 44261])
        loss =   criterion1(pred1, y1.float())  + criterion2(pred2, y2) #+ simcse_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)


fname = os.path.join('/home/Data', 'go.obo')

go = onto.Ontology(fname, with_rels=True, include_alt_ids=False)
# dat_fin = 'go_label.txt'
 
dataset = GOText2Dataset(go)
BATCH_SIZE = 8
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) 

LEARNING_RATE = 1e-5
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = OntoLMForSingleCLS(len(set(go.ont.keys())), 3)
 

param_optimizer = list(model.named_parameters())
# print(param_optimizer)
no_decay = [ 'model.linear', 'model.pool' , 'bias', 'LayerNorm.bias', 'LayerNorm.weight', 'cls_linear', 'neighbor_linear']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


# print(optimizer_grouped_parameters)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

# # print(optimizer_grouped_parameters)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # full training

model.to(device)
criterion1 = nn.BCELoss()
criterion2 = nn.CrossEntropyLoss()
N_EPOCHS = 50
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch + 1}")   
    train(model, dataloader, optimizer, device, criterion1, criterion2)


torch.save(model.model.state_dict(), '/home/struct2vec-neighbor.pth')