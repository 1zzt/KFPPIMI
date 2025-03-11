import torch
import torch.nn as nn

from transformers import  AutoModel, AutoTokenizer
import torch.nn.functional as F


class ProcessingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProcessingLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.activation1 = nn.GELU()
        self.linear2 = nn.Linear(512, 1024)
        self.activation2 = nn.GELU()
        self.linear3 = nn.Linear(1024, output_dim)
        self.layernorm = nn.LayerNorm(output_dim)
        self.activation3 = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.layernorm(x)
        x = self.activation3(x)
        return x


# TODO: change to prompt learning, the ontology structure as prompt

class OntoLanguageModel(nn.Module):
    def __init__(self):
        super(OntoLanguageModel, self).__init__()
        
        self.llm_model = AutoModel.from_pretrained('/home/zqzhangzitong/project/PPIMI/GO/oubiobert')
        self.bert_model = nn.Sequential(*list(self.llm_model.children())[0:])
        # Linear layer
        embedding_size = self.llm_model.config.hidden_size
        self.linear = ProcessingLayer(128, embedding_size)

        self.pool =  nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, embedding_size),
        )

    
    def get_extended_attention_mask(self, attention_mask):
        
       
        extended_attention_mask = attention_mask[:, None, None, :]
        

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float).min
        return extended_attention_mask
         
    def forward(self, onto_struct2vec, input_ids, token_type_ids, mask):
        batch_size = onto_struct2vec.shape[0]
        # linear_output: torch.Size([64, 1, 768])
        linear_output = self.linear(onto_struct2vec).unsqueeze(1)

        
        # input format: [CLS] onto_emb [SEP] sentence  the onehot embedding skip training 
        with torch.no_grad():   
            # token embedding：word embedding 将词汇表中的词或短语，映射为固定长度向量
            # h: torch.Size([b, 256, 768])  input_ids: tokenid tensor列表[b, 256]
            h = self.bert_model[0](input_ids=input_ids, token_type_ids=token_type_ids)
        # print(linear_output.size(), h.size(), input_ids.size())  
        h[:,1:2,:] = linear_output  # torch.Size([64, 1, 768])
        # mask shoud be  bz, L, 768 ; extended_attention_mask: torch.Size([64, 1, 1, 256])
        extended_attention_mask = self.get_extended_attention_mask(mask)    
        last_hidden_state = self.bert_model[1](h, extended_attention_mask).last_hidden_state    # last_hidden_state: torch.Size([64, 256, 768])
        # cls vec sep word1 word2 ; bert_out: torch.Size([64, 768])
        bert_out = (last_hidden_state[:,1] + torch.mean(last_hidden_state[:,3:-1], 1))/2
        # bert_out: torch.Size([64, 768])
        bert_out = self.pool(bert_out)
        return bert_out
 


class OntoLMForSingleCLS(nn.Module):
    def __init__(self, neighbor_size, anc_size):
        super(OntoLMForSingleCLS, self).__init__()

        self.model = OntoLanguageModel()
        # self.model = OntoLanguageModelV2()
        # for BP MF CC prediction
        self.cls_linear = nn.Sequential(
            nn.Linear(self.model.llm_model.config.hidden_size, self.model.llm_model.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.model.llm_model.config.hidden_size),
            nn.Linear(self.model.llm_model.config.hidden_size, anc_size),
        )
        # for neighbor prediction
        self.neighbor_linear = nn.Sequential(
            nn.Linear(self.model.llm_model.config.hidden_size, self.model.llm_model.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.model.llm_model.config.hidden_size),
            nn.Linear(self.model.llm_model.config.hidden_size, neighbor_size)
        )
    
    def forward(self,  input_ids, token_type_ids, mask, onto_struct2vec):

        bert_out = self.model(onto_struct2vec, input_ids, token_type_ids, mask)

        output1 = self.neighbor_linear(bert_out)    # torch.Size([64, 44261])
        output2 = self.cls_linear(bert_out) # torch.Size([64, 3])
         
        return output1, output2


