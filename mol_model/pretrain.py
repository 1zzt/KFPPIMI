import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
import yaml
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from torch.utils.data import Dataset

from dataset import *
import logging

logging.basicConfig(filename='/home/mol_model/checkpoint/training_log.txt', level=logging.INFO)

print('Start running...', datetime.datetime.now())

save_path = '/home/mol_model/checkpoint'


with open('/home/mol_model/config_pretrain.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

configuration = RobertaConfig(**config)


# 
tokenizer = RobertaTokenizer('/home/mol_model/tokenizer/vocab.json',
                             '/home/mol_model/tokenizer/merges.txt')


train_dataset = get_data(tokenizer)

model = RobertaForMaskedLM(configuration)


data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer, mlm = True, mlm_probability = 0.15
)


training_args = TrainingArguments(
    output_dir = save_path,
    # overwrite_output_dir = True,
    num_train_epochs = config['epochs'], 
    per_device_train_batch_size = config['batch_size'], 
    save_steps = 1000,
    # save_total_limit = 5,
    save_safetensors = False,
    save_strategy = 'epoch',
    logging_dir = save_path,
    logging_steps=1000,

)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator, 
    train_dataset = train_dataset,
    # compute_metrics = compute_metrics,
)


trainer.train()

trainer.save_model()

for obj in trainer.state.log_history:
    logging.info(str(obj))


print('Running Completed!', datetime.datetime.now())


