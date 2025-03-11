
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
import torch



corpus_path = '/home/mol_model/data/corpus.txt'


tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files = corpus_path,
    vocab_size = 52000,
    min_frequency = 2,
    show_progress = True,
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] 
)


tokenizer.post_processor = RobertaProcessing(
        sep = ('</s>', tokenizer.token_to_id('</s>')), 
        cls = ('<s>', tokenizer.token_to_id('<s>'))
    )


tokenizer_path = '/home/mol_model/tokenizer'
tokenizer.save_model(tokenizer_path)