from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer

import os

save_dir = "/home/pparth2/scratch/causal-lm-project/causal-lm-analysis/cached"
# GPT2

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

model.save_pretrained(os.path.join(save_dir,'gpt2')) #check the path
tokenizer.save_pretrained(os.path.join(save_dir,'gpt2')) # check the path

print('GPT2 done...')

# GPT2-Large

tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
model = AutoModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

model.save_pretrained(os.path.join(save_dir,'gpt2-large')) #check the path
tokenizer.save_pretrained(os.path.join(save_dir,'gpt2-large')) # check the path

print('GPT2-Large done...')
