import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import submitit
import os

from itertools import product
from pathlib import Path
from datetime import datetime

def get_features(text, model_name, target_dir, batch_size=16):
    model_dir = "/home/pparth2/scratch/causal-lm-project/causal-lm-analysis/cached"
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name))
    model = AutoModel.from_pretrained(os.path.join(model_dir, model_name), pad_token_id=tokenizer.eos_token_id).to(torch.device('cpu'))
    model = model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    #feats = []
    device = model.device

    feat_file = os.path.join(target_dir, 'features.npy')
    with open(feat_file, 'w') as f:
        for i in tqdm(range(0,len(text), batch_size)):
          tokenized_texts = tokenizer.batch_encode_plus(text[i:i+batch_size], return_tensors='pt', truncation=True, max_length=256, padding=True, add_special_tokens=False)['input_ids']
          if isinstance(tokenized_texts, list):
            tokenized_texts = torch.LongTensor(tokenized_texts).unsqueeze(0)
          out = model(input_ids=tokenized_texts.to(device), past_key_values=None,
                         output_hidden_states=True, return_dict=True)
          h = out.hidden_states[-1]  # (batch_size, seq_len, dim)
          for row in h[:, -1, :].detach().cpu().numpy():
              np.savetxt(f, [row])


    del tokenizer
    del model
    return None

def HyperFeatures(config):
    '''
    universal_dependencies: 'en_pronouns', 'en_esl', 'en_ewt', 'en_gum', 'en_gumreddit', 'en_lines', 'en_partut', 'en_pud'
    blimp: 'island_effects', 'anaphor_agreement', 's-selection', 'argument_structure', 'determiner_noun_agreement', 'subject_verb_agreement', 'ellipsis',
 'control_raising', 'quantifiers', 'irregular_forms', 'npi_licensing', 'binding', 'filler_gap_dependency'
    '''
    if config['dataset'] == 'universal_dependencies':
        dataset = load_dataset("universal_dependencies", config['sub_dataset'])[config['split']]['text']
    elif config['dataset'] == 'blimp':
        blimp_df = pd.read_csv('./blimp/blimp.csv')
        dataset = blimp_df[blimp_df.linguistics_term == config['sub_dataset']]["sentence_bad"].tolist() + blimp_df[blimp_df.linguistics_term == config['sub_dataset']]["sentence_good"].tolist()


    target_dir = os.path.join('Features', config['model'], config['dataset'], config['sub_dataset'], config['split'])

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    features = get_features(dataset, config['model'], target_dir, batch_size=config['batch_size'])

    #np.save(os.path.join(target_dir, 'features.npy'), features)

    return None


if __name__ == '__main__':

    sub_datasets = {
    'blimp': ['island_effects', 'anaphor_agreement', 's-selection', 'argument_structure', 'determiner_noun_agreement', 'subject_verb_agreement', 'ellipsis',
 'control_raising', 'quantifiers', 'irregular_forms', 'npi_licensing', 'binding', 'filler_gap_dependency'],
 'universal_dependencies':['en_pronouns', 'en_esl', 'en_ewt', 'en_gum', 'en_gumreddit', 'en_lines', 'en_partut', 'en_pud'],
 }
    splits = {
    'en_pronouns': ['test'],
    'en_esl': ['train', 'validation', 'test'], #train
    'en_ewt': ['train', 'validation', 'test'],
    'en_gum': ['train', 'validation', 'test'],
    'en_gumreddit': ['train', 'validation', 'test'],
    'en_lines': ['train', 'validation', 'test'],
    'en_partut': ['train', 'validation', 'test'],
    'en_pud': ['test']
    }
    PARAM_GRID = list(product(
    ['gpt2'],
    ['universal_dependencies', 'blimp']
    )
    )

    h_param_list = []

    for param_ix in range(len(PARAM_GRID)):

        params = PARAM_GRID[param_ix]

        model, dataset = params

        for sub_data in sub_datasets[dataset]:
            if 'blimp' not in dataset:
                for split in splits[sub_data]:
                    h_param_list.append({'dataset':dataset, 'sub_dataset': sub_data, 'split': split, 'model': model, 'batch_size':4})
            else:
                h_param_list.append({'dataset':dataset, 'sub_dataset': sub_data, 'model': model, 'batch_size':4, 'split': 'test'})

    # run by submitit
    d = datetime.today()
    exp_dir = (
        Path("./dumps/")
        / "projects"
        / "feature-analysis"
        / "dumps"
        / f"{d.strftime('%Y-%m-%d')}_rand_eval"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    submitit_logdir = exp_dir / "submitit_logs"
    num_gpus = 1
    workers_per_gpu = 10
    executor = submitit.AutoExecutor(folder=submitit_logdir)
    executor.update_parameters(
        timeout_min=5,
        gpus_per_node=num_gpus,
        slurm_additional_parameters={"account": "rrg-bengioy-ad"},
        tasks_per_node=num_gpus,
        cpus_per_task=workers_per_gpu,
        slurm_mem="32G",#16G
        slurm_array_parallelism=100,
    )
    job = executor.map_array(HyperFeatures, [h_param_list[0]])
    print(str(len(h_param_list)) + ' Jobs submitted!')
