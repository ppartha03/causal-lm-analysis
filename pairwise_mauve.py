from mauve import compute_mauve, discrete_mauve

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from itertools import product
from pathlib import Path
from datetime import datetime


import submitit
import os
import pandas as pd

import spacy
import json
nlp = spacy.load("en_core_web_sm")


def get_nouns_verbs(sentences):
  nouns = {}
  verbs = {}
  for s in sentences:
    toks = nlp(s)
    for np in toks.noun_chunks:
      if str(np).lower() in nouns:
        nouns[str(np).lower()]+=1
      else:
        nouns[str(np).lower()]=1

    for t in toks:
      if t.pos_ in ['VERB','AUX']:
          if str(t) in verbs:
            verbs[str(t).lower()]+=1
          else:
            verbs[str(t).lower()]=1

  return nouns, verbs

def plot_histograms(config, out):
    idxs = np.argsort(out.p_hist)[::-1]
    sample_p = np.random.multinomial(n=1000, pvals=out.p_hist[idxs])
    sample_q = np.random.multinomial(n=1000, pvals=out.q_hist[idxs])

    x = np.arange(out.p_hist.shape[0])
    plt.bar(x, sample_p, color='blue', alpha=0.3, label='P')
    plt.bar(x, sample_q, color='red', alpha=0.3, label='Q')
    plt.legend()
    plt.title(config['dataset1']+'_'+config['sub_dataset1']+'_' + config['split'] + '_and_'+config['dataset2']+'_'+config['sub_dataset2'])

    plt.savefig(os.path.join('Mauve', config['model'], config['dataset1']+'_'+config['sub_dataset1']+ '_' + config['split'] + '_and_'+config['dataset2']+'_'+config['sub_dataset2'], 'clusters_histograms.png'))
    plt.close()

    return None

def plot_and_save(config, out, arg = 'noun'):
    plt.plot(out.divergence_curve[:, 0], out.divergence_curve[:, 1])
    plt.title(config['dataset1'] + '_' + config['sub_dataset1'] + '_' + config['split'] + '_and_' + config['dataset2'] + '_' + config['sub_dataset2'])

    plt.savefig(os.path.join('Mauve', config['model'], config['dataset1'] + '_' + config['sub_dataset1'] + '_' + config['split'] + '_and_' + config['dataset2'] + '_' + config['sub_dataset2'], arg + '_divergence.png'))
    plt.close()

    return None


def HyperMauve(config):
    target_dir = os.path.join('Mauve', config['model'], config['dataset1'] + '_' + config['sub_dataset1'] + '_' + config['split'] + '_and_' + config['dataset2'] + '_' + config['sub_dataset2'])

    p_features_s = np.loadtxt(os.path.join('Features', config['model'], config['dataset1'], config['sub_dataset1'], config['split'], 'features.npy'), dtype=float)
    q_features_s = np.loadtxt(os.path.join('Features', config['model'], config['dataset2'], config['sub_dataset2'], 'test', 'features.npy'), dtype=float)

    p_text = load_dataset("universal_dependencies", config['sub_dataset1'])[config['split']]['text']
    q_text = load_dataset("universal_dependencies", config['sub_dataset2'])[config['split']]['text']
    #blimp_df = pd.read_csv('./blimp/blimp.csv')

    # q_text = blimp_df[blimp_df.linguistics_term == config['sub_dataset2']]["sentence_bad"].tolist() + blimp_df[blimp_df.linguistics_term == config['sub_dataset2']]["sentence_good"].tolist()
    # p_text = blimp_df[blimp_df.linguistics_term == config['sub_dataset1']]["sentence_bad"].tolist() + blimp_df[blimp_df.linguistics_term == config['sub_dataset1']]["sentence_good"].tolist()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    p_nouns, p_verbs = get_nouns_verbs(p_text)
    q_nouns, q_verbs = get_nouns_verbs(q_text)

    out_nouns = discrete_mauve.discrete_mauve(p_nouns, q_nouns)
    out_verbs = discrete_mauve.discrete_mauve(p_verbs, q_verbs)

    out = compute_mauve.compute_mauve(p_features=p_features_s, q_features=q_features_s,
                    device_id=0, max_text_length=256)

    plot_and_save(config, out_nouns, arg = 'noun')
    plot_and_save(config, out_verbs, arg = 'verb')
    plot_and_save(config, out, arg = 'feature')

    plot_histograms(config, out)

    # write cluster samples and metrics

    cluster_file = os.path.join(target_dir, 'cluster_details.json')
    mauve_scores_file = os.path.join(target_dir, 'mauve_scores.json')
    clusters = {}

    refs = {'p': config['dataset1'] + '_' + config['sub_dataset1'] + '_' + config['split'], 'q': config['dataset2'] + '_' +config['sub_dataset2']}
    for cluster_id, samples in out.cluster_stats.items():
        clusters[cluster_id] = {refs[k]: points for k, points in samples.items()}

    json.dump(clusters, open(cluster_file, 'w'))

    mauve_scores = {'mauve_features': out.mauve, 'mauve_nouns': out_nouns.mauve, 'mauve_verbs': out_verbs.mauve}

    json.dump(mauve_scores, open(mauve_scores_file, 'w'))

    return None


if __name__ == '__main__':

    sub_datasets = {
    'blimp': ['island_effects', 'anaphor_agreement', 's-selection', 'argument_structure', 'determiner_noun_agreement', 'subject_verb_agreement', 'ellipsis',
 'control_raising', 'quantifiers', 'irregular_forms', 'npi_licensing', 'binding', 'filler_gap_dependency'],
 'universal_dependencies': ['en_pronouns', 'en_esl', 'en_ewt', 'en_gum', 'en_gumreddit', 'en_lines', 'en_partut', 'en_pud']
 }
    splits = {
    'en_pronouns': ['test'],
    'en_esl': ['train', 'validation', 'test'],
    'en_ewt': ['train', 'validation', 'test'],
    'en_gum': ['train', 'validation', 'test'],
    'en_gumreddit': ['train', 'validation', 'test'],
    'en_lines': ['train', 'validation', 'test'],
    'en_partut': ['train', 'validation', 'test'],
    'en_pud': ['test']
    }

    PARAM_GRID = list(product(
    ['gpt2'],
    ['universal_dependencies'],
    ['universal_dependencies'], #['blimp']
    ['en_ewt', 'en_gum', 'en_lines', 'en_partut'],#['island_effects', 'anaphor_agreement', 's-selection', 'argument_structure', 'determiner_noun_agreement', 'subject_verb_agreement', 'ellipsis',
 #'control_raising', 'quantifiers', 'irregular_forms', 'npi_licensing', 'binding', 'filler_gap_dependency'],#['en_pronouns', 'en_esl', 'en_ewt', 'en_gum', 'en_gumreddit', 'en_lines', 'en_partut', 'en_pud'],
    ['en_ewt', 'en_gum', 'en_lines', 'en_partut'],#['island_effects', 'anaphor_agreement', 's-selection', 'argument_structure', 'determiner_noun_agreement', 'subject_verb_agreement', 'ellipsis',
 #'control_raising', 'quantifiers', 'irregular_forms', 'npi_licensing', 'binding', 'filler_gap_dependency']
    )
    )

    h_param_list = []

    for param_ix in range(len(PARAM_GRID)):

        params = PARAM_GRID[param_ix]

        model, dataset1, dataset2, sub_data1, sub_data2 = params

        #for split in splits[sub_data1]:
        h_param_list.append({'dataset1': dataset1, 'sub_dataset1': sub_data1, 'dataset2': dataset2, 'sub_dataset2': sub_data2, 'split': 'train', 'model': model})


    # run by submitit
    d = datetime.today()
    exp_dir = (
        Path("./dumps/")
        / "projects"
        / "causal-analysis"
        / "dumps"
        / f"{d.strftime('%Y-%m-%d')}_rand_eval"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    submitit_logdir = exp_dir / "submitit_logs"
    num_gpus = 1
    workers_per_gpu = 5
    executor = submitit.AutoExecutor(folder=submitit_logdir)
    executor.update_parameters(
        timeout_min=60,
        #gpus_per_node=num_gpus,
        slurm_additional_parameters={"account": "rrg-bengioy-ad"},
        #tasks_per_node=num_gpus,
        cpus_per_task=workers_per_gpu,
        slurm_mem="32G",
        slurm_array_parallelism=100,
    )
    job = executor.map_array(HyperMauve, h_param_list)
    print(str(len(h_param_list)) + ' Jobs submitted!')
