# Author: Krishna Pillutla
# License: GPLv3

import math
import numpy as np
import time
from types import SimpleNamespace

import faiss
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import auc as compute_area_under_curve

try:
    import torch
    FOUND_TORCH = True
except (ImportError, ModuleNotFoundError):
    FOUND_TORCH = False

try:
    import transformers
    FOUND_TRANSFORMERS = True
except (ImportError, ModuleNotFoundError):
    FOUND_TRANSFORMERS = False

if FOUND_TORCH and FOUND_TRANSFORMERS:
    # only needed for tokenizing
    from .utils import get_tokenizer, get_model, featurize_tokens_from_model, get_device_from_arg


MODEL, TOKENIZER, MODEL_NAME = None, None, None

def get_features_from_input(features, tokenized_texts, texts,
                            featurize_model_name, max_len, device_id, name,
                            verbose=False, batch_size=16):
    global MODEL, TOKENIZER, MODEL_NAME
    if features is None:
        # Featurizing is necessary. Make sure the required packages are available
        if not FOUND_TORCH:
            raise ModuleNotFoundError(
                """PyTorch not found. Please install PyTorch if you would like to use the featurization.
                    For details, see `https://github.com/krishnap25/mauve`
                    and `https://pytorch.org/get-started/locally/`.
                """)
        if not FOUND_TRANSFORMERS:
            raise ModuleNotFoundError(
                """Transformers not found. Please install Transformers if you would like to use the featurization.
                    For details, see `https://github.com/krishnap25/mauve`
                    and `https://huggingface.co/transformers/installation.html`.
                """)

        if tokenized_texts is None:
            # tokenize texts
            if TOKENIZER is None or MODEL_NAME != featurize_model_name:
                if verbose: print('Loading tokenizer')
                TOKENIZER = get_tokenizer(featurize_model_name)
            if verbose: print('Tokenizing text...')
            TOKENIZER.pad_token = TOKENIZER.eos_token
            tokenized_texts = [
                TOKENIZER.batch_encode_plus(texts[b_s: b_s + batch_size], return_tensors='pt', truncation=True,  padding=True, max_length=max_len)['input_ids']
                for b_s in range(0,len(text), batch_size)
            ]
        # use tokenized_texts to featurize
        if TOKENIZER is None or MODEL_NAME != featurize_model_name:
            if verbose: print('Loading tokenizer')
            TOKENIZER = get_tokenizer(featurize_model_name)
        if MODEL is None or MODEL_NAME != featurize_model_name:
            if verbose: print('Loading model')
            MODEL = get_model(featurize_model_name, TOKENIZER, device_id)
            MODEL_NAME = featurize_model_name
        else:
            MODEL = MODEL.to(get_device_from_arg(device_id))
        if verbose: print('Featurizing tokens')
        features = featurize_tokens_from_model(MODEL, tokenized_texts, name).detach().cpu().numpy()
    else:
        features = np.asarray(features)
    return features

def cluster_feats(p, q, num_clusters,
                  norm='none', whiten=True,
                  pca_max_data=-1,
                  explained_variance=0.9,
                  num_redo=5, max_iter=500,
                  seed=0, verbose=False):
    assert 0 < explained_variance < 1
    if verbose:
        print(f'seed = {seed}')
    assert norm in ['none', 'l2', 'l1', None]
    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    pca = PCA(n_components=None, whiten=whiten, random_state=seed+1)
    if pca_max_data < 0 or pca_max_data >= data1.shape[0]:
        pca.fit(data1)
    elif 0 < pca_max_data < data1.shape[0]:
        rng = np.random.RandomState(seed+5)
        idxs = rng.choice(data1.shape[0], size=pca_max_data, replace=False)
        pca.fit(data1[idxs])
    else:
        raise ValueError(f'Invalid argument pca_max_data={pca_max_data} with {data1.shape[0]} datapoints')
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    if verbose:
        print(f'performing clustering in lower dimension = {idx}')
    data1 = pca.transform(data1)[:, :idx+1]
    # Cluster
    data1 = data1.astype(np.float32)
    t1 = time.time()
    kmeans = faiss.Kmeans(data1.shape[1], num_clusters, niter=max_iter,
                          verbose=verbose, nredo=num_redo, update_index=True,
                          seed=seed+2)
    kmeans.train(data1)
    _, labels = kmeans.index.search(data1, 1)
    labels = labels.reshape(-1)
    t2 = time.time()
    if verbose:
        print('kmeans time:', round(t2-t1, 2), 's')

    q_labels = labels[:len(q)]
    p_labels = labels[len(q):]

    clusters = {k:{'p':[],'q':[]} for k in range(n_clusters)}
    for indx in range(len(p_labels)):
      clusters[p_labels[indx][0]]['p'].append(indx)

    for indx in range(len(q_labels)):
      clusters[q_labels[indx][0]]['q'].append(indx)

    q_bins = np.histogram(q_labels, bins=num_clusters,
                           range=[0, num_clusters], density=True)[0]
    p_bins = np.histogram(p_labels, bins=num_clusters,
                          range=[0, num_clusters], density=True)[0]
    return p_bins / p_bins.sum(), q_bins / q_bins.sum(), cluster_stats


def kl_multinomial(p, q):
    assert p.shape == q.shape
    if np.logical_and(p != 0, q == 0).any():
        return np.inf
    else:
        idxs = np.logical_and(p != 0, q != 0)
        return np.sum(p[idxs] * np.log(p[idxs] / q[idxs]))


def get_divergence_curve_for_multinomials(p, q, mixture_weights, scaling_factor):
    # TODO: check if extreme points are needed
    divergence_curve = [[0, np.inf]] # extreme point
    for w in np.sort(mixture_weights):
        r = w * p + (1 - w) * q
        divergence_curve.append([kl_multinomial(q, r), kl_multinomial(p, r)])
    divergence_curve.append([np.inf, 0]) # other extreme point
    return np.exp(-scaling_factor * np.asarray(divergence_curve))

def get_fronter_integral(p, q, scaling_factor=2):
    total = 0.0
    for p1, q1 in zip(p, q):
        if p1 == 0 and q1 == 0:
            pass
        elif p1 == 0:
            total += q1 / 4
        elif q1 == 0:
            total += p1 / 4
        elif abs(p1 - q1) > 1e-8:
            t1 = p1 + q1
            t2 = p1 * q1 * (math.log(p1) - math.log(q1)) / (p1 - q1)
            total += 0.25 * t1 - 0.5 * t2
        # else: contribution is 0
    return total * scaling_factor

def discrete_mauve(count_dict_p, count_dict_q, epsilon = 1e-4, divergence_curve_discretization_size=25):
  vocab_list = list(set(count_dict_p.keys()).union(set(count_dict_p.keys())))
  p_sum = sum([v for _,v in count_dict_p.items()])
  q_sum = sum([v for _,v in count_dict_q.items()])
  mixture_weights = np.linspace(1e-6, 1-1e-6, divergence_curve_discretization_size)
  mauve_scaling_factor=5

  q_list = []
  p_list = []

  for i in range(len(vocab_list)):
    if vocab_list[i] in count_dict_q:
      q_list.append(count_dict_q[vocab_list[i]]/float(q_sum))
    else:
      q_list.append(epsilon/float(q_sum))
    if vocab_list[i] in count_dict_p:
      p_list.append(count_dict_p[vocab_list[i]]/float(p_sum))
    else:
      p_list.append(epsilon/float(p_sum))

  q = np.array(q_list)
  p = np.array(p_list)

  del q_list
  del p_list

  divergence_curve = get_divergence_curve_for_multinomials(p, q, mixture_weights, mauve_scaling_factor)
  x, y = divergence_curve.T
  idxs1 = np.argsort(x)
  idxs2 = np.argsort(y)
  mauve_score = 0.5 * (
      compute_area_under_curve(x[idxs1], y[idxs1]) +
      compute_area_under_curve(y[idxs2], x[idxs2])
  )
  fi_score = get_fronter_integral(p, q)

  to_return = SimpleNamespace(divergence_curve=divergence_curve,
      mauve=mauve_score,
      frontier_integral=fi_score
  )

  return to_return
