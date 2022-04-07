from itertools import tee
import math
import torch
import json
from model.parser import HyperParameters
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from typing import List, Union, no_type_check_decorator
import seaborn as sns
import matplotlib.pyplot as plt
hparams = HyperParameters()

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def nwise(iterable, n=2):
    iterables = tee(iterable, n)
    [next(iterables[i]) for i in range(n) for j in range(i)]
    return zip(*iterables)



# instead of concatenating tensors, we need to create a full size tensor and run inference at
# step i on the firsts i-th words : we avoid creating new tensors in each loop
def run_inference(model,preprocessed_srcs,dataset, hparms = hparams):
    _,batch_size,_ = preprocessed_srcs.shape
    targets = [['<bos>']*batch_size]
    targets = [dataset.tgt_vocab(t) for t in targets]
    tgt = torch.zeros((dataset.max_comment_len,batch_size),dtype=torch.long, device=hparms.device, requires_grad = False)
    tgt[0,:] = torch.tensor(targets, device=hparams.device, requires_grad = False)
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(1,dataset.max_comment_len)):
            out, attns = model(preprocessed_srcs,tgt[:i,:])
            tgt[i,:] = out.argmax(2).transpose(0,1)[i-1,:]
    model.train()
    return tgt

def inferred_to_sentence(tensor, dataset):
    output = tensor.transpose(0,1).tolist()
    comments = []
    for sentence in output:
        comments.append(' '.join(dataset.tgt_vocab.lookup_tokens(sentence)))
    return comments

def bleu_score_(comment_str: List[str], tokens: List[List[str]]):
    tok_candidates = [sent.split(' ') for sent in comment_str]
    tok_copy = json.loads(json.dumps(tokens))
    for i, cand in enumerate(tok_copy):
        tok_copy[i] += ['<blank>']*(len(cand)-len(tok_copy[i]))
        tok_copy[i] = [tok_copy[i]]
    return bleu_score(tok_candidates,tok_copy,max_n=1,weights=[1])


def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]

def plot_attention(batch,attns, batch_item = 0, ent = 0,types=['std','coverage']):
    sns.set()
    fig, axes = plt.subplots(1,len(types),figsize=(20,10))
    for i,type in enumerate(types):
        leng = batch.target.lens[batch_item]
        att = attns[type][:leng,batch_item,(ent)*hparams.ENT_SIZE:(ent+1)*hparams.ENT_SIZE].detach().numpy()
        x = [d for d in batch.source.raw[batch_item][ent].data if d != '<blank']
        y = [i for i in batch.target.raw[batch_item] if i !='<blank>']
        sns.heatmap(att, cmap='RdBu_r', ax=axes[i])
        axes[i].set_title(type)
        axes[i].set_yticks([i for i in range(len(y))],y, rotation=math.pi/2+0.1)
        axes[i].set_xticks([i+0.5 for i in range(len(x))],x,rotation = 45)
    return fig