#%%
import os
import json
from typing import List
import torch
import re
from collections import namedtuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from makedata.elastic import Fetcher
from model.parser import HyperParameters
from torchtext.vocab import build_vocab_from_iterator, Vocab
from joblib import Parallel, delayed
import pycountry
es = Fetcher()
fields = es.fields

DATAPATH = 'datasets/idl2/'
DELIM = "|"
ENT_SIZE= HyperParameters.ENT_SIZE

def train_test_split_save(*data, train_size=0.85, save=True):
    train, test, tar_train, tar_test = train_test_split(*data,train_size=train_size,shuffle=True, random_state=42)
    to_return = (train, test, tar_train, tar_test)
    if save:
        open(DATAPATH+'src-train.txt','w').write('\n'.join(train))
        open(DATAPATH+'src-test.txt','w').write('\n'.join(test))
        open(DATAPATH+'tgt-train.txt','w').write('\n'.join(tar_train))
        open(DATAPATH+'tgt-test.txt','w').write('\n'.join(tar_test))
    return to_return

def preprocess(dataset='idl', to_drop = []):
    error = 0
    with open(f'./datasets/{dataset}/comments.json', 'r') as json_file:
        file = json.load(json_file)
        preprocessed = Parallel(n_jobs=-1)(delayed(preprocess_one)(data) for data in tqdm(file))
    preprocessed = [t for t in preprocessed if t is not None]
    preprocessed = list(zip(*preprocessed))
    file, source, target = preprocessed[0], preprocessed[1], preprocessed[2]
    return(file,source, target)

def preprocess_one(data):
    try:
        data['es_data'] = es.fetch_data(data)
        code = es.fetch_country(data)
        data['country'] = pycountry.countries.get(alpha_2 = code).name.lower()
        raw = data
        source = build_source(data)
        target = re.sub('\n','',data['comment']).lower()
        return (raw,source,target)
    except Exception as e:
        return None

def build_source(data):
    return ' && '.join([build_ent(list_ent)
        for list_ent in build_macroplan(data)])

def build_ent(list_ent):
    res = ''
    for ent in list_ent:
        name = ent['NAME'] if 'NAME' in ent else ent['IMPACT_NAME']
        res += ' && '.join([f'{name} | {key} | {value}'  for key, value in ent.items() if value != name])
    return res

def build_macroplan(data):
    useless = ['comment','analysis_id','current_analyse_id','entity_impact_id','impact_type','es_data']
    infos = [x for x in data.keys() if x not in useless and data[x] is not None]
    indexed_data = data['es_data']['impacts']
    matchs = (build_entity(data, infos),
    build_impacts(data,indexed_data))
    return matchs

def build_entity(data,infos):
    return [{info.upper():data[info] for info in infos}]

def splitPascal(string:str):
    if string.isupper():
        return string.lower()
    new = ''
    for i,c in enumerate(string):
        if c.isupper() or c.isnumeric():
            new += ' '
        new += c
    return new.strip().lower()

def build_impacts(data,impacts,treshold = 0.8):
    """
    Input : data
    Outputs : List of POSITION of first letter of impat
    """
    results = []
    prefix = data['impact_type'].split('::')[1]
    if prefix == 'MSA':
        prefix = 'CBF'
    concerned_impacts = impacts[prefix]
    results += impact_percentage_index(flat_index(concerned_impacts,[prefix]))
    return results

def flat_index(data:dict,prefixes=[],root=True):
    new = {}
    for entry in data:
        if type(data[entry]) is dict:
            if 'total' in data[entry] and type(data[entry]['total']) is dict and 'sum' in data[entry]['total']:
                if entry == 'none' : new['.'.join(prefixes+['total'])] =  data[entry]['total']['sum']
                else : new['.'.join(prefixes+[entry])] =  data[entry]['total']['sum']
            new.update(flat_index(data[entry],prefixes+[entry],False))
    return new

scopes = ['total','scope1','scope2','scope3','scope3Upstream']
def impact_percentage_index(index:dict):
    """Should take a dict of ONE impact only like CBF or GreenShare, etc..."""
    res = []
    mx = max([abs(float(v)) if type(v) in [str,float] else 0 for k,v  in index.items()])
    if mx == 0:
        return []
    for keys,value in index.items():
        dec = keys.split('.')
        sc = [prefix for prefix in dec if prefix in scopes]
        if sc == []:
            continue
        elif len(dec) == 2:
            res += [{'TYPE':'impact','IMPACT_PCT':round(abs(float(value)/mx)*100),'IMPACT_TYPE':dec[0],
        'IMPACT_VALUE':round(float(value)), 'SCOPE':sc[0],
        'IMPACT_NAME':' '.join([splitPascal(s) for s in dec if s not in scopes])}]
        else:
            res += [{'TYPE':'impact','IMPACT_PCT':round(abs(float(value)/mx)*100), 'IMPACT_TYPE':dec[0],
        'IMPACT_SUBTYPE':splitPascal(dec[1]),
        'IMPACT_VALUE':round(float(value)), 'SCOPE':sc[0],
        'IMPACT_NAME':' '.join([splitPascal(s) for s in dec if s not in scopes])}]

    return res


if __name__ == '__main__':
    raw, data, target = preprocess()
    items = train_test_split_save(data,target,save=True)
    with open(DATAPATH+'comments_and_data.json','w') as file:
        json.dump(items[0],file)

# %%
