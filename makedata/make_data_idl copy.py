#%%
import os
import json
import torch
import re
from sklearn.model_selection import train_test_split
from spacy.tokens import doc
import en_core_web_lg
from tqdm import tqdm
from makedata.elastic import Fetcher
from main.parser import Opts
import warnings
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)
nlp = en_core_web_lg.load()
es = Fetcher()
fields = es.fields

device = torch.device('cuda')
print('Cuda available :', torch.cuda.is_available())
DATAPATH = 'datasets/idl/'
PREPROCESSED_DATAPATH = 'preprocessed/idl/'
DELIM = "|" #It's not the regular | symbol
ENT_SIZE= Opts.ENT_SIZE
def makedir():
    if not os.path.isdir('preprocessed/'):
        os.mkdir('preprocessed/')
    if not os.path.isdir(PREPROCESSED_DATAPATH):
        os.mkdir(PREPROCESSED_DATAPATH)

def train_test_valid_split(*data, train_size=0.7, valid_size=0.15, test_size=0.15, save=True):
    train, test_valid, tar_train, tar_train_valid = train_test_split(*data,train_size=train_size,shuffle=True, random_state=42)
    test, valid, tar_test, tar_valid = train_test_split(test_valid, tar_train_valid, train_size=test_size/(valid_size+test_size), random_state=42)
    to_return = (train, test, valid, tar_train, tar_test, tar_valid)
    if save:
        open(DATAPATH+'src-train.txt','w').write('\n'.join(train))
        open(DATAPATH+'src-test.txt','w').write('\n'.join(test))
        open(DATAPATH+'src-valid.txt','w').write('\n'.join(valid))
        open(DATAPATH+'tar-train.txt','w').write('\n'.join(tar_train))
        open(DATAPATH+'tar-test.txt','w').write('\n'.join(tar_test))
        open(DATAPATH+'tar-valid.txt','w').write('\n'.join(tar_valid))
    return to_return

def preprocess(dataset='idl', to_drop = []):
    error = 0
    with open(f'./datasets/{dataset}/comments.json', 'r') as json_file:
        file = json.load(json_file)
        raw, source, target = [], [], []
        for data in tqdm(file):
            try:
                data['es_data'] = es.fetch_data(data)
                data['country'] = es.fetch_country(data)
                raw.append(data)
                source.append(build_source(data))
                target.append(re.sub('\n','',data['comment']).lower())
            except Exception as e:
                error += 1
    print(error)
    return(file,source, target)

def build_source(data):
    tar = ''
    for list_ent in build_macroplan(data):
        tar += build_ent(list_ent)
    return tar

def build_ent(list_ent):
    res = ''
    for ent in list_ent:
        tar = DELIM.join(['<ent>']*2)
        for key, value in ent.items():
            tar += ' ' + DELIM.join([re.sub(' ','_',str(value)),key])
        for i in range(ENT_SIZE-len(ent)):
            tar += ' ' + DELIM.join(['<blank>']*2)
        res += tar + ' '
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
    makedir()
    raw, data, target = preprocess()
    items = train_test_valid_split(data,target,save=True)
    os.system("onmt_build_vocab -config preprocess.yml")



# %%
