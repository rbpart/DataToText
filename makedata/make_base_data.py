#%%
import json
from operator import pos
import pickle
import unicodedata
from typing import List
import torch
import re
from collections import namedtuple
from sklearn.model_selection import train_test_split
from tqdm import TqdmExperimentalWarning, tqdm
from makedata.elastic import Fetcher
from model.parser import HyperParameters
from torchtext.vocab import build_vocab_from_iterator, Vocab
from joblib import Parallel, delayed
import nlpaug.augmenter.word as naw
from nltk.tokenize import MWETokenizer, word_tokenize

es = Fetcher()
fields = es.fields

DATAPATH = 'datasets/idl/'
DELIM = "|"
ENT_SIZE= HyperParameters.ENT_SIZE

def train_test_split_save(*data, train_size=0.85, save=True, augment_size=1):
    train, test, tar_train, tar_test = train_test_split(*data,train_size=train_size,shuffle=True, random_state=42)
    to_return = (train, test, tar_train, tar_test)
    if save:
        open(DATAPATH+'src-train.txt','w').write('\n'.join(train))
        open(DATAPATH+'src-test.txt','w').write('\n'.join(test))
        open(DATAPATH+'tgt-train.txt','w').write('\n'.join(tar_train))
        open(DATAPATH+'tgt-test.txt','w').write('\n'.join(tar_test))
    return to_return

def preprocess(to_drop = [], saved = True):
    if saved:
        function = preprocess_one_saved
        path = './datasets/idl/comments_and_data.json'
    else:
        function = preprocess_one
        path = './datasets/idl/comments.json'
    with open(path, 'r') as json_file:
        file = json.load(json_file)
        preprocessed = Parallel(n_jobs=-1)(delayed(function)(data) for data in tqdm(file))
        # preprocessed = [preprocess_one(data) for data in tqdm(file)]
    preprocessed = [t for t in preprocessed if t is not None]
    print(len(preprocessed))
    preprocessed = list(zip(*preprocessed))
    file, source, target = preprocessed[0], preprocessed[1], preprocessed[2]
    return(file,source, target)

def preprocess_one_saved(data):
    try:
        target = re.sub('\n','',data['comment']).lower()
        if len(target.split(' ')) < 10:
            return None
        raw = data
        source = build_source(data)
        return (raw,source,target)
    except Exception as e:
        return None

def preprocess_one(data):
    try:
        data['es_data'] = es.fetch_data(data)
        target = re.sub('\n','',data['comment']).lower()
        if len(target.split(' ')) < 10:
            return None
        data['country'] = es.fetch_country(data)
        raw = data
        source = build_source(data)
        return (raw,source,target)
    except Exception as e:
        return None

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

scopes = {
'total':'total',
'scope1':'scope_1',
'scope2':'scope_2',
'scope3':'scope_3',
'scope3Upstream':'upstream'}
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
        'IMPACT_VALUE':round(float(value)), 'SCOPE':scopes[sc[0]],
        'IMPACT_NAME':' '.join([splitPascal(s) for s in dec if s not in scopes])}]
        else:
            res += [{'TYPE':'impact','IMPACT_PCT':round(abs(float(value)/mx)*100), 'IMPACT_TYPE':dec[0],
        'IMPACT_SUBTYPE':splitPascal(dec[1]),
        'IMPACT_VALUE':round(float(value)), 'SCOPE':scopes[sc[0]],
        'IMPACT_NAME':' '.join([splitPascal(s) for s in dec if s not in scopes])}]

    return res

Entity =  namedtuple('Entity',('data','tags'))

def _load_entity( entity:str, info_token = ' ', split_token = '|'):
    data, tags = [], []
    for info in entity.split(info_token):
        data.append(info.split(split_token)[0])
        tags.append(info.split(split_token)[1])
    return ['<ent>']+data,  ['<ent>']+tags

def _load_entities(line:str,entity_token = '<ent>|<ent>') -> List[Entity]:
    return [Entity(*_load_entity(entity.strip())) for entity in line.split(entity_token) if entity != '']

def vocab_src(data, tokenizer: MWETokenizer):
    samples = [_load_entities(line.strip()) for line in tqdm(data) if line != '']
    samples_data = [_process_src(entity.data) for sample in samples for entity in sample ]
    feed_tokenizer(tokenizer,samples_data)
    src_vocab : Vocab = build_vocab_from_iterator(samples_data)
    src_vocab_feat : Vocab = build_vocab_from_iterator([entity.tags for sample in samples for entity in sample ])
    return src_vocab, src_vocab_feat

def _process_src(entitydata: List[str]) -> List[str]:
    return [data.lower().strip().strip('_') for data in entitydata]

def rebuild(tokenizer: MWETokenizer, word):
    dico = tokenizer._mwes
    for bit in word.split('_'):
        dico = dico[bit]
    while dico != {True:None} and dico != {}:
        k = [k for k in dico.keys() if k != True][0]
        word += f'_{k}'
        dico = dico[k]
    return word

def _process_tgt(line:str, tokenizer: MWETokenizer):
    s = line.replace(r'\n','').lower()
    s = re.sub('([.,!?()])', r'\1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    s = remove_accents(s)
    tokenized = tokenizer.tokenize(word_tokenize(s))
    tokenized = [rebuild(tokenizer,word) if '_' in word else word for word in tokenized]
    return ['<bos>'] + tokenized + ['<eos>']

def vocab_tgt(data):
    # aug = naw.SynonymAug(aug_src='wordnet',aug_max=20)
    # possible_targets = [new for line in tqdm(data) for new in aug.augment(line,n=10)]
    # print(possible_targets)
    tgt_vocab : Vocab = build_vocab_from_iterator(data,specials=['<unk>','<blank>','<bos>','<eos>'])
    tgt_vocab.set_default_index(tgt_vocab['<unk>'])
    return tgt_vocab

def feed_tokenizer(tokenizer, samples_data, i=1):
    tokenizer.add_mwe(('<bos>',))
    tokenizer.add_mwe(('<eos>',))
    tokenizer.add_mwe(('<blank>',))
    tokenizer.add_mwe(('<unk>',))
    for entity in samples_data:
        for words in entity:
            splits = word_tokenize(words.replace('_',' '))
            for j in range(1,len(splits)+1):
                tokenizer.add_mwe(splits[:j])

def build_vocabs_and_target(src, tgt, tokenizer: MWETokenizer, save = True):
    srcv,srvf = vocab_src(src, tokenizer)
    tgt = reprocess_tgt(target, tokenizer)
    tgtv = vocab_tgt(tgt)
    if save:
        torch.save(srcv,DATAPATH+'/src_word.vocab.pt')
        torch.save(srvf,DATAPATH+'/src_feat.vocab.pt')
        torch.save(tgtv,DATAPATH+'/tgt_word.vocab.pt')
    return (srcv, srvf, tgtv, [' '.join(t) for t in tgt])

def reprocess_tgt(data, tokenizer):
    samples = [_process_tgt(line, tokenizer) for line in tqdm(data) if line != '']
    return samples

def remove_accents(string):
    return ''.join(c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn')

if __name__ == '__main__':
    raw, data, target = preprocess(saved=True)
    tokenize = MWETokenizer()
    srcv, srvf, tgtv, target = build_vocabs_and_target(data,target, tokenize, save = True)
    pickle.dump(tokenize,open(DATAPATH+'/tokenizer.pickle','wb'))
    items = train_test_split_save(data,target,save=True)
    with open(DATAPATH+'comments_and_data.json','w') as file:
        json.dump(raw,file)




# %%
