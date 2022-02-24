#%%
import os
import json
import torch
import re
from sklearn.model_selection import train_test_split
from spacy.tokens import doc
import en_core_web_lg
from tqdm import tqdm
from data.elastic import Fetcher
import warnings
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)
nlp = en_core_web_lg.load()
es = Fetcher()
fields = es.fields

device = torch.device('cuda')
print('Cuda available :', torch.cuda.is_available())
DATAPATH = 'datasets/idl/'
PREPROCESSED_DATAPATH = 'preprocessed/idl/'
DELIM = "￨" #It's not the regular | symbol
ENT_SIZE= 6
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
                source.append(build_target(build_macroplan(data)))
                target.append(re.sub('\n','',data['comment']).lower())
            except Exception as e:
                error += 1
    print(error)
    return(file,source, target)

def build_target(macroplan):
    tar = ''
    for ent in macroplan:
        tar += build_ent(ent)
    return tar

def build_ent(ent):
    tar = DELIM.join(['<ent>']*2)
    for key, value in ent.items():
        tar += ' ' + DELIM.join([key,str(value)])
    for i in range(ENT_SIZE-len(ent)):
        tar += ' ' + DELIM.join(['<blank>']*2)
    return tar + ' '

def build_macroplan(data):
    useless = ['comment','analysis_id','current_analyse_id','entity_impact_id','impact_type','es_data','name']
    infos = [x for x in data.keys() if x not in useless and data[x] is not None]
    tokens = nlp(re.sub('\n','',data['comment']).lower())
    indexed_data = data['es_data']
    matchs = (infos_recognition(data,tokens,infos),
    country_recognition(data,tokens),
    impact_recognition(data,tokens,indexed_data['impacts']),
    entity_recognition(data,tokens))
    return order_token(matchs)

def order_token(args):
    tokens = []
    for token_list in args:
        tokens.extend(token_list)
        tokens.sort(key=lambda x : x['POSITION'])
    for i,tok in enumerate(tokens):
        tok.pop('POSITION')
        tok['REL_POSITION'] = i
    return tokens

def infos_recognition(data :dict, tokens:doc, infos, treshold=0.9):
    """
    Input : data
    Outputs : List of POSITION of first letter of entity name
    """
    results = []
    target = data['comment']
    for info in infos:
        tokenized_info = nlp(data[info])
        similars = [X for X in tokens.ents if X.similarity(tokenized_info) > treshold]
        similars.sort(key=lambda x: x.similarity(tokenized_info))
        if similars != []:
            results += [{'TYPE':info,info.upper():data[info].lower(),'TYPE':match.text,'POSITION':(match.start_char,match.end_char)} for match in similars]
        else:
            similars_tok = [X for X in tokens if X.similarity(tokenized_info) > treshold]
            results += [{'TYPE':info,info.upper():data[info].lower(),'NAME':match.text,'POSITION':(match.idx,match.idx+len(match))} for match in similars_tok]
            if similars_tok == []:
                results += [{'TYPE':info,info.upper():data[info].lower(),'NAME':tokenized_info.text,'POSITION':(match.span()[0],match.span()[1])} for match in re.finditer(tokenized_info.text.lower(),target.lower())]
    return results

def entity_recognition(data,tokens,treshold=0.9):
    results = []
    tokenized_info = nlp(data['name'])
    similars = [X for X in tokens.ents if X.similarity(tokenized_info) > treshold]
    similars.sort(key=lambda x: x.similarity(tokenized_info))
    for ent in tokens.ents:
        if ent.label_ == 'ORG':
            results += [{'TYPE':'company','NAME':ent.text,'MAIN_SECTOR':data['main_sector'],'SUB_SECTOR':data['sub_sector'],'POSITION':(ent.start_char,ent.end_char)}]
    return results

def country_recognition(data, tokens):
    results = []
    for ent in tokens.ents:
        if ent.label_ == 'GPE':
            results += [{'TYPE':'country','PLACE':ent.text,'POSITION':(ent.start_char,ent.end_char)}]
    return results

def splitPascal(string:str):
    new = ''
    for c in string:
        if c.isupper() or c.isnumeric():
            new += ' '
        new += c
    return new.strip().lower()

def impact_recognition(data,tokens,impacts,treshold = 0.8):
    """
    Input : data
    Outputs : List of POSITION of first letter of impat
    """
    results = []
    prefix = data['impact_type'].split('::')[1]
    if prefix == 'MSA':
        prefix = 'CBF'
    concerned_impacts = impacts[prefix]
    index = impact_percentage_index(reversed_index(concerned_impacts,[prefix]))
    results += extract_impact_indexed(index,data,tokens)
    # for impact in concerned_impacts:
    #     try:
    #         sub_impacts = [sb for sb in concerned_impacts[impact].keys() if sb in ['Occupational','ChangeOfLandUse','Ghg','Acidification','Eutrophication','scope3Upstream']]
    #     except:
    #         sub_impacts = []
    #     for imp in [impact]+sub_impacts:
    #         results += extract_impact_NER(imp,data,tokens,treshold)
    return results

def reversed_index(data:dict,prefixes=[],root=True):
    new = {}
    for entry in data:
        if type(data[entry]) is dict:
            if 'total' in data[entry] and type(data[entry]['total']) is dict and 'sum' in data[entry]['total']:
                if entry == 'none' : new[data[entry]['total']['sum']] =  '.'.join(prefixes+['total'])
                else : new[data[entry]['total']['sum']] =  '.'.join(prefixes+[entry])
            new.update(reversed_index(data[entry],prefixes+[entry],False))
    return new

def impact_percentage_index(reversed_index:dict):
    """Should take a dict of ONE impact only like CBF or GreenShare, etc..."""
    res = {}
    mx = max([abs(float(v)) if type(v) in [str,float] else 0 for v  in reversed_index.keys()])
    if mx == 0:
        return {}
    for k,v in reversed_index.items():
        res[k] = v
        res[abs(float(k)/mx)*100] = v
    return res

def extract_impact_NER(impact, data, tokens, treshold):
    impact = splitPascal(impact)
    tokenized_info = nlp(impact)
    orgs = [X for X in tokens if X.similarity(tokenized_info) > treshold]
    orgs.sort(key=lambda x: x.similarity(tokenized_info))
    return [{'TYPE':'impact','IMPACT_NAME':impact,'POSITION':(trigger.idx,trigger.idx+len(trigger))} for trigger in orgs]


def norm_dist(val1,val2):
    if float(val2) != 0:
        return abs(float(val1)-float(val2))/float(val2)
    return 0

def extract_impact_indexed(index,data,tokens,diff=0.01):
    results = []
    nums = [(tok,tok.lefts,tok.rights) for tok in tokens if tok.text.isnumeric()]
    for trigger in nums:
        for value in index:
            if norm_dist(value,trigger[0].text) < diff:
                if trigger[0].ent_type_ == 'PERCENT':
                    results += [{'TYPE':'impact','IMPACT_PCT':trigger[0].text,'POSITION':(trigger[0].idx,trigger[0].idx+len(trigger[0])), 'IMPACT_NAME':index[value], 'IMPACT_VALUE': [val for val in index if index[value] == index[val] and val != trigger[0].text][0]}]
                else:
                    results += [{'TYPE':'impact','IMPACT_VALUE':trigger[0].text,'POSITION':(trigger[0].idx,trigger[0].idx+len(trigger[0])), 'IMPACT_NAME':index[value], 'IMPACT_PCT': [val for val in index if index[value] == index[val] and val != trigger[0].text][0]}]
    return results

if __name__ == '__main__':
    makedir()
    raw, data, target = preprocess()
    items = train_test_valid_split(data,target,save=False)
    os.system("onmt_build_vocab -config preprocess.yml")


# %%
