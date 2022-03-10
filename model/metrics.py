#%%
import re
from spacy.tokens import doc
import en_core_web_lg
from model.utils import damerau_levenshtein_distance
import warnings
from typing import List
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)
nlp = en_core_web_lg.load()
DATAPATH = 'datasets/idl/'
PREPROCESSED_DATAPATH = 'preprocessed/idl/'
DELIM = "|" #It's not the regular | symbol
ENT_SIZE= 6
#%%
def to_dict(ent):
    dic = {}
    for data,tag in zip(ent.data,ent.tags):
        dic[tag] = data
    return dic

def entities_probably_equal(ent1,ent2):
    return len(set(ent1.data)-set(ent2.data)) > 0

def entity_probably_in(ent1,list_ents):
    for ent in list_ents:
        if entities_probably_equal(ent,ent1):
            return True
    return False

def reprocess(src):
    processed = []
    for ent in src:
        processed.append(to_dict(ent))
    return processed

def metrics(batch_inferred, batch_golden, batch_src):
    for inferred, golden, src in zip(batch_inferred,batch_golden,batch_src):
        metric = _metrics(inferred,golden,src)
    return metric

def _metrics(inferred_str,golden_tokens,src):
    # TODO: suppress when all nice and tidy
    src = reprocess(src)
    golden_str = ' '.join(golden_tokens).replace(' <blank>','')
    entities_golden = extract_entities(golden_str,src)
    entities_inferred = extract_entities(inferred_str,src)

    RGPPerc, RGPNum = 0, 0
    TP,FP, FN, TN = 0,0,0,0
    ContentOrdering = 0
    for entity,key,value,position in entities_inferred:
        if entity_probably_in(entity,src):
            RGPNum += 1
        if entity_probably_in(entity,entities_golden):
            TP += 1
        else:
            FP += 1
    for entity,key,value,position in entities_golden:
        if not entity_probably_in(entity,entities_inferred):
            FN += 1
    TN = len(src) - FN - TP - FP
    if len(entities_inferred) != 0:
        RGPPerc = RGPNum/len(entities_inferred)
        CSP_Precision = TP/(TP+FP)
        CSP_Recall = TP/(TP+FN)
    else :
        CSP_Precision, CSP_Recall, RGPPerc = 0, 0, 0
    ContentOrdering = damerau_levenshtein_distance(' '.join(entities_golden), ' '.join(entities_inferred))
    return RGPNum, RGPPerc, CSP_Precision, CSP_Recall, ContentOrdering

def order_ent(args):
    ents = []
    for token_list in args:
        ents.extend(token_list)
    ents.sort(key=lambda x : x[3].span()[0])
    return ents

def extract_entities(string, src):
    infos = [ent for ent in src if ent['TYPE'] != 'impact']
    impacts = [ent for ent in src if ent['TYPE'] == 'impact']
    print(string)
    matchs = (
        infos_recognition(string,src),)
        #impact_recognition(tokens,src))
    matchs = order_ent(matchs)
    print(matchs)
    return matchs

def infos_recognition(string: str, infos: List[dict], treshold=0.9):
    """
    Input : data
    Outputs : List of POSITION of first letter of entity name
    """
    results = []
    for entity in infos:
        in_tok = [(entity,key,value,match) for key, value in entity.items() for match in re.finditer(' ('+value.lower()+')\b',string.lower())]
        in_tok = [match for match in in_tok if match[3] is not None]
        results += in_tok
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

# %%
