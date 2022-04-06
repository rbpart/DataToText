#%%
from elasticsearch import Elasticsearch
import requests
import json

class Fetcher():
    es = Elasticsearch('esm1.icebergdatalab.com',verify_certs=True)
    index = 'db_client_entity_aggregated_production'
    fields = es.indices.get_mapping(index)[index]['mappings']

    def __init__(self) -> None:
        return

    def fetch_data(self,data):
        if data['analysis_id'] is not None:
            return self.es.search(index=self.index,query={
                "bool":{
                    "must":[
                        {"match":{"analysis.analysis_id":data['analysis_id']}},
                        {"match":{"year":str(data['year'])}}]
                }
            })['hits']['hits'][0]['_source']
        else:
            return self.es.search(index=self.index,query={
                "bool":{
                    "must":[
                        {"match":{"entity.name":data['name']}},
                        {"match":{"year":str(data['year'])}}]
                }
            })['hits']['hits'][0]['_source']

    def flatten(self,dic):
        res = {}
        if 'properties' in dic:
            res.update(self.flatten(dic['properties']))
        elif 'type' in dic:
            return {'type':dic['type']}
        else:
            for prop in dic:
                res.update({prop:self.flatten(dic[prop])})
        return res

    entityurl = 'http://api-entity.icebergdatalab.com/api/v1/entity?uid='
    def fetch_country(self,data):
        response = requests.get(self.entityurl+data['es_data']['entity']['uid'])
        if response.status_code == 200:
            tx= json.loads(response.text)['data'][0]
            if 'country' in tx:
                return tx['country']
        return None

# %%
