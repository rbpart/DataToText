#%%
from model.parser import HyperParameters
from model.build_model import build_model
from model.dataset import IDLDataset
from model.utils import bleu_score_
from model.metrics import metrics

if __name__=="__main__":
    hparameters = HyperParameters()
    hparameters.device = 'cpu'
    dataset = IDLDataset(hparameters, 'test')
    test_dataset = IDLDataset(hparameters, 'test')
    model = build_model(hparameters,dataset)
    model.load_params('model/models/experiment15/checkpoint_7500/model_7500.pt')
    model.to(hparameters.device)
    batch = dataset[[300,10]]
    outputs = model(batch)
    outputs = model.infer_to_sentence(batch,2,3,predictions=10,
                                    beam_width=10, best = True,
                                    block_n=4)
    score = bleu_score_(outputs,batch.target.raw)
    print(score)
    print(outputs)
#scores = metrics(outputs,batch.target.raw,batch.source.raw)

# %%
