import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.utils.data as tud
from model.dataset import Batch, Source, Target


def greedy_search(model,batch: Batch,
                bos_idx,
                predictions = 20,
                progress_bar = True):
    with torch.no_grad():
        _,batch_size,_ = batch.source.tensor.shape
        beggining = [[bos_idx]*batch_size]
        vocabulary_size = model.decoder.embeddings.num_embeddings
        tgt = torch.zeros((predictions,batch_size), dtype=torch.long, device=model.device, requires_grad = False)
        tgt[0,:] = torch.tensor(beggining, device=model.device, requires_grad = False)
        predictions_iterator = range(predictions)
        if progress_bar:
            predictions_iterator = tqdm(predictions_iterator)
        for i in predictions_iterator:
                batch.target.tensor = torch.where(tgt[:i,:] < vocabulary_size,tgt[:i,:],0)
                out = model.forward(batch)
                tgt[i,:] = out.argmax(2).transpose(0,1)[i-1,:] #maybe problem with this line cause repetition
    return tgt.transpose(0,1)

def beam_search(model,
                batch: Batch,
                bos_idx,
                predictions = 20,
                beam_width = 5,
                batch_size = 50,
                progress_bar = True,
                best = True,
                anti_begayement = True):
    """
    Implements Beam Search to compute the output with the sequences given in X. The method can compute
    several outputs in parallel with the first dimension of X.
    Parameters
    ----------
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.
    predictions: int
        The number of tokens to append to X.
    beam_width: int
        The number of candidates to keep in the search.
    batch_size: int
        The batch size of the inner loop of the method, which relies on the beam width.
    progress_bar: bool
        Shows a tqdm progress bar, useful for tracking progress with large tensors.
    Returns
    -------
    Y: LongTensor of shape (examples, length + predictions)
        The output sequences.
    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the
        probability of the next token at every step.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        Y = torch.ones(1, batch.source.tensor.shape[1], device=device, dtype=torch.long)*bos_idx
        batch.target.tensor = Y
        # The next command can be a memory bottleneck, can be controlled with the batch
        # size of the predict method.
        next_probabilities = model.forward(batch)[:, -1, :]
        vocabulary_size = model.decoder.embeddings.num_embeddings
        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1)\
        .topk(k = beam_width, axis = -1)
        Y = Y.repeat((beam_width, 1)).view(1,-1)
        next_chars = next_chars.view(1, -1)
        # les mots sont tous sur la même dimension source après source [source1(10 mots), source2(10 mots)...]
        Y = torch.cat((Y, next_chars), axis = 0)
        batch.source.tensor = batch.source.tensor.repeat_interleave(beam_width,dim=1).transpose(0,1)
        batch.source.lens = batch.source.lens.repeat_interleave(beam_width)
        batch.source.map = batch.source.map.to_dense().repeat_interleave(beam_width,dim=1).transpose(0,1).to_sparse()
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        if progress_bar:
            predictions_iterator = tqdm(predictions_iterator)
        for i in predictions_iterator:
            # on répète les sources l'une après l'autre
            Y = Y.transpose(0,1)
            dataset = tud.TensorDataset(
                batch.source.tensor,
                batch.source.map,
                batch.source.lens,
                Y)
            loader = tud.DataLoader(dataset, batch_size = batch_size)
            next_probabilities = []
            iterator = iter(loader)
            if progress_bar > 1:
                iterator = tqdm(iterator)
            for srcI,src_mapI,src_lenI, y in iterator:
                srcI = srcI.transpose(0,1)
                src_mapI = src_mapI.transpose(0,1)
                y = y.transpose(0,1)
                batch_ = Batch(Source(srcI,src_lenI,src_mapI,batch.source.raw),
                            batch.vocab,
                            Target(torch.where(y>=vocabulary_size,0,y),None,None))
                next_probabilities.append(
                    model.forward(batch_)[:, -1, :].log_softmax(-1))
            next_probabilities = torch.cat(next_probabilities, axis = 0)
            next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
            if anti_begayement:
                next_probabilities[:,:,Y[:,i-1]] = 1e-20
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim = 1)
            probabilities, idx = probabilities.topk(k = beam_width, axis = -1)
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            Y = Y.transpose(0,1)
            best_candidates += torch.arange(Y.shape[1] // beam_width, device = device).unsqueeze(-1) * beam_width
            Y[i,:] = Y[i,best_candidates.flatten()-1]
            Y = torch.cat((Y, next_chars.view(1,-1)), axis = 0)
        if best:
            return Y.view(-1, beam_width, predictions+1)[:,0,:]
        return Y.view(-1, beam_width, predictions+1), probabilities