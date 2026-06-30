import numpy as np
from hashlib import sha256
import importlib
import json

def sizeOfShard(container, shard):
    '''
    Returns the size (in number of points) of the shard before any unlearning request.
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    
    return shards[shard].shape[0]

def realSizeOfShard(container, label, shard):
    '''
    Returns the actual size of the shard (including unlearning requests).
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    requests = np.load('containers/{}/requestfile-{}.npy'.format(container, label), allow_pickle=True)
    
    return shards[shard].shape[0] - requests[shard].shape[0]

def getShardHash(container, label, shard, until=None):
    '''
    Returns a hash of the indices of the points in the shard lower than until
    that are not in the requests (separated by :).
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    requests = np.load('containers/{}/requestfile-{}.npy'.format(container, label), allow_pickle=True)

    if until == None:
        until = shards[shard].shape[0]
    indices = np.setdiff1d(shards[shard][:until], requests[shard])
    string_of_indices = ':'.join(indices.astype(str))
    return sha256(string_of_indices.encode()).hexdigest()

def fetchShardBatch(container, label, shard, batch_size, dataset, offset=0, until=None):
    '''
    Generator returning batches of points in the shard that are not in the requests
    with specified batch_size from the specified dataset
    optionnally located between offset and until (slicing).
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    requests = np.load('containers/{}/requestfile-{}.npy'.format(container, label), allow_pickle=True)
    
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))
    if until == None or until > shards[shard].shape[0]:
        until = shards[shard].shape[0]

    limit = offset
    while limit <= until - batch_size:
        limit += batch_size
        indices = np.setdiff1d(shards[shard][limit-batch_size:limit], requests[shard])
        yield dataloader.load(indices)
    if limit < until:
        indices = np.setdiff1d(shards[shard][limit:until], requests[shard])
        yield dataloader.load(indices)

def fetchTestBatch(dataset, batch_size):
    '''
    Generator returning batches of points from the specified test dataset
    with specified batch_size.
    '''
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

    limit = 0
    while limit <= datasetfile['nb_test'] - batch_size:
        limit += batch_size
        yield dataloader.load(np.arange(limit - batch_size, limit), category='test')
    if limit < datasetfile['nb_test']:
        yield dataloader.load(np.arange(limit, datasetfile['nb_test']), category='test')


#Funktioniert nicht
def fetchValBatch(container, label, shard, batch_size, dataset):
    """
    Generator returning batches of validation points for a shard.
    Lädt die Val-Daten aus dem dataloader und gibt sie in Batches zurück.
    """
    with open(dataset) as f:
        datasetfile = json.loads(f.read())

    # Lade den dataloader
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

    # Lade alle Val-Daten auf einmal (da Val-Daten nicht in Shards aufgeteilt sind)
    val_indices = np.arange(datasetfile.get('nb_val', 0))
    if len(val_indices) == 0:
        raise ValueError("Keine Val-Daten in datasetfile gefunden.")

    # Lade alle Val-Daten
    X_val, y_val = dataloader.load(val_indices, category='val')
    # Gib Batches zurück
    limit = 0
    while limit <= len(X_val) - batch_size:
        limit += batch_size
        yield X_val[limit-batch_size:limit], y_val[limit-batch_size:limit]
    if limit < len(X_val):
        yield X_val[limit:], y_val[limit:]
    
