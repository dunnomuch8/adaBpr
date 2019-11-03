import numpy as np
from functools import reduce
from joblib import Parallel, delayed
from IPython import embed
NUM_THREADS=8
BATCH_SIZE = 5000
# def slices(slicer):
    # score = slices((slice(i,j),test_users))(self.m_U)
    # return reduce(lambda a,b:lambda z:a(z[b]),slicer,lambda x:x)

class model(object):
    def __init__(self, m_U=None, m_V=None, bu=np.array([]), bv=np.array([]), Rate=None):
        if Rate is None:
            assert m_V.shape[1] == m_U.shape[1]
            self.n_users = m_U.shape[0]
        else:
            self.n_users = Rate.shape[0]
        assert bu.size==0 or len(m_U) == len(bu)
        assert bv.size==0 or len(m_V) == len(bv)
        self.m_U = m_U
        self.m_V = m_V
        self.bv = bv
        self.bu = bu
        self.Rate = Rate #b4 train filter
        self.alpha = 1
    def get_rate(self, test_users, availableItems):
        score = self.m_U[test_users].dot(self.m_V[availableItems].T)
        score += self.bu.size and self.bu[test_users].reshape(-1, 1)
        score += self.bv.size and self.bv[availableItems].reshape(1, -1)
        # if write_rate is not None:
        #     write_rate += score
        #     return write_rate
        # if trainR is not None:
            # gt = (trainR > 0 * 1).toarray()
            # score[gt] = -np.inf
        # score *= self.alpha
        return score

def get_top(modelList, i,j, trainR, write_rate,test_users=None,  availableItems=slice(None)):
    test_users=test_users[i:j] if test_users else slice(i,j)
    rate = []
    for m in modelList:
        if m.Rate is None:
            score = m.get_rate(test_users,  availableItems)*m.alpha
        else:
            score = m.Rate[test_users][:,availableItems]*m.alpha

        if write_rate is not None:
            write_rate[i:j] += score

        if trainR is not None:
            gt = (trainR[i:j] > 0).toarray()
            score[gt] = -np.inf
        rate.append(score)
        # rate[-1]=0.7*rate[-1]-0.3*np.log2(trainR.getnnz(0)/trainR.shape[0])
    rate = sum(rate)
    return np.argsort(rate, 1)[:, ::-1]


def get_precision(modelList, i, j, testR, trainR=None, write_rate=None):
    top = get_top(modelList, i, j, trainR, write_rate)
    allrank = np.argsort(top, 1) + 1
    testR = testR[i:j]
    data = np.zeros(testR.nnz, dtype=np.float32)
    for e, (ranku, Ru) in enumerate(zip(allrank, testR)):
        pos = Ru.indices  # np.where(Ru.toarray()[0])[0]
        rankui = ranku[pos]
        data[testR.indptr[e]:testR.indptr[e + 1]
             ] = (np.argsort(np.argsort(rankui)) + 1) / rankui

    return data  # ,row,col


def get_hits(modelList, i, j, M, testR, trainR, inids, test_users, availableItems):
    # embed()
    top = get_top(modelList, i, j, trainR, None,test_users,  availableItems)
    gt = testR[i:j].toarray()
    total_hit = np.zeros((1 if isinstance(inids, int)else 3, j - i, M))
    for i in range(total_hit.shape[1]):
        total_hit[0, i] = gt[i, top[i, :M]]  #top= top200idx
        if len(total_hit)>1:
            # choose top M in 2 sets repectively
            cond = np.isin(top[i, :], inids)
            part = gt[i, top[i, cond][:M]]
            total_hit[1, i, :len(part)] = part
            part = gt[i, top[i, ~cond][:M]]
            total_hit[2, i, :len(part)] = part
            # cond = np.isin(top, inids)#true/false top200idx
            # total_hit[3, i] = total_hit[0, i]*cond#1,0,1 200
            # total_hit[4, i] = total_hit[0, i]*(1-cond)#1,0,1 200
    return total_hit, top[:,:M]


def get_novelty(IM, trainR, M):
    '''IM:I[:, :M] from get_hits()'''
    popsum = np.log2(trainR.getnnz(0)[IM].cumsum(-1)[:,M])
    novmaxR = np.log2(trainR.shape[0])*(np.array(M)+1)
    novelty = 1-popsum/novmaxR
    return novelty

def get_subRecall(hits, IM, trainR, testR):
    '''IM:I[:, :M] from get_hits()'''
    idx = np.argsort(trainR.getnnz(0))[::-1]
    group = np.array_split(idx, 8)
    bar = np.zeros((len(group),)+IM.shape) #g,u,m
    for i in range(IM.shape[0]):
        for e, inids in enumerate(group):
            cond = np.isin(IM[i], inids)  # true/false top200idx
            bar[e, i] = hits[i]*cond  # 1,0,1 200
    for i in range(len(group)):
        bar[i] = np.cumsum(bar[i], 1)/testR[:, group[i]].getnnz(1).reshape(-1, 1)
    return bar

def get_recall(modelList, testR, trainR,  M, test_users=None, availableItems=slice(None), inids=False):
    assert trainR is  None or testR.shape == trainR.shape
    assert test_users is None or len(test_users) == testR.shape[0]
    assert availableItems == slice(None) or len(availableItems) == testR.shape[1]
    if inids:
        inids = modelList[0].inids[availableItems]
        cond = ((testR.getnnz(0) > 0)==inids)
        if cond.all():
            inids=1 
        elif not cond.any():
            inids=2
        else:inids=np.where(inids)[0]
    else:
        inids=0

    m_num_users = testR.shape[0]
    n = m_num_users // BATCH_SIZE + 1
    # get_hits(modelList, 0 * BATCH_SIZE, min((0 + 1) * BATCH_SIZE, m_num_users), M[-1] + 1, testR, trainR, inids, test_users, availableItems)
    res = Parallel(n_jobs=NUM_THREADS, verbose=0, backend="threading")(delayed(get_hits)(modelList, i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, m_num_users), M[-1] + 1, testR, trainR, inids, test_users, availableItems) for i in range(n))
    
    # To compute novelty or partial set recall, return top[:,:M]from get_hits()
    # IM = np.concatenate(list(zip(*res))[1])
    # novelty=np.nanmean(get_novelty(IM, trainR, M),0)
    # res = np.concatenate(list(zip(*res))[0],1)  #n 3 u m
    # subRecall = np.nanmean(get_subRecall(res[0], IM, trainR, testR)[:,:, M],1)
    # hits = np.cumsum(res,-1)  #3um
    hits = np.cumsum(np.concatenate(res,1),-1)  #3um

    recall = np.zeros((3, len(M))) 
    hits_eachuser = testR.getnnz(1).reshape(-1, 1)
    recall[0] = np.nanmean(hits[0][:, M]/hits_eachuser,0)
    if len(hits)>1:
        inset = testR[:, inids].getnnz(1).reshape(-1, 1)             
        recall[1] = np.nanmean(hits[1][:, M]/inset,0)
        recall[2] = np.nanmean(hits[2][:, M]/(hits_eachuser-inset),0)
    else:
        if inids == 2:
            recall[1] = totalrecall
        elif inids == 1:
            recall[0] = totalrecall
        else:
            recall = recall[:1]
    return recall


def get_accuracy(modelList, testR, trainR, write_rate=None):
    if trainR:
        assert testR.shape == trainR.shape
    m_num_users = testR.shape[0]
    n = m_num_users // BATCH_SIZE + 1
    # i=0
    # get_precision(modelList,i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, m_num_users),testR,trainR,write_rate)
    res = Parallel(n_jobs=NUM_THREADS, verbose=0, backend="threading")(delayed(get_precision)(modelList, i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, m_num_users), testR, trainR, write_rate) for i in range(n))
    data = []
    for d in res:
        data.extend(d)
    precision = np.array(data)
    return precision

