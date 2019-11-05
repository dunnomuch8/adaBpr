from lightfm import LightFM
import numpy as np
from scipy.sparse import csr_matrix
from eval import get_accuracy, model, get_top
from joblib import Parallel, delayed
import logging
FORMAT = "[%(asctime)-15s] %(message)s"
logging.basicConfig(format=FORMAT)  # , filename='ada.log')
logger = logging.getLogger('adaboost')
NUM_THREADS = 8
BATCH_SIZE = 5000
N_FACTORS = 400
N_EPOCHS = 100


def bpr(train_data, u):
    model = LightFM(no_components=N_FACTORS, learning_schedule="adagrad", loss="warp", max_sampled=30)
    logger.warning('fit')
    model.fit(train_data, epochs=N_EPOCHS, num_threads=NUM_THREADS, verbose=False, sample_weight=u)
    bv, V = model.get_item_representations()
    bu, U = model.get_user_representations()
    return U, V, bu, bv


def adaboost(test_data, train_data, iteration, saveTime, modelList):
    n_u, n_v = train_data.shape
    betau = np.repeat(1 / train_data.getnnz(1), np.diff(train_data.indptr))
    u = csr_matrix((betau, train_data.indices, train_data.indptr), shape=train_data.shape)
    if modelList is not None:
        precisionTR = get_accuracy(modelList, train_data, None)
        saveTime = modelList[0].Rate is not None or False
    else:
        modelList = []
        precisionTR = np.zeros_like(betau)
        if saveTime:
            logger.warning("mode:savePredict")
        else:
            logger.warning("mode:saveModel")
    for t in range(iteration):
        u.data = betau * np.exp(-precisionTR)
        u.data /= u.data.sum()
        u.data *= u.nnz
        U, V, bu, bv = bpr(train_data, u.tocoo())
        single_model = model(U, V, bu, bv)
        if len(modelList) == 0 and saveTime:
            modelList.append(model(Rate=np.zeros((n_u, n_v), dtype=np.float32)))
            precisionTR = get_accuracy([single_model], train_data, None, modelList[0].Rate)  
            modelList[0].Rate *= 0.5 * np.log((u.data * (1 + precisionTR)).sum() / (u.data * (1 - precisionTR)).sum())
            # print(get_recall(modelList,  test_data, train_data, [49,99]))
            continue

        precisionTR = get_accuracy([single_model], train_data, None)  
        single_model.alpha = 0.5 * np.log((u.data * (1 + precisionTR)).sum() / (u.data * (1 - precisionTR)).sum())
        if saveTime:
            precisionTR = get_accuracy([single_model], train_data, None, modelList[0].Rate)  # write R*alp
        else:
            modelList.append(single_model)
            if len(modelList) > 1:
                precisionTR = get_accuracy(modelList, train_data, None)
        # print(get_recall(modelList,  test_data, train_data, [49, 99]))
    modelList[0].inids = np.asarray((train_data.sum(0) > 0))[0]
    return modelList


def getRecList(ensemble, num_rec, trainR=None, test_users=None, availableItems=None):
    if trainR is not None:
        assert trainR.shape[1] == (len(availableItems) if availableItems != slice(None) else ensemble[0].m_V.shape[0])
        assert trainR.shape[0] == (len(test_users) if test_users else ensemble[0].m_U.shape[0])
    m_num_users = ensemble[0].n_users
    n = m_num_users // BATCH_SIZE + 1
    tops = Parallel(n_jobs=NUM_THREADS, verbose=0, backend="threading")(delayed(lambda x: get_top(*x)[:, :num_rec])([ensemble, i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, m_num_users), trainR, None, test_users, availableItems]) for i in range(n))
    return np.concatenate(tops, 0)
