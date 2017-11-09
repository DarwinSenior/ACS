import pandas as pd
import numpy as np
import scipy as sp
from scipy.linalg import eigh, sqrtm
from scipy.cluster.vq import kmeans2, whiten

def read_ucidata_gama(filename):
    f = pd.read_csv(filename, comment='%', header=None)
    s, t = f[0], f[1]
    min_i = min(s.min(), t.min())
    max_i = max(s.max(), t.max())
    size = max_i - min_i + 1
    W = np.zeros((size, size))
    for s, t, r in f.itertuples(index=False, name=None):
        d1 = s - min_i
        d2 = t - min_i
        W[d1, d2] = r
        W[d2, d1] = r
    return W

def read_bitcoin(filename):
    f = pd.read_csv(filename)
    s, t = f['source'], f['target']
    min_i, max_i = (min(s.min(), t.min()), max(s.max(), t.max()))
    size = max_i - min_i + 1
    W = np.zeros((size, size))
    for s, t, r, _ in f.itertuples(index=False, name=None):
        d1 = s - min_i
        d2 = t - min_i
        assert (W[d1, d2] == r or W[d1, d2] == 0)
        W[d1, d2] = r
        W[d2, d1] = r
    return W


def splitGraph(W: np.ndarray):
    Wp = np.copy(W)
    Wm = np.copy(W)
    Wp[W<0] = 0
    Wm[W>0] = 0
    return Wp, Wm

def D(W):
    return np.diag(np.sum(W, axis=1))

def LBR(W):
    Wp, Wm = splitGraph(W)
    Dp = D(Wp)
    return Dp - Wp + Wm

def LBN(W):
    Dpm = D(W)
    return (1/Dpm)@(LBR(W))

def LSR(W):
    Dpm = D(W)
    Wp, Wm = splitGraph(W)
    return Dpm - Wp + Wm

def LSN(W):
    Dpm = D(W)
    Dqrt = Dpm**(-1.2)
    return Dqrt @ LSR(W) @ Dqrt

def spectual_clustering(L, k):
    v, w = eigh(L, eigvals=(0, k-1))
    centroid, label = kmeans2(whiten(w), k)
    return label


if __name__ == '__main__':
    W = read_ucidata_gama('./ucidata-gama.edges')
    print(W)
