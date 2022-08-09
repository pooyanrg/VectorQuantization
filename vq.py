import numpy as np


def vq(X, K, maxIteration):
    (N, d) = X.shape
    c = np.random.rand(K, d)

    for i in range(maxIteration):
        dist = np.zeros([N, K])
        for j in range(K):
            cj = np.array(c[j, :], dtype=float, ndmin=2)
            cj_repeated = np.repeat(cj, N, axis=0)
            d = X - cj_repeated
            d **= 2
            d = np.sum(d, axis=1)
            dist[:, j] = d
        S = np.argmin(dist, axis=1)
        for j in range(K):
            index = (S == j)
            Sj = X[index, :]
            if (Sj.shape[0] > 0):
                cj = np.mean(Sj, axis=0)
            c[j, :] = cj

    return (c)


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

MnistTrainX = sio.loadmat ('../../../datasets/mnist/MnistTrainX')['MnistTrainX']
MnistTrainY = sio.loadmat ('../../../datasets/mnist/MnistTrainY')['MnistTrainY']
idx = MnistTrainY == 2
idx = idx.reshape ([idx.shape[0]])
X = MnistTrainX[idx,:]


K = 9
maxIteration = 1
c = vq(X,K,maxIteration)

L = int(np.sqrt(K-1)+1)
img = np.zeros([L*29,L*29])
for i in range (L):
    for j in range (L):
        k = i * L + j
        if (k < K):
            img[i*29:i*29+28, j*29:j*29+28] = np.reshape(c[k,:], [28,28])

imh = plt.imshow (img, cmap='gray');
plt.show()
