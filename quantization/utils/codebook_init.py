import numpy as np
np.random.seed(15)

def PQ_codebook_init(cls, dataset):
    pq_codebook = np.zeros((cls.M, cls.Ks, cls.Ds))
    n = dataset.shape[0]
    points = dataset[np.random.choice(n, cls.Ks, replace=False)]

    for i,point in enumerate(points):
        pq_codebook[:,i,:] = point.reshape(cls.M, cls.Ds)

    return pq_codebook
