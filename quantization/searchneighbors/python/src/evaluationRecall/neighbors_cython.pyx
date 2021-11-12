import numpy as np
cimport numpy as np
cimport cython
from joblib import Parallel, delayed

from cython.parallel import prange

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def sort_topk_adc_score(str metric, np.ndarray[DTYPE_t, ndim=1] adc_score, topk):
    cdef np.ndarray[np.int64_t, ndim=1] ind 
    if metric == "dot_product":
        ind = np.argpartition(adc_score, -topk)[-topk:]
        return np.flip(ind[np.argsort(adc_score[ind])])

    if metric == "l2_distance":
        ind = np.argpartition(adc_score, topk)[0:topk]
        return ind[np.argsort(adc_score[ind])]

def compute_distance(np.ndarray[DTYPE_t, ndim=1] query, 
                        np.ndarray[DTYPE_t, ndim=3] pq_codebook,
                        np.ndarray[np.int64_t, ndim=2] codes, 
                        str metric, 
                        int M, 
                        int Ds, 
                        np.ndarray[np.int64_t, ndim=2] index):

    cdef np.ndarray[DTYPE_t, ndim=2] q = query.reshape(M, Ds)
    if metric == "dot_product":
        lookup_table = np.matmul(pq_codebook, q[:, :, np.newaxis])[:, :, 0]

        tem_value  = lookup_table[index, codes]
        inner_prod = np.sum(tem_value, axis=1)
        return inner_prod

    if metric == "l2_distance":
        lookup_table = np.linalg.norm(pq_codebook - q[:, np.newaxis, :], axis=2) ** 2
        dists = np.sum(lookup_table[range(M), codes], axis=1)
        return dists

def search_neighbors(np.ndarray[DTYPE_t, ndim=1] query, 
                                                int topk,
                                                np.ndarray[DTYPE_t, ndim=3] pq_codebook,
                                                np.ndarray[np.int64_t, ndim=2] codes,
                                                str metric,
                                                int M,
                                                int Ds,
                                                np.ndarray[np.int64_t, ndim=2] index):

    cdef np.ndarray[DTYPE_t, ndim=1] adc_score = compute_distance(query, pq_codebook, codes, metric, M, Ds, index)
    return sort_topk_adc_score(metric, adc_score, topk)

def neighbors(np.ndarray[DTYPE_t, ndim=2] queries, int topk,
                                                np.ndarray[DTYPE_t, ndim=3] pq_codebook,
                                                np.ndarray[np.int64_t, ndim=2] codes,
                                                str metric,
                                                int M,
                                                int Ds,
                                                np.ndarray[np.int64_t, ndim=2] index):
    """
    Args:
        queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
        topk (int): this method will return topk neighbors for each query
    Returns:
        np.ndarray: topk neighbors for each query with shape=(nq, topk)  
    
    """
    cdef int n = queries.shape[0]
    cdef np.ndarray[np.int64_t, ndim=2] neighbors_matrix = np.zeros((n, topk), dtype=int)
    cdef np.ndarray[DTYPE_t, ndim=1] q
    for i in range(n):
        q = queries[i]
        neighbors_matrix[i] = search_neighbors(q, topk, pq_codebook, codes, metric, M, Ds, index)

    return neighbors_matrix

# def par_neighbors_cython(np.ndarray[DTYPE_t, ndim=2] queries, 
#                                                 int topk,
#                                                 np.ndarray[DTYPE_t, ndim=3] pq_codebook,
#                                                 np.ndarray[np.int64_t, ndim=2] codes,
#                                                 str metric,
#                                                 int M,
#                                                 int Ds,
#                                                 np.ndarray[np.int64_t, ndim=2] index,
#                                                 int njobs=10):
#     """
#     Args:
#         queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
#         topk (int): this method will return topk neighbors for each query
#     Returns:
#         np.ndarray: topk neighbors for each query with shape=(nq, topk)  
    
#     """
#     assert queries.ndim == 2
#     cdef int n = queries.shape[0]
#     cdef np.ndarray[np.int64_t, ndim=2] neighbors_matrix = np.zeros((n, topk), dtype=int)
#     cdef np.ndarray[DTYPE_t, ndim=1] q

#     result = Parallel(n_jobs = njobs,backend='multiprocessing')(delayed(search_neighbors)(q, topk, pq_codebook, codes, metric, M, Ds, index) for q in queries)

#     for i in range(n):
#         neighbors_matrix[i] = result[i]
#     return neighbors_matrix

def par_neighbors_cython(queries, 
                                                topk,
                                                pq_codebook,
                                                codes,
                                                metric,
                                                M,
                                                Ds,
                                                index,
                                                njobs=10):
    """
    Args:
        queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
        topk (int): this method will return topk neighbors for each query
    Returns:
        np.ndarray: topk neighbors for each query with shape=(nq, topk)  
    
    """
    assert queries.ndim == 2
    n = queries.shape[0]
    neighbors_matrix = np.zeros((n, topk), dtype=int)

    result = Parallel(n_jobs = njobs,backend='multiprocessing')(delayed(search_neighbors)(q, topk, pq_codebook, codes, metric, M, Ds, index) for q in queries)

    for i in range(n):
        neighbors_matrix[i] = result[i]
    return neighbors_matrix

# def par_neighbors_prange(np.ndarray[DTYPE_t, ndim=2] queries, int topk,
#                                                 np.ndarray[DTYPE_t, ndim=3] pq_codebook,
#                                                 np.ndarray[np.int64_t, ndim=2] codes,
#                                                 str metric,
#                                                 int M,
#                                                 int Ds,
#                                                 np.ndarray[np.int64_t, ndim=2] index):
#     """
#     Args:
#         queries (np.ndarray): Input matrix with shape=(nq, D), where nq is the number of queries. 
#         topk (int): this method will return topk neighbors for each query
#     Returns:
#         np.ndarray: topk neighbors for each query with shape=(nq, topk)  
    
#     """
#     cdef int n = queries.shape[0]
#     cdef np.ndarray[np.int64_t, ndim=2] neighbors_matrix = np.zeros((n, topk), dtype=int)
#     cdef np.ndarray[DTYPE_t, ndim=1] q
#     for i in prange(n, num_threads=10):
#         q = queries[i]
#         neighbors_matrix[i] = search_neighbors(q, topk, pq_codebook, codes, metric, M, Ds, index)

#     return neighbors_matrix 