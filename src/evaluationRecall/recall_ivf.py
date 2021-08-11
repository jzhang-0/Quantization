import numpy as np
from .recall import Recall_PQ

class Recall_PQIVF(Recall_PQ):
    """
    the asymmetric distance computation combined with an inverted file system (IVFADC) 

    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        D (int): The dim of each vector
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M

        vq_code_book (np.ndarray): shape=(k', D) with dtype=np.float32.
            vq_code_book[m] means m-th codeword (D-dim)
        vq_codes (np.ndarray): VQ codes with shape=(n, ) and dtype=np.int

        pq_codebook (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
            codebook[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        pq_codes (np.ndarray): PQ codes with shape=(n, M) and dtype=np.int

        metric (str): dot_product or l2_distance        
    """

    def __init__(self, M, Ks, D, pq_codebook, pq_codes, vq_code_book, vq_codes, metric) -> None:
        super().__init__(M, Ks, D, pq_codebook, pq_codes, metric=metric)
        self.vq_codes = vq_codes
        self.vq_code_book = vq_code_book

        k_v = vq_code_book.shape[0]
        self.vq_cluster = [[] for i in range(k_v)]
        data_index = 0
        for i in vq_codes:
            self.vq_cluster[i].append(data_index)
            data_index += 1

    def dataIndex_tosearch(self, cluster_id):
        index = []
        for i in cluster_id:
            index += self.vq_cluster[i]
        return index

    def _vq(self, query, num_centroids_to_search):
        if self.metric == "dot_product":
            inner_M = self.vq_code_book @ query
            c_id = np.argsort(inner_M)[-num_centroids_to_search:]
            c_id = np.flip(c_id)

            inner_1 = inner_M[c_id]

            return c_id, inner_1

        if self.metric == "l2_distance":
            dist = np.linalg.norm(query - self.vq_code_book, axis=1) ** 2
            c_id = np.argsort(dist)[0:num_centroids_to_search]

            dist1 = dist[c_id]
            return c_id, dist1

    def search_neighbors_IVFADC(self, query, num_centroids_to_search, topk=64):
        metric = self.metric
        pq_codebook = self.pq_codebook
        pq_codes = self.pq_codes

        M = self.M
        Ds = self.Ds

        cluster_id, adc1 = self._vq(query, num_centroids_to_search)
        index = self.dataIndex_tosearch(cluster_id)

        if len(index) < topk:
            raise Exception("num_centroids_to_search are too small")

        adc = np.zeros(len(index))
        if metric == "dot_product":
            q = query.reshape(M, Ds)
            adc = np.array(
                [adc1[i] for i in range(num_centroids_to_search) for j in range(len(self.vq_cluster[cluster_id[i]]))])

            lookup_table = np.matmul(pq_codebook, q[:, :, np.newaxis])[:, :, 0]
            inner_prod_2 = np.sum(lookup_table[range(M), pq_codes[index, :]], axis=1)
            adc = adc + inner_prod_2

        if metric == "l2_distance":
            i1 = 0
            i2 = 0
            for i in cluster_id:
                q_i = query - self.vq_code_book[i]
                q_i = q_i.reshape(M, Ds)

                lookup_table = np.linalg.norm(pq_codebook - q_i[:, np.newaxis, :], axis=2) ** 2
                dists = np.sum(lookup_table[range(M), pq_codes[self.vq_cluster[i], :]], axis=1)

                i2 += len(self.vq_cluster[i])
                adc[i1:i2] = dists
                i1 = i2
        index = np.array(index)
        return index[self._sort_topk_adc_score(adc, topk)]

    def neighbors_ivf(self, queries, num_centroids_to_search, topk):
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)
        for i in range(n):
            q = queries[i]
            neighbors_matrix[i] = self.search_neighbors_IVFADC(q, num_centroids_to_search, topk)

        return neighbors_matrix

    def pqivf_recall(self, queries, num_centroids_to_search, topk, ground_truth):
        ground_truth = np.array(ground_truth)

        neighbors_matrix = self.neighbors_ivf(queries, num_centroids_to_search, topk)
        recall = self.compute_recall(neighbors_matrix, ground_truth)

        nr = neighbors_matrix.shape[1]
        if ground_truth.ndim == 1:
            ng = 1
        if ground_truth.ndim == 2:
            ng = ground_truth.shape[1]

        print(f"recall {ng}@{nr} = {recall}")
