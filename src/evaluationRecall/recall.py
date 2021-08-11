import numpy as np

class Recall:
    def compute_recall(self, neighbors, ground_truth):
        total = 0
        for gt_row, row in zip(ground_truth, neighbors):
            total += np.intersect1d(gt_row, row).shape[0]
        return total / ground_truth.size

    def brute_force_search(self, target_set, test_set, metric="l2_distance"):
        if metric != "dot_product" and metric != "l2_distance":
            raise Exception(f"not suport {metric},optional:l2_distance or dot_product")

        if metric == "dot_product":
            inner_product = target_set @ test_set.T
            return np.argmax(inner_product, axis=0)

        if metric == "l2_distance":
            q_2 = np.linalg.norm(test_set, axis=1).reshape(-1, 1) ** 2
            x_2 = np.linalg.norm(target_set, axis=1).reshape(1, -1) ** 2

            q_x = test_set @ target_set.T
            gt_distance = q_2 - 2 * q_x + x_2

            gt = np.argmin(gt_distance, axis=1)
            return gt


class Recall_PQ(Recall):
    """
    For the PQ indexing phase of database vectors,
    a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `Ks` codewords.

    For the querying phase, given a new `D`-dim query vector, the distance beween the query
    and the database PQ-codes are efficiently approximated via Asymmetric Distance.

    All vectors must be np.ndarray 

    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011
    
    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        D (int): The dim of each vector
        pq_codebook (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
            codebook[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        pq_codes (np.ndarray): PQ codes with shape=(n, M) and dtype=np.int
        metric (str): dot_product or l2_distance        
    
    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        D (int): The dim of each vector
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M
        pq_codebook (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32, codebook[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        pq_codes (np.ndarray): PQ codes with shape=(n, M) and dtype=np.int
        metric (str): dot_product or l2_distance

    """

    def __init__(self, M, Ks, D, pq_codebook, pq_codes, metric="l2_distance") -> None:
        self.M = M
        self.Ks = Ks
        self.D = D
        assert D % M == 0, "D must be divisible by M"
        self.Ds = D // M

        self.pq_codebook = pq_codebook
        self.pq_codes = pq_codes
        assert pq_codebook.shape == (M, Ks, self.Ds), "pq_codebook.shape must equal to (M,Ks,Ds)"

        self.metric = metric
        if metric != "dot_product" and metric != "l2_distance":
            raise Exception(f"not suport {metric},optional:l2_distance or dot_product")

    def compute_distance(self, query):
        """
        The distances (the squared Euclidean distance or inner product) are computed by comparing each sub-vector of the query
        to the codewords for each sub-subspace.
        `lookup_table[m][ks]` contains the squared Euclidean distance or inner product between
        the `m`-th sub-vector of the query and the `ks`-th codeword
        for the `m`-th sub-space (`self.codewords[m][ks]`).
        By looking up table to compute distance.

        Args:
            query (np.ndarray): Input vector with shape=(D, ) 

        Returns:
            np.ndarray: Asymmetric Distances with shape=(n, )  

        """
        pq_codebook = self.pq_codebook
        codes = self.pq_codes
        metric = self.metric

        M = self.M
        Ds = self.Ds

        q = query.reshape(M, Ds)
        if metric == "dot_product":
            lookup_table = np.matmul(pq_codebook, q[:, :, np.newaxis])[:, :, 0]
            inner_prod = np.sum(lookup_table[range(M), codes], axis=1)
            return inner_prod

        if metric == "l2_distance":
            lookup_table = np.linalg.norm(pq_codebook - q[:, np.newaxis, :], axis=2) ** 2
            dists = np.sum(lookup_table[range(M), codes], axis=1)
            return dists

    def _sort_topk_adc_score(self, adc_score, topk):
        metric = self.metric

        if metric == "dot_product":
            ind = np.argpartition(adc_score, -topk)[-topk:]
            return np.flip(ind[np.argsort(adc_score[ind])])

        if metric == "l2_distance":
            ind = np.argpartition(adc_score, topk)[0:topk]
            return ind[np.argsort(adc_score[ind])]

    def search_neightbors(self, query, topk):
        """
        Args:
            pq_codebook, codes, query, metric are the same as method compute_distance
            topk (int): this method will return topk neighbors

        Returns:
            index (np.darray): query's topk neighbors  

        """
        adc_score = self.compute_distance(query)
        return self._sort_topk_adc_score(adc_score, topk)

    def neighbors(self, queries, topk):
        """
        Args:
            queries (np.ndarray): Input matrix with shape=(nq, D) 
            topk (int): this method will return topk neighbors for each query

        Returns:
            np.ndarray: topk neighbors for each query with shape=(nq, topk)  
        
        """
        assert queries.ndim == 2
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)
        for i in range(n):
            q = queries[i]
            neighbors_matrix[i] = self.search_neightbors(q, topk)

        self.neighbors_matrix = neighbors_matrix

        return neighbors_matrix

    def pq_recall(self, queries, topk, ground_truth):
        ground_truth = np.array(ground_truth)

        try:
            if topk > self.neighbors_matrix.shape[1]:
                neighbors_matrix = self.neighbors(queries, topk)
            else:
                neighbors_matrix = self.neighbors_matrix[:, 0:topk]
        except AttributeError:
            neighbors_matrix = self.neighbors(queries, topk)

        recall = self.compute_recall(neighbors_matrix, ground_truth)

        nr = neighbors_matrix.shape[1]
        if ground_truth.ndim == 1:
            ng = 1
        if ground_truth.ndim == 2:
            ng = ground_truth.shape[1]

        print(f"recall {ng}@{nr} = {recall}")
