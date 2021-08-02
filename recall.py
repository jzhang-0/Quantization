import numpy as np
# from joblib import Parallel, delayed

class Recall:
    def compute_recall(self, neighbors, ground_truth):
        total = 0
        for gt_row, row in zip(ground_truth, neighbors):
            total += np.intersect1d(gt_row, row).shape[0]
        return total / ground_truth.size

    def brute_force_search(self, target_set, test_set, option):
        if option != "dot_product" and option != "l2_distance":
            raise Exception(f"not suport {option},optional:l2_distance or dot_product")

        if option == "dot_product":
            inner_product = target_set @ test_set.T
            return np.argmax(inner_product, axis=0)

        if option == "l2_distance":
            q_2 = np.linalg.norm(test_set, axis=1).reshape(-1,1) ** 2
            x_2 = np.linalg.norm(target_set, axis=1).reshape(1,-1) ** 2

            q_x = test_set @ target_set.T
            gt_distance = q_2 - 2*q_x + x_2

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
        option (str): dot_product or l2_distance        
    
    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        D (int): The dim of each vector
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M
        pq_codebook (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32, codebook[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        pq_codes (np.ndarray): PQ codes with shape=(n, M) and dtype=np.int
        option (str): dot_product or l2_distance        

    """

    def __init__(self, M, Ks, D, pq_codebook, pq_codes, option="l2_distance") -> None:
        self.M = M
        self.Ks = Ks
        self.D = D
        assert D%M == 0, "D must be divisible by M"
        self.Ds = D // M    

        self.pq_codebook = pq_codebook
        self.pq_codes = pq_codes
        assert pq_codebook.shape == (M,Ks,self.Ds),"pq_codebook.shape must equal to (M,Ks,Ds)"

        self.option = option
        if option != "dot_product" and option != "l2_distance":
            raise Exception(f"not suport {option},optional:l2_distance or dot_product")        
    
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
        option = self.option

        M = self.M
        Ds = self.Ds

        q = query.reshape(M, Ds)
        if option == "dot_product":
            lookup_table = np.matmul(pq_codebook, q[:, :, np.newaxis])[:, :, 0]
            i1 = np.arange(M)
            inner_prod = np.sum(lookup_table[i1, codes[..., i1]], axis=1)  
            return inner_prod

        if option == "l2_distance":
            lookup_table =  np.linalg.norm(pq_codebook - q[:, np.newaxis,:], axis=2) ** 2
            dists = np.sum(lookup_table[range(M), codes], axis=1)
            return dists

    def _sort_topk_adc_score(self, adc_score, topk):
        option = self.option

        if option == "dot_product":
            ind = np.argpartition(adc_score, -topk)[-topk:]
            return np.flip(ind[np.argsort(adc_score[ind])])

        if option == "l2_distance":
            ind = np.argpartition(adc_score,topk)[0:topk]
            return ind[np.argsort(adc_score[ind])]


    def search_neightbors(self, query, topk):
        """
        Args:
            pq_codebook, codes, query, option are the same as method compute_distance
            topk (int): this method will return topk neighbors

        Returns:
            index (np.darray): query's topk neighbors  

        """
        adc_score = self.compute_distance(query)
        return self._sort_topk_adc_score(adc_score,topk)

    def neighbors(self, queries, topk):
        """
        Args:
            queries (np.ndarray): Input matrix with shape=(nq, D) 
            topk (int): this method will return topk neighbors for each query

        Returns:
            np.ndarray: topk neighbors for each query with shape=(nq, topk)  
        
        """        
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)
        for i in range(n):
            q = queries[i]
            neighbors_matrix[i] = self.search_neightbors(q, topk)

        return neighbors_matrix

class Recall_PQIVF(Recall_PQ):
    """
    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        D (int): The dim of each vector
    
    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        D (int): The dim of each vector
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M

        query (np.ndarray): Input vector with shape=(D, )

        vq_code_book (np.ndarray): shape=(k', D) with dtype=np.float32.
            codebook[m] means m-th codeword (D-dim)

        vq_codes (np.ndarray): VQ codes with shape=(n, ) and dtype=np.int

        topk (int): this method will return topk neighbors
    """

    def __init__(self, M, Ks, D, pq_codebook, pq_codes,vq_codes, vq_code_book, option) -> None:
        super().__init__(M, Ks, D, pq_codebook, pq_codes, option=option)
        self.vq_codes = vq_codes
        self.vq_code_book = vq_code_book

    def ivf_search_neightbors(self,query,topk):

        adc_score = self.compute_distance(query)
        return self._sort_topk_adc_score(adc_score,topk)        


    