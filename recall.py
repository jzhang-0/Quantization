import numpy as np
from scipy.spatial.distance import cdist

# from joblib import Parallel, delayed

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
        assert D%M == 0, "D must be divisible by M"
        self.Ds = D // M    

        self.pq_codebook = pq_codebook
        self.pq_codes = pq_codes
        assert pq_codebook.shape == (M,Ks,self.Ds),"pq_codebook.shape must equal to (M,Ks,Ds)"

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
            i1 = np.arange(M)
            inner_prod = np.sum(lookup_table[i1, codes[..., i1]], axis=1)  
            return inner_prod

        if metric == "l2_distance":
            lookup_table =  np.linalg.norm(pq_codebook - q[:, np.newaxis,:], axis=2) ** 2
            dists = np.sum(lookup_table[range(M), codes], axis=1)
            return dists

    def _sort_topk_adc_score(self, adc_score, topk):
        metric = self.metric

        if metric == "dot_product":
            ind = np.argpartition(adc_score, -topk)[-topk:]
            return np.flip(ind[np.argsort(adc_score[ind])])

        if metric == "l2_distance":
            ind = np.argpartition(adc_score,topk)[0:topk]
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

        self.neighbors_matrix = neighbors_matrix

        return neighbors_matrix

    def pq_recall(self, queries, topk, ground_truth):
        ground_truth = np.array(ground_truth)

        try:
            neighbors_matrix = self.neighbors_matrix[:,0:topk]
            if topk > neighbors_matrix.shape[1]:
                neighbors_matrix = self.neighbors(queries,topk)

        except AttributeError:
            neighbors_matrix = self.neighbors(queries,topk)

        recall = self.compute_recall(neighbors_matrix,ground_truth)

        nr = neighbors_matrix.shape[1]
        if ground_truth.ndim == 1:
            ng = 1
        if ground_truth.ndim == 2 :
            ng = ground_truth.shape[1]
        
        print(f"recall {ng}@{nr} = {recall}")

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
            vq_code_book[m] means m-th codeword (D-dim)

        vq_codes (np.ndarray): VQ codes with shape=(n, ) and dtype=np.int

        topk (int): this method will return topk neighbors
    """

    def __init__(self, M, Ks, D, pq_codebook, pq_codes, vq_code_book,vq_codes, metric) -> None:
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

    @profile
    def _vq(self, query, num_centroids_to_search):
        if self.metric == "dot_product":
            inner_M = self.vq_code_book @ query
            c_id = np.argsort(inner_M)[-num_centroids_to_search:]
            c_id = np.flip(c_id)

            inner_1 = inner_M[c_id]

            return c_id, inner_1  

        if self.metric == "l2_distance":
            dist = np.linalg.norm(query-self.vq_code_book,axis=1)**2
            c_id = np.argsort(dist)[0:num_centroids_to_search]

            dist1 = dist[c_id]
            return c_id, dist1            

    @profile
    def search_neighbors_IVFADC(self, query, num_centroids_to_search, topk=64):
        metric = self.metric
        pq_codebook = self.pq_codebook
        pq_codes = self.pq_codes

        M = self.M
        Ds = self.Ds

        cluster_id,adc1 = self._vq(query,num_centroids_to_search)
        index = self.dataIndex_tosearch(cluster_id)

        if len(index) < topk:
            raise Exception("num_centroids_to_search are too small")

        adc = np.zeros(len(index))

        # i1 = 0
        # i2 = 0
        # for i,a1 in zip(cluster_id,adc1):
        #     i2 += len(self.vq_cluster[i])
        #     adc[i1:i2] = a1
        #     i1 = i2

        if metric == "dot_product":
            q = query.reshape(M, Ds)
            adc = np.array([adc1[i]  for i in range(num_centroids_to_search) for j in range(len(self.vq_cluster[cluster_id[i]]))])

            lookup_table = np.matmul(pq_codebook, q[:, :, np.newaxis])[:, :, 0]
            inner_prod_2 = np.sum(lookup_table[range(M), pq_codes[index, :]], axis=1)  
            adc = adc + inner_prod_2
            
        if metric == "l2_distance":
            i1 = 0
            i2 = 0
            for i in cluster_id:
                q_i = query - self.vq_code_book[i]
                q_i = q_i.reshape(M, Ds)

                lookup_table =  np.linalg.norm(pq_codebook - q_i[:, np.newaxis,:], axis=2) ** 2
                dists = np.sum(lookup_table[range(M), pq_codes[self.vq_cluster[i], :]], axis=1)
                
                i2 += len(self.vq_cluster[i])
                adc[i1:i2] = dists
                i1 = i2
        index = np.array(index)
        return index[self._sort_topk_adc_score(adc,topk)]

    def neighbors_ivf(self,queries,num_centroids_to_search,topk):
        n = queries.shape[0]
        neighbors_matrix = np.zeros((n, topk), dtype=int)
        for i in range(n):
            q = queries[i]
            neighbors_matrix[i] = self.search_neighbors_IVFADC(q,num_centroids_to_search, topk)

        return neighbors_matrix

    def pqivf_recall(self,queries,num_centroids_to_search,topk,ground_truth):
        ground_truth = np.array(ground_truth)

        neighbors_matrix = self.neighbors_ivf(queries, num_centroids_to_search,  topk)
        recall = self.compute_recall(neighbors_matrix,ground_truth)
        
        nr = neighbors_matrix.shape[1]
        if ground_truth.ndim == 1:
            ng = 1
        if ground_truth.ndim == 2 :
            ng = ground_truth.shape[1]
        
        print(f"recall {ng}@{nr} = {recall}")     


    