* python 向量化实现查表:100s 
* matlab 向量化实现查表:50s (80%时间在索引上)
* matlab for实现查表:非常慢 
* julia for实现查表:64s (多线程(36)下: 4s)
    * @inbounds 54s
* julia 向量化实现查表:78s (多线程(36)下: 4s)
