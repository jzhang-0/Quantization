using NPZ
vq_codes = npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/testdata/vq_code_kv400.npy")
vq_codebook = npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/testdata/vq_code_book_kv400.npy")

pq_codes = npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/testdata/code_Kv400_M8_K16_sample_num5000.npy")
pq_codebook = npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/testdata/pq_codebook_Kv400_M8_K16_sample_num5000.npy")
tr100 = NPZ.npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/true_neighbors_top100_32D.npy")

tr100 .+= 1 
vq_codes .+= 1
pq_codes .+= 1

queries = npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/queries_32D.npy")

i=1
query = queries[i,:]
metric = "dot_product"

M=8
Ks=16
D=32
Ds=4
num_centroid_tosearch=20

q = query
topk = 512



struct P1
    M::Int8
    Ks::Int8
    metric::String
end

struct P2
    D::Int16
    Ds::Int16 
    k_v::Int64
    function P2(D,k_v)
        Ds = D*k_v
        new(D, Ds, k_v)
    end
end


struct C
    M::Int8
    Ks::Int8
    metric::String

    D::Int16
    Ds::Int16 
    k_v::Int64

    function C(M, Ks, metric, D, k_v)
        p1 = P1(M, Ks, metric)
        M, Ks, metric = p1.M, p1.Ks, p1.metric
        
        p2 = P2(D,k_v)
        D, Ds, k_v = p2.D, p2.Ds, p2.k_v 

        new(M, Ks, metric, D, Ds, k_v)
    end 
end






export 
    AbstractSN,f


abstract type AbstractSN end

function f(a)
    println(a)
end

# @benchmark 


module T
    # using Main:f
    using Main
    function test(a)
        f(a)
    end

    function tt(a::AbstractSN)
        println(a)
    end
end
