using NPZ

code = NPZ.npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/code_M8_K16_sample_num10000.npy");
codebook = NPZ.npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/codebook.npy");

queries = NPZ.npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/queries_32D.npy");
t100 = NPZ.npzread("/amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/true_neighbors_top100_32D.npy")

code .+= 1
module Nm
    include("neighbors.jl")
end
sp = Nm.SearchNeighbors_PQ(M=8, Ks=16, D=32, metric = "dot_product", pq_codebook = codebook, pq_codes = code);
# Nm.exinit(sp);
time = @elapsed Nm.get_neighbors(sp, queries, 512)
println(time);

# julia --threads 36 /amax/home/zhangjin/scann+/QuantizationEvaluation/src/Neighbors_jl/recall.jl


