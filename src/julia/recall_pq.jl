using NPZ
using ArgParse

s = ArgParseSettings()
include("./args.jl")
s = args_data(s)
s = args_pq(s)

config = parse_args(s)

pq_codes = NPZ.npzread(config["pq_codes"]);
pq_codes .+= 1

pq_codebook = NPZ.npzread(config["pq_codebook"]);

M=config["M"]
Ks = config["Ks"]
D = config["D"]
Ds = Int(D/M)

if length(size(pq_codebook))==1
    pq_codebook_ = zeros(Float32, M, Ks, Ds)
    for i in 1:M
        cc = pq_codebook[(i-1)*Ks*Ds + 1 : i*Ks*Ds]
        cc =reshape(cc, Ds, Ks)
        cc = transpose(cc)
        pq_codebook_[i,:,:] = cc
    end
end
pq_codebook = pq_codebook_


queries = NPZ.npzread(config["queries"]);
tr100 = NPZ.npzread(config["tr100"])
tr100 .+= 1;


include("./neighbors.jl")
sp = SN_pq.SearchNeighbors_PQ(config["M"], config["Ks"], config["D"], pq_codebook, pq_codes, config["metric"]);

time = @elapsed neighbors_MA = SN_pq.get_neighbors(sp, queries, 512);
println(time);

include("./utils.jl")
recall_atN(neighbors_MA, tr100)

show(to)



