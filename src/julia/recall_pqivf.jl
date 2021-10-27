using NPZ
using ArgParse

s = ArgParseSettings()
include("./args.jl")
s = args_data(s)
s = args_pq(s)
s = args_vq(s)
s = args_recall(s)

config = parse_args(s)

pq_codes = NPZ.npzread(config["pq_codes"]);
pq_codebook = NPZ.npzread(config["pq_codebook"]);
pq_codes .+= 1

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
        global pq_codebook = pq_codebook_
    end
end
# Assert length(size(pq_codebook)) == 3;
vq_codes = NPZ.npzread(config["vq_codes"]);
vq_codebook = NPZ.npzread(config["vq_codebook"]);
vq_codes .+= 1

queries = NPZ.npzread(config["queries"]);
tr100 = NPZ.npzread(config["tr100"])
tr100 .+= 1;


include("./neighbors.jl")

spi = SN_pqivf.SearchNeighbors_PQIVF( 
    config["M"], 
    config["Ks"], 
    pq_codebook, 
    pq_codes, 
    vq_codebook, 
    vq_codes,
    config["metric"])

time = @elapsed neighbors_MA = SN_pqivf.get_neighbors(spi, queries, config["num_leaves_to_search"], 512);
println(time);

include("./utils.jl")
recall_atN(neighbors_MA, tr100)

