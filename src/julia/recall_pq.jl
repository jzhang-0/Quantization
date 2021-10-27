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

queries = NPZ.npzread(config["queries"]);
tr100 = NPZ.npzread(config["tr100"])
tr100 .+= 1;


include("./neighbors.jl")
sp = SN_pq.SearchNeighbors_PQ(M=config["M"], Ks=config["Ks"], D=config["D"], metric = config["metric"], 
pq_codebook = pq_codebook, pq_codes = pq_codes);

time = @elapsed neighbors_MA = SN_pq.get_neighbors(sp, queries, 512);
println(time);

include("./utils.jl")
recall_atN(neighbors_MA, tr100)





