using NPZ
using ArgParse

s = ArgParseSettings()
include("./args.jl")
s = args_data(s)
s = args_pq(s)
s = args_vq(s)
config = parse_args(s)

pq_codes = NPZ.npzread(config["pq_codes"]);
pq_codebook = NPZ.npzread(config["pq_codebook"]);
pq_codes .+= 1

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

time = @elapsed neighbors_MA = SN_pqivf.get_neighbors(spi, queries, 20, 512);
println(time);

include("./utils.jl")
recall_atN(neighbors_MA, tr100)

