using NPZ
include("./aq.jl")

data = NPZ.npzread("/amax/home/zhangjin/Anisotropic-Quantization/data/glove/glove-100-angular-normalized.npy")

M = 50
K = 16
train_data = data[1:10000,:]
aq_codebook,aq_codes = aq_train(M, K, train_data)
npzwrite("aq_codebook.npy", aq_codebook)

aq_codebook = codebook_trans2To3(aq_codebook,M)
aq_codes,~ = aq_encode(data,aq_codebook, M)

aq_codes .-= 1
npzwrite("aq_codes.npy", aq_codes)
