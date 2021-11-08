
function codebook_trans3To2(aq_codebook::AbstractArray{<:AbstractFloat, 3})
    (M,K,D) = size(aq_codebook)
    aq_codebook_2d = zeros(M*K,D)
    for i in 1:M
        for j in 1:K
            aq_codebook_2d[(i-1)*K + j,:] = aq_codebook[i,j,:]
        end
    end
    return aq_codebook_2d
end 

function codebook_trans2To3(aq_codebook::AbstractArray{<:AbstractFloat, 2},M::Integer)
    (MK,D) = size(aq_codebook)
    K = Int(MK / M)

    aq_codebook_3d = zeros(M, K, D)
    for i in 1:M
        for j in 1:K
            aq_codebook_3d[i,j,:] = aq_codebook[(i-1)*K + j,:]
        end
    end
    return aq_codebook_3d
end

function aq_codesTo_codesB(aq_codes::AbstractArray{<:Integer,2}, K::Integer)
    # aq_codes = (n, M)
    # self.aq_codes_2 = aq_codes + np.arange(M) * K
    rangeM = (1:M) .- 1
    Arr1 = convert(Array{Int64,1}, rangeM.*K)
    codesB = broadcast(+,aq_codes,Arr1[[CartesianIndex()],:])
    return codesB
end