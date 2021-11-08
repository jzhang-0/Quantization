using SparseArrays
using LinearAlgebra
using Krylov
const newaxis = [CartesianIndex()]

include("./utils.jl")
include("./utils_dec.jl")

@dec _time function aq_update_codebook(data::AbstractArray{<:AbstractFloat,2}, aq_codes_B::AbstractArray{<:Integer,2},K::Integer)
    """
    data size = (n, D)
    aq_codes_B size= (n, M)

    return
        aq_codebook size = (MK, D)    
    """
    (n,M) = size(aq_codes_B)
    (n,D) = size(data)

    B = zeros(Float32, n, M*K)
    for i in 1:n
        for j in 1:M
            B[i,aq_codes_B[i,j]] = 1
        end
    end

    B = sparse(B)
    # M1 = Matrix(transpose(B)*B) + I(M*K)
    # aq_codebook = inv(M1)*transpose(B)*data

    atol=1e-6
    btol=1e-6
    atol = convert(Float32, atol)
    btol = convert(Float32, btol)
    
    λ = 1.0e-6
    λ = convert(Float32, λ)
    λ > 0.0 && (N = I / λ)
    data = Float32.(data)
    
    aq_codebook = zeros(M*K,D)
    for dim in 1:D
        (aq_codebook[:,dim],~) = lsmr(B, data[:,dim], λ=λ, sqd=λ > 0, atol=atol, btol=btol, N=N)
    end
    # aq_codebook = data \ B
    return aq_codebook
end

function aq_single_encode_CoordinatesDown(
                point::AbstractArray{<:AbstractFloat,1},
                aq_codebook::AbstractArray{<:AbstractFloat, 3},
                M::Integer,
                iter=3
                )
    # point size = (D)
    # aq_codebook size= (M, K, D)
    (M,K,D) = size(aq_codebook)
    code = zeros(Int, M)
    code .+= 1

    compress_vec = zeros(D)
    for i in 1:M
        compress_vec += aq_codebook[i,code[i],:]
    end
    
    for t in 1:iter
        for i in 1:M
            compress_vec_deli = compress_vec - aq_codebook[i,code[i],:]
            r_i = point - compress_vec_deli
    
            r_vec = broadcast(-, r_i[newaxis,:], aq_codebook[i,:,:]) # (K,D)
            error = sum(r_vec.^2,dims=2)
            code[i] = argmin(error)[1]
            compress_vec = compress_vec_deli + aq_codebook[i,code[i],:]
        end
    end
    r = point - compress_vec
    error = sum(r.^2)
    return code, error
end

@dec _time function aq_encode(
    data::AbstractArray{<:AbstractFloat,2},
    aq_codebook::AbstractArray{<:AbstractFloat, 3},
    M::Integer,
)
    (n, D) = size(data)
    aq_codes = zeros(Int, n, M)
    loss = 0
    Threads.@threads for i in 1:n
        point = data[i,:]
        (code, error) = aq_single_encode_CoordinatesDown(point,aq_codebook,M)
        loss += error
        aq_codes[i,:] = code
    end
    # loss += sum(aq_codebook.^2)
    return aq_codes,loss
end

function decode(aq_code::AbstractArray{<:Integer,1},aq_codebook::AbstractArray{<:AbstractFloat, 3})
    (M,K,D) = size(aq_codebook)
    compress_vec = zeros(D)
    for i in 1:M
        compress_vec += aq_codebook[i, aq_code[i], :]
    end
    return compress_vec
end

function compute_loss(aq_codes, aq_codebook, data)
    (M,K,D) = size(aq_codebook)
    (n,D) = size(data)

    loss = 0
    for i in 1:n
        compress_vec = decode(aq_codes[i,:], aq_codebook)
        loss += sum((data[i,:] - compress_vec).^2)
    end
    return loss    
end

function aq_codebook_init(M::Integer, K::Integer, data::AbstractArray{<:AbstractFloat,2})
    (n,D) = size(data)
    aq_codebook = zeros(M, K, D)
    randix = rand(1:n, M, K)
    for i in 1:M
        for j in 1:K
            aq_codebook[i,j,:] = data[randix[i,j],:]
        end
    end
    return aq_codebook
end

function aq_train(
    M::Integer, 
    K::Integer, 
    data::AbstractArray{<:AbstractFloat,2},
    maxiter=20)
    
    aq_codebook = aq_codebook_init(M,K,data) # ndim=3
    aq_codes,loss = aq_encode(data,aq_codebook,M)
    println("loss = ",loss)
    loss_change = 1

    i = 0
    while (i<=maxiter) & (abs(loss_change) > 0.01)
        i += 1
        println("iter_num:",i)
        aq_codes_B = aq_codesTo_codesB(aq_codes, K)
        aq_codebook = aq_update_codebook(data, aq_codes_B, K) # ndim = 2
        aq_codebook = codebook_trans2To3(aq_codebook, M)

        old_loss = loss

        aq_codes, loss = aq_encode(data, aq_codebook, M)

        new_loss = loss
        loss_change = round((old_loss - new_loss )/old_loss, digits = 2)
        println("newloss = ",new_loss, ",loss_change = ",loss_change)    
    end

    aq_codebook = codebook_trans3To2(aq_codebook)
    return aq_codebook,aq_codes
    
end

