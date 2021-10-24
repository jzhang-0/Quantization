abstract type SearchNeighbors end

Base.@kwdef struct SearchNeighbors_PQ <: SearchNeighbors
    M::Int8
    Ks::Int8
    D::Int16
    Ds = Int16(D/M) 
    pq_codebook::Array{Float32, 3} = rand(Float32,(1,2,3))
    pq_codes::Array{Int64, 2} = rand(Int8, (5,3))
    metric::String
end

# typeof(x)
# x = SearchNeighbors_PQ(M=8, Ks=16, D=32, metric = "dot_product")

function exinit(sp::SearchNeighbors_PQ)
    for i in 1:sp.M
        sp.pq_codes[:,i] .+= sp.Ks*(i-1);
    end    
end

function maxk!(ix, a, k; initialized=false)
    # k:topk
    # ix = collect(1:length(a))

    partialsortperm!(ix, a, 1:k, rev=true, initialized=initialized)
    # @views collect(zip(ix[1:k], a[ix[1:k]]))
    @views ix[1:k]
end


function compute_scores(sp::SearchNeighbors, query::Array{Float64,1})
    codebook = sp.pq_codebook;
    pq_codes = sp.pq_codes;
    lookuptable = zeros(sp.M, sp.Ks);
    for i in 1:sp.M
        C = @view codebook[i,:,:];
        q = @view query[ ((i-1)*sp.Ds + 1) : ((i-1)*sp.Ds + sp.Ds)];
        lookuptable[i,:] = C*q;
    end
    n = size(sp.pq_codes)[1];
    scores = zeros(n);
    for i = 1:n
        s_ = 0;
        for j = 1:sp.M
            s_ = s_ + lookuptable[j, pq_codes[i,j]];
        end
        scores[i] = s_;
    end

    # lookuptable = transpose(lookuptable) 
    # scores = sum(lookuptable[pq_codes],dims=2)
    # scores = reshape(scores, n)

    return scores
end

function search_neighbors(sp::SearchNeighbors_PQ, query::Array{Float64,1},topk::Int)
    scores = compute_scores(sp, query)
    ix = collect(1:length(scores));
    indices = maxk!(ix,scores,topk);
    return indices
end

function get_neighbors(sp::SearchNeighbors_PQ, queries::Array{Float64,2},topk::Int)
    n = size(queries)[1];
    neighbors_matrix = zeros(Int,n,topk)
    Threads.@threads for i in 1:n
    # for i in 1:n
        q = queries[i,:];
        neighbors_matrix[i,:] = search_neighbors(sp, q, topk);
    end
    return neighbors_matrix
end

# @benchmark 