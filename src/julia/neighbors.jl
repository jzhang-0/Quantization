
export 
    AbstractSN,maxk!
abstract type AbstractSN end

function maxk!(a, k::Int; initialized=false)
    # k:topk
    ix = collect(1:length(a))
    partialsortperm!(ix, a, 1:k, rev=true, initialized=initialized)
    # @views collect(zip(ix[1:k], a[ix[1:k]]))
    @views ix[1:k]
end

# @benchmark 


module SN_pq
    using Main

    export 
        SearchNeighbors_PQ,
        compute_table,
        compute_scores_

    struct SearchNeighbors_PQ <: AbstractSN
        M::Integer
        Ks::Integer
        D::Integer
        pq_codebook::Array{Float32, 3} 
        pq_codes::Array{Int64, 2} 
        metric::String
        Ds::Integer
        function SearchNeighbors_PQ(M, Ks, D, pq_codebook, pq_codes, metric)
            Ds = Int64(D/M) 
            new(M, Ks, D, pq_codebook, pq_codes, metric, Ds)            
        end
    end

    # typeof(x)
    # x = SearchNeighbors_PQ(M=8, Ks=16, D=32, metric = "dot_product")

    function exinit(sp::AbstractSN)
        for i in 1:sp.M
            sp.pq_codes[:,i] .+= sp.Ks*(i-1);
        end    
    end

    function compute_table(codebook::Array{Float32,3}, query::Array{Float64,1})
        M, Ks, Ds = size(codebook) 
        lookuptable = zeros(M, Ks);
        for i in 1:M
            C = @view codebook[i,:,:];
            q = @view query[ ((i-1)*Ds + 1) : ((i-1)*Ds + Ds)];
            lookuptable[i,:] = C*q;
        end
        return lookuptable
    end

    function compute_scores_(lookuptable::Array{Float64,2}, pq_codes::Array{Int,2})
        n, M = size(pq_codes);
        scores = zeros(n);
        for i = 1:n
            s_ = 0;
            for j = 1:M
                @inbounds s_ = s_ + lookuptable[j, pq_codes[i,j]];
            end
            scores[i] = s_;
        end
        return scores    
    end

    function compute_scores(sp::AbstractSN, query::Array{Float64,1})
        codebook = sp.pq_codebook;
        pq_codes = sp.pq_codes;

        lookuptable = compute_table(codebook,query);    
        scores = compute_scores_(lookuptable,pq_codes);

        return scores
    end

    function search_neighbors(sp::AbstractSN, query::Array{Float64,1},topk::Int)
        scores = compute_scores(sp, query)
        indices = maxk!(scores, topk);
        return indices
    end

    function get_neighbors(sp::AbstractSN, queries::Array{Float64,2},topk::Int)
        n = size(queries)[1];
        neighbors_matrix = zeros(Int,n,topk)
        Threads.@threads for i in 1:n
        # for i in 1:n
            q = queries[i,:];
            neighbors_matrix[i,:] = search_neighbors(sp, q, topk);
        end
        return neighbors_matrix
    end

end

module SN_ivf
    using Main
    export 
        coarse_search,
        SearchNeighbors_IVF


    struct SearchNeighbors_IVF <: AbstractSN
        k_v::Integer 
        D::Integer
        vq_codebook::Array{<:AbstractFloat, 2}
        vq_codes::Array{<:Integer, 1} 
        metric::String
        
        function SearchNeighbors_IVF(vq_codebook, vq_codes, metric)
            k_v,D = size(vq_codebook)
            new(k_v, D, vq_codebook, vq_codes, metric)
        end
    end    

    function coarse_search(si::AbstractSN, query::Vector, num_centroid_tosearch::Integer)
        score = si.vq_codebook*query # (k_v,)
        index = maxk!(score, num_centroid_tosearch)

        score_v = score[index]
        return index, score_v
    end   

end

module SN_pqivf
    export SearchNeighbors_PQIVF

    using Main
    using ..SN_pq
    using ..SN_ivf

    struct SearchNeighbors_PQIVF <: AbstractSN
        M::Int8
        Ks::Int8
        D::Int16
        Ds::Int16 
        pq_codebook::Array{<:AbstractFloat, 3} 
        pq_codes::Array{<:Integer, 2} 

        k_v::Int64
        vq_codebook::Array{<:AbstractFloat, 2}
        vq_codes::Array{<:Integer, 1} 
        metric::String
        id_index_table::Dict{Int64,Vector{Int64}}

        function SearchNeighbors_PQIVF(M, Ks, pq_codebook, pq_codes, vq_codebook, vq_codes, metric)
            si = SearchNeighbors_IVF(vq_codebook, vq_codes, metric)
            k_v, D, vq_codebook, vq_codes, metric = si.k_v, si.D, si.vq_codebook, si.vq_codes, si.metric
            
            sp = SearchNeighbors_PQ(M, Ks, D, pq_codebook, pq_codes, metric)
            M,Ks,Ds,pq_codebook,pq_codes = sp.M, sp.Ks, sp.Ds, sp.pq_codebook, sp.pq_codes
            
            dict = Dict()
            for i in 1:k_v
                dict[i] = []
            end

            n = length(vq_codes)
            for i in 1:n
                codei = vq_codes[i]
                append!(dict[codei],i)
            end
            
            new(M, Ks, D, Ds, pq_codebook, pq_codes, k_v, vq_codebook, vq_codes, metric, dict)
        end 
    end

    function id_indexed(spi::AbstractSN, index::SubArray{Int64, 1}, scores_v::Vector{Float64})
        ids = [] # Array{Int64,1}
        scores_vq = []
        for (i,s) in zip(index,scores_v)
            id_i = spi.id_index_table[i]
            ids = [ids;id_i]
            len = length(id_i)
            scores_vq = [scores_vq;repeat([s],len)]
        end
        ids = convert(Array{Int64,1}, ids)
        scores_vq = convert(Array{Float64,1}, scores_vq)
        return ids, scores_vq
    end

    function search_neighbors(spi::AbstractSN, q::Vector{<:AbstractFloat}, num_centroid_tosearch::Int, topk::Int)
        index,scores_v = coarse_search(spi, q, num_centroid_tosearch)
        ids,scores_vq = id_indexed(spi, index, scores_v)
        lookuptable = compute_table(spi.pq_codebook, q) # (M,Ks)
        pq_codes_ids = spi.pq_codes[ids,:]
        scores_pq = compute_scores_(lookuptable,pq_codes_ids)
        scores = scores_vq + scores_pq
        
        len_s = length(scores)
        if topk > len_s
            i_ = maxk!(scores, len_s)
            ids_ = ids[i_]
            ids_ = [ids_;repeat([0],topk - len_s)] 
        else
            i_ = maxk!(scores, topk)
            ids_ = ids[i_]        
        end
        return ids_
    end

    function get_neighbors(spi::AbstractSN, queries::Array{Float64,2}, num_centroid_tosearch::Int,  topk::Int)
        n = size(queries)[1];
        neighbors_matrix = zeros(Int,n,topk)
        Threads.@threads for i in 1:n
            q = queries[i,:];
            neighbors_matrix[i,:] = search_neighbors(spi, q, num_centroid_tosearch, topk);
        end
        return neighbors_matrix
    end 


end