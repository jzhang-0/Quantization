classdef SearchNeighbors_PQ
    
    properties
        M
        metric = "dot_product"
        Ks
        D
        pq_codebook
        pq_codes
        Ds
    end
    
    methods
        function obj = SearchNeighbors_PQ(M,Ks,D,pq_codebook,pq_codes)
            obj.M = M;
            obj.Ks = Ks;
            obj.D = D;
            obj.Ds = D/M;
            obj.pq_codebook = pq_codebook;
            obj.pq_codes = pq_codes;
        end
        
    end
        
        function score = compute_distance(obj,query)
%             pq_codebook = obj.pq_codebook;
%             codes = obj.pq_codes;
%             metric = obj.metric;
%             M = obj.M;
%             Ks = pbj.Ks;
%             Ds = obj.Ds;
            
            lookup_table = zeros(obj.M,obj.Ks);
            q = reshape(query,[obj.Ds,obj.M]);
            for i = 1:obj.M
                ci = obj.pq_codebook(i,:,:);
                cii = reshape(ci,[obj.Ks,obj.Ds]);
                lookup_table(i,:) = cii*q(:,i);
            end
            %%%%%%%%%%%%%%%%%%%%
            value = lookup_table(obj.pq_codes); % 耗时 80%
            score = sum(value,2);
            %%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%%
            % [n,~] = size(obj.pq_codes);
            % score = zeros(n, 1);
            % for i = 1:n
            %     s_ = 0;
            %     for j = 1:obj.M
            %         s_ = s_ + lookup_table(j,obj.pq_codes(i,j));
            %     end
            %     score(i) = s_;
            % end
            %%%%%%%%%%%%%%%%%%%%


        end

        function neighbors_matrix = neighbors(obj, queries, topk)
            [n,~] = size(queries);
            neighbors_matrix = zeros(n, topk, "int64");
            for i = 1:n
                q = queries(i,:);
                score = obj.compute_distance(q);
                neighbors_matrix(i,:) = obj.topk_indices(score,topk);                
            end
        end

    end
    methods(Static)
        function pp(a)
            fprintf("%d",a);
        end
        
        function b = topk_indices(score,topk)
            % score n维向量， topk 整数
            [~,b] = maxk(score,topk);
        end
end
end

