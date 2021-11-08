using ArgParse

function args_pq(s)
    @add_arg_table s begin
        "--M"
            help = "M"
            arg_type = Int
            default = -1
        "--Ks"
            help = "Ks"
            arg_type = Int
            default = -1
        "--D"
            help = "D"
            arg_type = Int
            default = -1
        "--pq_codebook", "-b"
            help = "path"
            arg_type = String
            default = "-1"
        "--pq_codes","-c"
            help = "path"
            arg_type = String
            default = "-1"        

        "--metric","-m"
            help = "D"
            arg_type = String
            default = "-1"
        end
        return s
end

function args_data(s)
    @add_arg_table s begin

        "--queries","-q"
        help = "path"
        arg_type = String
        default = "-1"

        "--datan"
        help = "data name"
        arg_type = String
        default = "-1"

        "--tr100"
        help = "path"
        arg_type = String
        default = "-1"
    end
    return s    
end

function args_vq(s)
    @add_arg_table s begin
        "--vq_codebook"
        help = "path"
        arg_type = String
        default = "-1"

        "--vq_codes"
        help = "path"
        arg_type = String
        default = "-1"
    end
end

function args_recall(s)
    @add_arg_table s begin
        "--num_leaves_to_search"
        help = "num_leaves_to_search"
        arg_type = Int
        default = -1

        "--topk"
        help = "num_leaves_to_search"
        arg_type = Int
        default = -1

    end
end