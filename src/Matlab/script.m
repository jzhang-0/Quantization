addpath("src/Neighbors_matlab/npy-matlab/npy-matlab")
codebook = readNPY('codebook.npy');
code = readNPY('code_M8_K16_sample_num10000.npy');
code = code + 1;
queries = readNPY("queries_32D.npy");

% profile on;
tic
sp = SearchNeighbors_PQ(8,16,32,codebook,code);
neightbors = sp.neighbors(queries,512);
toc
% profile viewer;


% 向量化处理，每次申请内存 总耗时50s 
% 