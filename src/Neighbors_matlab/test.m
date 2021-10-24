addpath("C:\\Users\\14027\\matlab_ex\\npy-matlab\\npy-matlab")
codebook = readNPY('codebook.npy');
code = readNPY('code_M8_K16_sample_num10000.npy');
code = code + 1;
queries = readNPY("queries_32D.npy");
query = queries(1,:);
 
reshape