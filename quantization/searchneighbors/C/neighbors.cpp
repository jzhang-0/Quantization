#include <iostream>
#include "neighbors.h"
using namespace std;

class SearchNeighbors
{
public:
    std::string metric;
    SearchNeighbors(std::string metric);
    int sort_topk();
    // ~SearchNeighbors();
};

// SearchNeighbors::~SearchNeighbors()
// {
// }
SearchNeighbors::SearchNeighbors(std::string metric)
            // :metric{metric} // extended initializer lists only available with -std=c++11 or -std=gnu++11
        {
            SearchNeighbors::metric = metric;
            if (metric != "dot_product" & metric != "l2_distance"){
                throw "metric optional:l2_distance or dot_product";
            }
            cout<<1<<"\n";
        };

int SearchNeighbors::sort_topk(){

    if (metric == "dot_product"){
        cout << "dot_product" << "\n";
        // pass
    };
    return 0;
};


class SearchNeighbors_PQ: public SearchNeighbors{
    public:
        int M;
        int Ks;
        int D;
        int Ds;
        SearchNeighbors_PQ(std::string metric, 
                            int M, 
                            int Ks,
                            int D);
};

SearchNeighbors_PQ::SearchNeighbors_PQ(std::string metric, 
                            int M, 
                            int Ks,
                            int D)
            :SearchNeighbors(metric)
        {
            SearchNeighbors_PQ::M = M;
            SearchNeighbors_PQ::D = D;
            SearchNeighbors_PQ::Ks = Ks;
            SearchNeighbors_PQ::Ds = D/M;
        };

int main(){
    // SearchNeighbors a("dot_product");
    SearchNeighbors_PQ a("dot_product",8,16,32);
    cout << a.metric << "\n";
    cout << a.Ds << "\n";
    a.sort_topk();
    return 0;
}