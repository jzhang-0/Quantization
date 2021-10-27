#include <iostream>

int main(){
    long int a = 1ULL*10000;
    long int b = 0;
    float d=0;
    float c = 0.1;
    for(long int k=0; k<a;k++)
    {
        for (long int k2=0;k2<100000;k2++)
        {
        d = d + c;
        // d = d + c;
        }
    }
    std::cout << b << "\n";

    return 0;
}