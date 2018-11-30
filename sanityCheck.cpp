#include<iostream>
#include "matrix.cpp"
using namespace std;
int main()
{
    initiate();
    int n=300,m=45;
    Matrix A(n,m),B(m,n);
    A.randomize();
    B.randomize();

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t start_seq;
    cudaEventCreate(&start_seq);

    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEvent_t stop_seq;
    cudaEventCreate(&stop_seq);

    cudaEventRecord(start, NULL);

    Matrix M = A*B;

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    cout<<"Parallel mult time:"<<msecTotal<<endl;
    // return 0;

    Matrix ch(n,n);

    cudaEventRecord(start_seq, NULL);
    double te=0;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            ch[i][j]=0;
            for(int k=0;k<m;k++){
                ch[i][j]+=A[i][k]*B[k][j];
            }
            double error=abs(ch[i][j]-M[i][j]);
            te+=error;
            // cout<<error<<endl;
            if(error>1e-8){
                cout<<"Error found at: "<<i<<" "<<j<<endl;
                assert(0);
            }
        }
    }
    cudaEventRecord(stop_seq, NULL);
    cudaEventSynchronize(stop_seq);
    float msecTotal_seq = 0.0f;
    cudaEventElapsedTime(&msecTotal_seq, start_seq, stop_seq);
    cout<<"Sequential mult time:"<<msecTotal_seq<<endl;    


    cout<<"Elements of both matrices are within tolerance limit! Success!"<<endl;

    cout<<te<<endl;
    // ch.print();
    // M.print();
}