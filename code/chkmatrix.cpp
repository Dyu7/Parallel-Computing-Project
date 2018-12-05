#include<iostream>
#include "matrix.cpp"
using namespace std;
int main()
{
    int n,m;
    cin>>n>>m;
    Matrix A(n,m),B(n,m);
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            cin>>A[i][j];
        }
    }
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            cin>>B[i][j];
        }
    }
    vector<double> v = {1,2,3,4,5};
    Matrix M = A*B;
    M = M.T();
    M.randomize();
    // M = X;
    // M = M/2;
    M = v;
    v = M.toVector();
    // M = M.T();
    for(int i=0;i<M.n;i++)
    {
        for(int j=0;j<M.m;j++)
        {
            int x = M[i][j];
            cout<<v[i]<<" ";
        }
        cout<<endl;
    }
}