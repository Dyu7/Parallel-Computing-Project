#pragma once
#include<vector>
#include<random>
#include<cassert>
#include<iostream>
#include "kernel.cu"
class Matrix{
    std::vector<std::vector<double> > data;
    int _n,_m;
public:
    const int &n = _n, &m = _m; 
    Matrix(int rows=0, int cols=0)
    {
        _n = rows;
        _m = cols;
        data.resize(n);
        for(int i=0;i<n;i++)
        {
            data[i].assign(m,0);
        }
    }
    Matrix(const std::vector<double> &v)
    {
        _n = v.size();
        _m = 1;
        data.resize(n);
        for(int i=0;i<n;i++)
        {
            data[i].resize(1);
            data[i][0] = v[i];
        }
    }
    Matrix(const std::vector<std::vector<double> > &v)
    {
        _n = v.size();
        assert(v.size()>=1);
        _m = v[0].size();
        for(int i=0;i<n;i++)
            assert(v[i].size()==m);
        data = v;
    }
    Matrix(const Matrix &m)
    {
        _n = m.n;
        _m = m.m;
        data = m.data;
    }
    Matrix operator =(const Matrix m)
    {
        data = m.data;
        _n = m.n;
        _m = m.m;
        return *this; 
    }
    Matrix operator =(const std::vector<double> &v)
    {
        _n = v.size();
        _m = 1;
        data.resize(n);
        for(int i=0;i<n;i++)
        {
            data[i].resize(1);
            data[i][0] = v[i];
        }
        return *this;
    }
    std::vector<double>& operator [](const int &idx)
    {
        assert(idx>=0&&idx<n);
        return data[idx];
    }
    const std::vector<double>& operator [](const int &idx) const
    {
        assert(idx>=0&&idx<n);
        return data[idx];
    }
    template<typename T> Matrix friend operator *(const T &val,const Matrix &m)
    {
        Matrix ans(m.n,m.m);
        for(int i=0;i<m.n;i++)
            for(int j=0;j<m.m;j++)
            {
                ans[i][j] = val*m[i][j];
            }
        return ans;
    }
    template<typename T> Matrix friend operator *(const Matrix &m,const T &val)
    {
        Matrix ans(m.n,m.m);
        for(int i=0;i<m.n;i++)
            for(int j=0;j<m.m;j++)
            {
                ans[i][j] = val*m[i][j];
            }
        return ans;
    }
    const Matrix seqmult(const Matrix &b)
    {
        Matrix &a = *this;
        assert(a.m==b.n);
        Matrix ans(a.n,b.m);
        int t = a.m;
        for(int i=0;i<a.n;i++)
        {
            for(int j=0;j<b.m;j++)
            {
                for(int k=0;k<t;k++)
                {
                    ans[i][j] += a[i][k]*b[k][j];
                }
            }
        }
        return ans;
    }
    const Matrix operator * (const Matrix &b)
    {
        // return seqmult(b);
        Matrix &a = *this;
        assert(a.m==b.n);
        Matrix lul = par_mat_mult(a,b);
        return lul;
    }
    template<typename T> Matrix friend operator +(const T &val,const Matrix &m)
    {
        Matrix ans(m.n,m.m);
        for(int i=0;i<m.n;i++)
            for(int j=0;j<m.m;j++)
            {
                ans[i][j] = val+m[i][j];
            }
        return ans;
    }
    template<typename T> Matrix friend operator +(const Matrix &m,const T &val)
    {
        Matrix ans(m.n,m.m);
        for(int i=0;i<m.n;i++)
            for(int j=0;j<m.m;j++)
            {
                ans[i][j] = val+m[i][j];
            }
        return ans;
    }
    const Matrix operator + (const Matrix &b)
    {
        Matrix &a = *this;
        assert(a.n==b.n&&a.m==b.m);
        Matrix ans(a.n,a.m);
        for(int i=0;i<ans.n;i++)
        {
            for(int j=0;j<ans.m;j++)
            {
                ans[i][j] = a[i][j]+b[i][j];
            }
        }
        return ans;
    }
    template<typename T> Matrix friend operator -(const T &val,const Matrix &m)
    {
        Matrix ans(m.n,m.m);
        for(int i=0;i<m.n;i++)
            for(int j=0;j<m.m;j++)
            {
                ans[i][j] = val-m[i][j];
            }
        return ans;
    }
    template<typename T> Matrix friend operator -(const Matrix &m,const T &val)
    {
        Matrix ans(m.n,m.m);
        for(int i=0;i<m.n;i++)
            for(int j=0;j<m.m;j++)
            {
                ans[i][j] = m[i][j]-val;
            }
        return ans;
    }
    const Matrix operator - (const Matrix &b)
    {
        Matrix &a = *this;
        assert(a.n==b.n&&a.m==b.m);
        Matrix ans(a.n,a.m);
        for(int i=0;i<ans.n;i++)
        {
            for(int j=0;j<ans.m;j++)
            {
                ans[i][j] = a[i][j]-b[i][j];
            }
        }
        return ans;
    }
    Matrix friend operator - (const Matrix &a)
    {
        Matrix ans(a.n,a.m);
        for(int i=0;i<ans.n;i++)
        {
            for(int j=0;j<ans.m;j++)
            {
                ans[i][j] = -a[i][j];
            }
        }
        return ans;
    }
    template<typename T> Matrix friend operator /(const Matrix &m,const T &val)
    {
        Matrix ans(m.n,m.m);
        for(int i=0;i<m.n;i++)
            for(int j=0;j<m.m;j++)
            {
                ans[i][j] = m[i][j]/val;
            }
        return ans;
    }
    Matrix friend hadamard(const Matrix &a, const Matrix &b)
    {
        assert(a.n==b.n&&a.m==b.m);
        Matrix ans = a;
        for(int i=0;i<ans.n;i++)
            for(int j=0;j<ans.m;j++)
                ans[i][j]*=b[i][j];
        return ans;
    }
    Matrix T()
    {
        Matrix ans(m,n);
        for(int i=0;i<ans.n;i++)
            for(int j=0;j<ans.m;j++)
            {
                ans[i][j] = data[j][i];
            }
        return ans;
    }
    void randomize()
    {
        assert(n>0&&m>0);
        std::uniform_real_distribution<> dis(-1,1);
        std::mt19937 gen(time(0));
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
            {
                data[i][j] = dis(gen);
            }
    }
    std::vector<double> toVector()
    {
        assert(m==1);
        return T()[0];
    }
    void activate(double(*func)(double))
    {
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
            {
                data[i][j] = func(data[i][j]);
            }
    }
    void print(){
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                std::cout<<data[i][j]<<" ";
            }
            std::cout<<std::endl;
        }
    }
    Matrix friend coladd(Matrix &a, Matrix &b)
    {
        Matrix ans = a;
        assert(b.m==1);
        for(int i=0;i<a.n;i++)
            for(int j=0;j<a.m;j++)
            {
                ans[i][j]+=b[i][0];
            }
        return ans;
    }
    Matrix friend collapse(Matrix &a)
    {
        Matrix ans(a.n,1);
        for(int i=0;i<a.n;i++)
            for(int j=0;j<a.m;j++)
            {
                ans[i][0]+=a[i][j];
            }
        return ans;
    }
    std::vector<std::vector<double> > to2DVector()
    {
        return data;
    }
};