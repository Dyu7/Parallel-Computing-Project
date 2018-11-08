#pragma once
#include "matrix.cpp"
#include<vector>
#include<cmath>
#include<iostream>
using namespace std;
class NeuralNetwork{
    int inputsize,outputsize;
    double alpha;
    vector<int> layersizes;
    vector<double(*)(double)> activation;
    vector<double(*)(double)> diff;
    vector<Matrix> weights;
    vector<Matrix> biases;
    static double sigmoid(double x)
    {
        return 1/(1+exp(-x));
    }
    static double sigmoiddiff(double y) //y=sig(x)
    {
        return y*(1-y);
    }
    static double tanh(double x)
    {
        return ::tanh(x);
    }
    static double tanhdiff(double y) //y=tanh(x)
    {
        return 1-y*y;
    }
    static double relu(double x)
    {
        return x>0?x:0;
    }
    static double reludiff(double y) //y=relu(x)
    {
        return y==0?0:1;
    }
    static double nil(double x)
    {
        return x;
    }
    static double nildiff(double y)
    {
        return 1;
    }
    static const vector<double(*)(double)> act_func_pool;
    static const vector<double(*)(double)> diff_func_pool;
public:
    enum activation_function{
        SIGMOID=0,
        TANH=1,
        RELU=2,
        NIL=3
    };
    NeuralNetwork(int numInput, int numOutput, double learning_rate)
    {
        inputsize = numInput;
        outputsize = numOutput;
        alpha = learning_rate;
    }
    void addLayer(int size, enum activation_function type)
    {
        Matrix tmp;
        if(layersizes.empty())
        {
            tmp = Matrix(size,inputsize);
            tmp.randomize();
            weights.push_back(tmp);
        }
        else
        {
            tmp = Matrix(size,layersizes.back());
            tmp.randomize();
            weights.push_back(tmp);
        }
        tmp = Matrix(size,1);
        tmp.randomize();
        biases.push_back(tmp);
        layersizes.push_back(size);
        activation.push_back(act_func_pool[type]);
        diff.push_back(diff_func_pool[type]);
    }
    void outputActivation(enum activation_function type)
    {
        Matrix tmp;
        if(layersizes.empty())
        {
            tmp = Matrix(outputsize,inputsize);
            tmp.randomize();
            weights.push_back(tmp);
        }
        else
        {
            tmp = Matrix(outputsize,layersizes.back());
            tmp.randomize();
            weights.push_back(tmp);
        }
        tmp = Matrix(outputsize,1);
        tmp.randomize();
        biases.push_back(tmp);
        layersizes.push_back(outputsize);
        activation.push_back(act_func_pool[type]);
        diff.push_back(diff_func_pool[type]);
    }
    template<typename T> Matrix predict(const T &inputs)
    {
        Matrix tmp(inputs);
        int numlayers = layersizes.size();
        for(int i=0;i<numlayers;i++)
        {
            tmp = (weights[i]*tmp);
            tmp = coladd(tmp,biases[i]);
            tmp.activate(activation[i]);
        }
        return tmp;
    }
    template<typename T> void train(T &inputs, T &results, int iters=100000)
    {
        while(iters--)
        {
            Matrix targets(results);
            Matrix tmp(inputs);
            int numlayers = layersizes.size();
            vector<Matrix> layerdata;
            layerdata.push_back(Matrix(inputs));
            for(int i=0;i<numlayers;i++)
            {
                tmp = (weights[i]*tmp);
                tmp = coladd(tmp,biases[i]);
                tmp.activate(activation[i]);
                layerdata.push_back(tmp);
            }
            Matrix outputs = layerdata.back();
            Matrix output_errors = targets-outputs;
            for(int i=numlayers-1;i>=0;i--)
            {
                layerdata.pop_back();
                Matrix gradients = outputs;
                gradients.activate(diff[i]);
                Matrix data_T = layerdata.back().T();
                gradients = (alpha/gradients.m)*hadamard(gradients,output_errors);
                Matrix weights_delta = gradients*data_T;
                weights[i] = weights[i]+weights_delta;
                biases[i] = biases[i]+collapse(gradients);
                Matrix weights_T = weights[i].T();
                output_errors = weights_T*gradients;
                outputs = layerdata.back();
            }
        }
    }
};
const vector<double(*)(double)> NeuralNetwork::act_func_pool = {NeuralNetwork::sigmoid,NeuralNetwork::tanh,NeuralNetwork::relu,NeuralNetwork::nil};
const vector<double(*)(double)> NeuralNetwork::diff_func_pool = {NeuralNetwork::sigmoiddiff,NeuralNetwork::tanhdiff,NeuralNetwork::reludiff,NeuralNetwork::nildiff};
