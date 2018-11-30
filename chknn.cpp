#include "nn.cpp"
#include "readMNIST.cpp"
#include<bits/stdc++.h>
using namespace std;
int main()
{
    initiate();
    int hidden = 32, hidden2 = 20, iters = 1000, numsamples = 20;
    vector<vector<double> > testinput;
    read_Mnist(testinput);
    vector<double> testoutput;
    read_Mnist_Label(testoutput);
    testinput.resize(numsamples);
    testoutput.resize(numsamples);
    int input = testinput[0].size();
    int output = 10;
    Matrix in(testinput);
    in = in.T();
    Matrix out(10,testoutput.size());
    for(int i=0;i<testoutput.size();i++)
    {
        out[testoutput[i]][i] = 1;
    }
    // out = out.T();
    NeuralNetwork n(input,output,1);
    n.addLayer(hidden,NeuralNetwork::SIGMOID); 
    // n.addLayer(hidden2,NeuralNetwork::RELU);
    // n.addLayer(hidden2,NeuralNetwork::NIL);
    n.outputActivation(NeuralNetwork::SIGMOID);
    cout<<"Before:-\n";
    vector<vector<double> > outputs = n.predict(in).T().to2DVector();
    int idx = 0;
    for(auto j:outputs)
    {
        int k = 10;
        double mx = 0, ans = -1;
        for(int i=0;i<10;i++)
            if(j[i]>mx)
            {
                mx = j[i];
                ans = i;
            }
        cout<<"Predicted:"<<ans<<' '<<"Actual:"<<testoutput[idx++]<<'\n';
    }
    n.train(in,out,iters);
    cout<<"After:-\n";
    outputs = n.predict(in).T().to2DVector();
    idx = 0;
    for(auto j:outputs)
    {
        int k = 10;
        double mx = 0, ans = -1;
        for(int i=0;i<10;i++)
            if(j[i]>mx)
            {
                mx = j[i];
                ans = i;
            }
        cout<<"Predicted:"<<ans<<' '<<"Actual:"<<testoutput[idx++]<<'\n';
    }
}