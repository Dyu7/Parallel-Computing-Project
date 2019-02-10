// #define parallel
#include "nn.cpp"
#include<bits/stdc++.h> 
#include "kernel.cu"
using namespace std;
void readData(vector<vector<double> > &in, vector<double> &out)
{
    ifstream ifile("iris.data");
    int num_entries;
    ifile>>num_entries;
    for(int i=1;i<=num_entries;i++)
    {
        vector<double> tmp(4);
        for(int j=0;j<4;j++) ifile>>tmp[j];
        in.push_back(tmp);
        int x;
        ifile>>x;
        out.push_back(x);
    }
}
int main()
{
    initiate();
    int hidden = 32, hidden2 = 20, iters = 10000, numsamples = 150;
    vector<vector<double> > testinput;
    vector<double> testoutput;
    readData(testinput,testoutput);
    int input = testinput[0].size();
    int output = 3;
    Matrix in(testinput);
    in = in.T();
    Matrix out(3,testoutput.size());
    for(int i=0;i<150;i++)
    {
        out[0][i] = out[1][i] = out[2][i] = 0;
        out[testoutput[i]][i] = 1;
    }
    NeuralNetwork n(input,output,0.05);
    //memory for cuda
    initialise_memory(max({input,output,numsamples,hidden2,hidden})+2,max({input,output,numsamples,hidden2,hidden})+5);
    
    n.addLayer(hidden,NeuralNetwork::SIGMOID); 
    // n.addLayer(hidden2,NeuralNetwork::RELU);
    // n.addLayer(hidden2,NeuralNetwork::NIL);
    n.outputActivation(NeuralNetwork::SIGMOID);
    cout<<"Before:-\n";
    vector<vector<double> > outputs = n.predict(in).T().to2DVector();
    int idx = 0;
    for(auto j:outputs)
    {
        double mx = 0, ans = -1;
        for(int i=0;i<j.size();i++)
        {
            if(mx<j[i])
            {
                mx = j[i];
                ans = i;
            }
        }
        cout<<"Predicted:"<<ans<<' '<<"Actual:"<<testoutput[idx++]<<'\n';
    }
    n.train(in,out,iters);
    cout<<"After:-\n";
    outputs = n.predict(in).T().to2DVector();
    idx = 0;
    int accuracy=0;
    for(auto j:outputs)
    {
        double mx = 0, ans = -1;
        for(int i=0;i<j.size();i++)
        {
            if(mx<j[i])
            {
                mx = j[i];
                ans = i;
            }
        }
        accuracy += (ans==testoutput[idx]);
        cout<<"Predicted:"<<ans<<' '<<"Actual:"<<testoutput[idx++]<<'\n';
    }
    accuracy*=100;
    accuracy/=numsamples;
    cout<<"Accuracy:"<<accuracy<<"%"<<endl;
    free_memory();
}