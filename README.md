# Parallel-Computing-Project


The report can be found [here](/Report.pdf).   

## Instructions to run  

In the [code](/code) directory, open a terminal window and run the following command(s):    
**Iris Dataset** :`nvcc -std=c++14 chknniris.cpp -lcublas && ./a.out`  
**MNIST Dataset** :`nvcc -std=c++14 chknnmnist.cpp -lcublas && ./a.out`  
