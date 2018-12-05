- matrix.cpp
    - designed a modular operator overloaded matrix class for C++, which was used in nn.cpp
    Variables
        - data -> stores the matrix data
        - _n -> stores no of rows
        - _m -> stores no of cols
    Functions
        - operators [],+.-,*,/,= -> overloaded all mathematical operators to work for matrix-matrix as well as matrix-scalar operations.
        - seqmult -> Performs sequential matrix multiplication.
        - hadamard -> elementwise multiplication of matrices, c[i][j] = a[i][j]*b[i][j]
        - randomize() -> initialize values of matrices to random values
        - toVector() -> converts a column matrix to C++ STL vector and returns
        - activate(func) -> applies the passed function on every element of the matrix, a[i][j] = func(a[i][j])
        - coladd(a,b) -> add a column vector(b) to every column of matrix a
        - collapse(a) -> breaks matrix to column vector, and returns sum of all vectors
        - to2DVector() -> converts the matrix to C++ STL nested vector format and returns

- nn.cpp
    - The neural network library, fully modular, multiple activation function as well as multi-hidden layer support
    Variables
        - inputsize, outputsize -> input and output layer sizes respectively
        - alpha -> learning rate of gradient descent
        - activation -> stores activation function of each layer
        - diff -> stores the corresponding differention function for each layer
        - weights -> layer transition weights for each layer transition
        - biases -> layer transition bias for each layer transition
    Functions
        - addLayer(size,type) -> adds a layer to the neural network model of the given size and activation function type
        - outputActivation(type) -> sets output layer activation function
        - predict(input) -> returns the predict output for the given inputs in matrix form
        - train(input,results,iters) -> Performs batch training using gradient descent for iters no. of times, trying to fit the output to results
        - sigmoid, tanh, relu, nil -> activation functions
        - sigmoiddiff, tanhdiff, relu, nildiff -> differention of activation functions

- chkmatrix.cpp
    A small test code to verify that all matrix functions are working properly

- chknnmnist.cpp
    Test code that verifies performance of the neural network on the MNIST dataset

- chknniris.cpp
    Test code that verifies performance of the neural network on the IRIS dataset

- readMNIST.cpp
    CPP code to read binary format data of MNIST dataset