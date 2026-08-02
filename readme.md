# Neural network framwork in C++

This project aims to build and train neural networks and convolutional neural networks from scratch in C++ with no external maths or machine learning libraries.

The main component is the library `nnf` which implements tensors, automatic differentiation, a sequential model and various layers and loss functions. This comes with some tests and benchmarks.

There is also an example `mnist_example` which uses the library to train a neural network on the MNIST Digits dataset and has a simple GUI app allowing users to draw digits and see the output of the model.

The project is still in development but is at a stage where the MNIST model can train and run correctly.

### Build from source

Only tested on Windows.

Requires C++ 20 and CMake 3.23.

```
git clone "https://github.com/Lecdi/nnf"
cd nnf
mkdir build
cd build
cmake ..
```

The generated project provides a library `nnf` along with unit tests and benchmarks, and executables `mnist_train_full` and `mnist_app`. (You must train the model before attempting to build the app. Since this takes a long time, it is best to do it in release mode - if you want quicker training, you can instead use `mnist_train_partial` and customise the constants in the code to change the amount of training data used, but the model will be less accurate.)
