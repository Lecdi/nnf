#pragma once

#include <filesystem>
#include <nnf/tensor/tensor.hpp>

namespace mnist
{
    struct MNISTData
    {
    	nnf::Tensor train_data_60k{ {60000, 784} };
    	nnf::Tensor train_labels_60k{ {60000, 10} };
    	nnf::Tensor test_data_10k{ {10000, 784} };
    	nnf::Tensor test_labels_10k{ {10000, 10} };
    };
    
    MNISTData load_mnist(const std::filesystem::path &mnist_directory);
}
