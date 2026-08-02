#include <mnist/load_data.hpp>

#include <cstdint>
#include <filesystem>
#include <ios>
#include <stdexcept>
#include <zlib.h>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace mnist
{
    static uint32_t read_big_endian(gzFile file)
    {
    	uint8_t bytes[4]{};
    	if (gzread(file, bytes, 4) != 4)
    		throw std::ios_base::failure("Failed to read header");
    
    	return
    		(static_cast<uint32_t>(bytes[0]) << 24) |
    		(static_cast<uint32_t>(bytes[1]) << 16) |
    		(static_cast<uint32_t>(bytes[2]) <<  8) |
    		(static_cast<uint32_t>(bytes[3])      ) ;
    }
    
    static nnf::Vector<float> load_mnist_images_gz(const std::filesystem::path &filepath)
    {
    	gzFile file = gzopen(filepath.string().c_str(), "rb");
    	if (!file)
    		throw std::ios_base::failure("Failed to open file");
    
    	uint32_t magic = read_big_endian(file);
    	uint32_t count = read_big_endian(file);
    	uint32_t rows  = read_big_endian(file);
    	uint32_t cols  = read_big_endian(file);
    
    	if (magic != 2051)
    		throw std::invalid_argument("File is not an MNIST images gz file");
    
    	auto data = nnf::Vector<float>(count * rows * cols);
    
    	uint8_t pixel{};
    	for (float &value : data)
    	{
    		if (gzread(file, &pixel, 1) != 1)
    			throw std::ios_base::failure("Failed to read data");
    		value = static_cast<float>(pixel) / 255.f;
    	}
    
    	return data;
    }
    
    static nnf::Vector<float> load_mnist_labels_gz(const std::filesystem::path &filepath)
    {
    	gzFile file = gzopen(filepath.string().c_str(), "rb");
    	if (!file)
    		throw std::ios_base::failure("Failed to open file");
    
    	uint32_t magic = read_big_endian(file);
    	uint32_t count = read_big_endian(file);
    
    	if (magic != 2049)
    		throw std::invalid_argument("File is not an MNIST labels gz file");
    
    	auto data = nnf::Vector<float>(count * 10);
    
    	uint8_t label_class_number{};
    	for (nnf::usize i = 0; i < count; ++i)
    	{
    		if (gzread(file, &label_class_number, 1) != 1)
    			throw std::ios_base::failure("Failed to read data");
    		if (label_class_number >= 10)
    			throw std::runtime_error("Label is not a one-digit integer");
    		data[i * 10 + label_class_number] = 1.f;
    	}
    
    	return data;
    }
    
    MNISTData load_mnist(const std::filesystem::path &mnist_directory)
    {
    	MNISTData result{};
    	result.train_data_60k.set_raw_data(load_mnist_images_gz(mnist_directory / "train-images-idx3-ubyte.gz"));
    	result.train_labels_60k.set_raw_data(load_mnist_labels_gz(mnist_directory / "train-labels-idx1-ubyte.gz"));
    	result.test_data_10k.set_raw_data(load_mnist_images_gz(mnist_directory / "t10k-images-idx3-ubyte.gz"));
    	result.test_labels_10k.set_raw_data(load_mnist_labels_gz(mnist_directory / "t10k-labels-idx1-ubyte.gz"));
    	return result;
    }
}
