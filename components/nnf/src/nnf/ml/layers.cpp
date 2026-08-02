#include <nnf/ml/layers.hpp>

#include <memory>
#include <fstream>
#include <utility>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/autodiff/base_operations.hpp>
#include <nnf/autodiff/unary_scalar_operations.hpp>
#include <nnf/io/serialization.hpp>
#include <nnf/ml/param_initializers.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

nnf::ml::FlattenLayer &nnf::ml::FlattenLayer::set_prev_value(autodiff::Value prev_value)
{
	out_value = prev_value.flattened();
	return *this;
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::flatten_layer()
{
	return std::make_unique<FlattenLayer>();
}

nnf::ml::ReshapeLayer::ReshapeLayer(VectorInput<usize> new_dims)
	: new_dims_{ std::move(new_dims.values) }
{
}

nnf::ml::ReshapeLayer &nnf::ml::ReshapeLayer::set_prev_value(autodiff::Value prev_value)
{
	out_value = prev_value.reshaped(new_dims_);
	return *this;
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::reshape_layer(VectorInput<usize> new_dims)
{
	return std::make_unique<ReshapeLayer>(std::move(new_dims));
}

nnf::ml::SoftmaxLayer &nnf::ml::SoftmaxLayer::set_prev_value(autodiff::Value prev_value)
{
	out_value = prev_value.softmax();
	return *this;
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::softmax_layer()
{
	return std::make_unique<SoftmaxLayer>();
}

nnf::ml::MaxPool2DLayer::MaxPool2DLayer(ssize h, ssize w)
	: h_{ h }, w_{ w }
{
}

nnf::ml::MaxPool2DLayer &nnf::ml::MaxPool2DLayer::set_prev_value(autodiff::Value prev_value)
{
	out_value = prev_value.max_pool_2d(h_, w_);
	return *this;
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::max_pool_2d_layer(ssize h, ssize w)
{
	return std::make_unique<MaxPool2DLayer>(h, w);
}

nnf::ml::DenseLayer::DenseLayer(
	usize in_elems,
	usize out_elems,
	std::unique_ptr<autodiff::UnaryScalarOperator> activation,
	std::unique_ptr<ParamInitializer> weights_initializer
)
	: in_elems_{ in_elems }, out_elems_{ out_elems }, activation_{ std::move(activation) },
	params_weights_{ {out_elems, in_elems} }, grad_weights_{ {out_elems, in_elems} }, weights_{ {out_elems, in_elems} },
	params_biases_{ {out_elems} }, grad_biases_{ {out_elems} }, biases_{ {out_elems} },
	weights_initializer_{ std::move(weights_initializer) }
{
}

nnf::ml::DenseLayer &nnf::ml::DenseLayer::init_params()
{
	(*weights_initializer_)(params_weights_);
	params_biases_.make_zero();

	weights_.input(params_weights_);
	biases_.input(params_biases_);

	grad_weights_.make_zero();
	grad_biases_.make_zero();

	return *this;
}

nnf::ml::DenseLayer &nnf::ml::DenseLayer::read_params_from(std::ifstream &stream)
{
	Vector<float> weights_data;
	io::read_floats(weights_data, params_weights_.num_elems(), stream);
	params_weights_.set_from_contiguous_data(std::move(weights_data));

	Vector<float> biases_data;
	io::read_floats(biases_data, params_biases_.num_elems(), stream);
	params_biases_.set_from_contiguous_data(std::move(biases_data));

	weights_.input(params_weights_);
	biases_.input(params_biases_);

	grad_weights_.make_zero();
	grad_biases_.make_zero();

	return *this;
}

nnf::ml::DenseLayer &nnf::ml::DenseLayer::write_params_to(std::ofstream &stream)
{
	io::write_floats(params_weights_.get_contiguous_data(), stream);
	io::write_floats(params_biases_.get_contiguous_data(), stream);
	return *this;
}

nnf::ml::DenseLayer &nnf::ml::DenseLayer::set_prev_value(autodiff::Value prev_value)
{
	out_value = (*activation_)(matvecmul(weights_, prev_value) - biases_);
	return *this;
}

nnf::ml::DenseLayer &nnf::ml::DenseLayer::accumulate_grad()
{
	grad_weights_ += weights_.grad();
	grad_biases_ += biases_.grad();
	return *this;
}

nnf::ml::DenseLayer &nnf::ml::DenseLayer::update_params(float scale_by)
{
	params_weights_ -= grad_weights_ * scale_by;
	params_biases_ -= grad_biases_ * scale_by;
	weights_.input(params_weights_);
	biases_.input(params_biases_);
	return *this;
}

nnf::ml::DenseLayer &nnf::ml::DenseLayer::reset_grad()
{
	grad_weights_.make_zero();
	grad_biases_.make_zero();
	return *this;
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::dense_relu_layer(usize in_elems, usize out_elems)
{
	return std::make_unique<DenseLayer>(
		in_elems, out_elems,
		autodiff::relu_operator(),
		he_kaiming_initializer(in_elems)
	);
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::dense_sigmoid_layer(usize in_elems, usize out_elems)
{
	return std::make_unique<DenseLayer>(
		in_elems, out_elems,
		autodiff::sigmoid_operator(),
		xavier_glorot_initializer(in_elems, out_elems)
	);
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::dense_tanh_layer(usize in_elems, usize out_elems)
{
	return std::make_unique<DenseLayer>(
		in_elems, out_elems,
		autodiff::tanh_operator(),
		xavier_glorot_initializer(in_elems, out_elems)
	);
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::dense_no_activation_layer(usize in_elems, usize out_elems)
{
	return std::make_unique<DenseLayer>(
		in_elems, out_elems,
		autodiff::identity_operator(),
		xavier_glorot_initializer(in_elems, out_elems)
	);
}

nnf::ml::Conv2DLayer::Conv2DLayer(
	usize in_h,
	usize in_w,
	usize num_kernels,
	usize kernel_h,
	usize kernel_w,
	usize stride_h,
	usize stride_w,
	std::unique_ptr<autodiff::UnaryScalarOperator> activation,
	std::unique_ptr<ParamInitializer> kernel_weights_initializer
)
	: in_h_{ in_h }, in_w_{ in_w }, kernel_h_{ kernel_h }, kernel_w_{ kernel_w }, stride_h_{ stride_h },
	stride_w_{ stride_w }, num_kernels_{ num_kernels },
	out_dims_{ Tensor::get_conv2d_dims(in_h, in_w, num_kernels, kernel_h, kernel_w, stride_h, stride_w) },
	params_kernel_weights_{ {num_kernels, kernel_h, kernel_w} },
	grad_kernel_weights_{ {num_kernels, kernel_h, kernel_w} }, kernel_weights_{ {num_kernels, kernel_h, kernel_w} },
	params_biases_{ {out_dims_} }, grad_biases_{ {out_dims_} }, biases_{ {out_dims_} },
	activation_{ std::move(activation) }, kernel_weights_initializer_{ std::move(kernel_weights_initializer) }
{
}

nnf::ml::Conv2DLayer &nnf::ml::Conv2DLayer::init_params()
{
	(*kernel_weights_initializer_)(params_kernel_weights_);
	params_biases_.make_zero();

	kernel_weights_.input(params_kernel_weights_);
	biases_.input(params_biases_);

	grad_kernel_weights_.make_zero();
	grad_biases_.make_zero();

	return *this;
}

nnf::ml::Conv2DLayer &nnf::ml::Conv2DLayer::read_params_from(std::ifstream &stream)
{
	Vector<float> kernel_weights_data;
	io::read_floats(kernel_weights_data, params_kernel_weights_.num_elems(), stream);
	params_kernel_weights_.set_from_contiguous_data(std::move(kernel_weights_data));

	Vector<float> biases_data;
	io::read_floats(biases_data, params_biases_.num_elems(), stream);
	params_biases_.set_from_contiguous_data(std::move(biases_data));

	kernel_weights_.input(params_kernel_weights_);
	biases_.input(params_biases_);

	grad_kernel_weights_.make_zero();
	grad_biases_.make_zero();

	return *this;
}

nnf::ml::Conv2DLayer &nnf::ml::Conv2DLayer::write_params_to(std::ofstream &stream)
{
	io::write_floats(params_kernel_weights_.get_contiguous_data(), stream);
	io::write_floats(params_biases_.get_contiguous_data(), stream);
	return *this;
}

nnf::ml::Conv2DLayer &nnf::ml::Conv2DLayer::set_prev_value(autodiff::Value prev_value)
{
	out_value = (*activation_)(prev_value.conv2d(kernel_weights_, stride_h_, stride_w_) - biases_);
	return *this;
}

nnf::ml::Conv2DLayer &nnf::ml::Conv2DLayer::accumulate_grad()
{
	grad_kernel_weights_ += kernel_weights_.grad();
	grad_biases_ += biases_.grad();
	return *this;
}

nnf::ml::Conv2DLayer &nnf::ml::Conv2DLayer::update_params(float scale_by)
{
	params_kernel_weights_ -= grad_kernel_weights_ * scale_by;
	params_biases_ -= grad_biases_ * scale_by;
	kernel_weights_.input(params_kernel_weights_);
	biases_.input(params_biases_);
	return *this;
}

nnf::ml::Conv2DLayer &nnf::ml::Conv2DLayer::reset_grad()
{
	grad_kernel_weights_.make_zero();
	grad_biases_.make_zero();
	return *this;
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::conv2d_relu_layer(
	usize in_h,
	usize in_w,
	usize num_kernels,
	usize kernel_h,
	usize kernel_w,
	usize stride_h,
	usize stride_w
)
{
	return std::make_unique<Conv2DLayer>(
		in_h, in_w, num_kernels, kernel_h, kernel_w, stride_h, stride_w,
		autodiff::relu_operator(),
		he_kaiming_initializer(kernel_h * kernel_w)
	);
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::conv2d_sigmoid_layer(
	usize in_h,
	usize in_w,
	usize num_kernels,
	usize kernel_h,
	usize kernel_w,
	usize stride_h,
	usize stride_w
)
{
	return std::make_unique<Conv2DLayer>(
		in_h, in_w, num_kernels, kernel_h, kernel_w, stride_h, stride_w,
		autodiff::sigmoid_operator(),
		xavier_glorot_initializer(kernel_h * kernel_w, kernel_h * kernel_w * num_kernels)
	);
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::conv2d_tanh_layer(
	usize in_h,
	usize in_w,
	usize num_kernels,
	usize kernel_h,
	usize kernel_w,
	usize stride_h,
	usize stride_w
)
{
	return std::make_unique<Conv2DLayer>(
		in_h, in_w, num_kernels, kernel_h, kernel_w, stride_h, stride_w,
		autodiff::tanh_operator(),
		xavier_glorot_initializer(kernel_h * kernel_w, kernel_h * kernel_w * num_kernels)
	);
}

std::unique_ptr<nnf::ml::Layer> nnf::ml::conv2d_no_activation_layer(
	usize in_h,
	usize in_w,
	usize num_kernels,
	usize kernel_h,
	usize kernel_w,
	usize stride_h,
	usize stride_w
)
{
	return std::make_unique<Conv2DLayer>(
		in_h, in_w, num_kernels, kernel_h, kernel_w, stride_h, stride_w,
		autodiff::identity_operator(),
		xavier_glorot_initializer(kernel_h * kernel_w, kernel_h * kernel_w * num_kernels)
	);
}
