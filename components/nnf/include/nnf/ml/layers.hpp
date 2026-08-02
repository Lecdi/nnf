#pragma once

#include <array>
#include <fstream>
#include <memory>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/autodiff/base_operations.hpp>
#include <nnf/ml/param_initializers.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf::ml
{
	class Layer
	{
	public:
		virtual ~Layer() {}

		virtual Layer &init_params() = 0;
		virtual Layer &read_params_from(std::ifstream &stream) = 0;
		virtual Layer &write_params_to(std::ofstream &stream) = 0;
		virtual Layer &set_prev_value(autodiff::Value prev_value) = 0;

		virtual Layer &accumulate_grad() = 0;
		virtual Layer &update_params(float scale_by) = 0;
		virtual Layer &reset_grad() = 0;

		autodiff::Value out_value;
	};

	class FlattenLayer : public Layer
	{
	public:
		FlattenLayer &init_params() override { return *this; }
		FlattenLayer &read_params_from(std::ifstream &stream) override { return *this; }
		FlattenLayer &write_params_to(std::ofstream &stream) override { return *this; }
		FlattenLayer &set_prev_value(autodiff::Value prev_value) override;

		FlattenLayer &accumulate_grad() override { return *this; }
		FlattenLayer &update_params(float scale_by) override { return *this; }
		FlattenLayer &reset_grad() override { return *this; }
	};

	std::unique_ptr<Layer> flatten_layer();

	class ReshapeLayer : public Layer
	{
	public:
		ReshapeLayer(VectorInput<usize> new_dims);

		ReshapeLayer &init_params() override { return *this; }
		ReshapeLayer &read_params_from(std::ifstream &stream) override { return *this; }
		ReshapeLayer &write_params_to(std::ofstream &stream) override { return *this; }
		ReshapeLayer &set_prev_value(autodiff::Value prev_value) override;

		ReshapeLayer &accumulate_grad() override { return *this; }
		ReshapeLayer &update_params(float scale_by) override { return *this; }
		ReshapeLayer &reset_grad() override { return *this; }

	private:
		Vector<usize> new_dims_;
	};

	std::unique_ptr<Layer> reshape_layer(VectorInput<usize> new_dims);

	class SoftmaxLayer : public Layer
	{
	public:
		SoftmaxLayer &init_params() override { return *this; }
		SoftmaxLayer &read_params_from(std::ifstream &stream) override { return *this; }
		SoftmaxLayer &write_params_to(std::ofstream &stream) override { return *this; }
		SoftmaxLayer &set_prev_value(autodiff::Value prev_value) override;

		SoftmaxLayer &accumulate_grad() override { return *this; }
		SoftmaxLayer &update_params(float scale_by) override { return *this; }
		SoftmaxLayer &reset_grad() override { return *this; }
	};

	std::unique_ptr<Layer> softmax_layer();

	class MaxPool2DLayer : public Layer
	{
	public:
		MaxPool2DLayer(ssize h, ssize w);

		MaxPool2DLayer &init_params() override { return *this; }
		MaxPool2DLayer &read_params_from(std::ifstream &stream) override { return *this; }
		MaxPool2DLayer &write_params_to(std::ofstream &stream) override { return *this; }
		MaxPool2DLayer &set_prev_value(autodiff::Value prev_value) override;

		MaxPool2DLayer &accumulate_grad() override { return *this; }
		MaxPool2DLayer &update_params(float scale_by) override { return *this; }
		MaxPool2DLayer &reset_grad() override { return *this; }

	private:
		ssize h_, w_;
	};

	std::unique_ptr<Layer> max_pool_2d_layer(ssize h, ssize w);

	class DenseLayer : public Layer
	{
	public:
		DenseLayer(
			usize in_elems,
			usize out_elems,
			std::unique_ptr<autodiff::UnaryScalarOperator> activation,
			std::unique_ptr<ParamInitializer> weights_initializer
		);

		DenseLayer &init_params() override;
		DenseLayer &read_params_from(std::ifstream &stream) override;
		DenseLayer &write_params_to(std::ofstream &stream) override;
		DenseLayer &set_prev_value(autodiff::Value prev_value) override;

		DenseLayer &accumulate_grad() override;
		DenseLayer &update_params(float scale_by) override;
		DenseLayer &reset_grad() override;

	private:
		usize in_elems_;
		usize out_elems_;
		std::unique_ptr<autodiff::UnaryScalarOperator> activation_;
		std::unique_ptr<ParamInitializer> weights_initializer_;

		autodiff::InputValue weights_;
		autodiff::InputValue biases_;

		Tensor params_weights_;
		Tensor grad_weights_;
		Tensor params_biases_;
		Tensor grad_biases_;
	};

	std::unique_ptr<Layer> dense_relu_layer(usize in, usize out);
	std::unique_ptr<Layer> dense_sigmoid_layer(usize in, usize out);
	std::unique_ptr<Layer> dense_tanh_layer(usize in, usize out);
	std::unique_ptr<Layer> dense_no_activation_layer(usize in, usize out);

	class Conv2DLayer : public Layer
	{
	public:
		Conv2DLayer(
			usize in_h,
			usize in_w,
			usize num_kernels,
			usize kernel_h,
			usize kernel_w,
			usize stride_h,
			usize stride_w,
			std::unique_ptr<autodiff::UnaryScalarOperator> activation,
			std::unique_ptr<ParamInitializer> kernels_initializer
		);

		Conv2DLayer &init_params() override;
		Conv2DLayer &read_params_from(std::ifstream &stream) override;
		Conv2DLayer &write_params_to(std::ofstream &stream) override;
		Conv2DLayer &set_prev_value(autodiff::Value prev_value) override;

		Conv2DLayer &accumulate_grad() override;
		Conv2DLayer &update_params(float scale_by) override;
		Conv2DLayer &reset_grad() override;

	private:
		usize in_h_;
		usize in_w_;
		usize num_kernels_;
		usize kernel_h_;
		usize kernel_w_;
		usize stride_h_;
		usize stride_w_;
		std::array<usize, 3> out_dims_;
		std::unique_ptr<autodiff::UnaryScalarOperator> activation_;
		std::unique_ptr<ParamInitializer> kernel_weights_initializer_;

		autodiff::InputValue kernel_weights_;
		autodiff::InputValue biases_;

		Tensor params_kernel_weights_;
		Tensor grad_kernel_weights_;
		Tensor params_biases_;
		Tensor grad_biases_;
	};

	std::unique_ptr<Layer> conv2d_relu_layer(
		usize in_h,
		usize in_w,
		usize num_kernels,
		usize kernel_h,
		usize kernel_w,
		usize stride_h = 1,
		usize stride_w = 1
	);
	std::unique_ptr<Layer> conv2d_sigmoid_layer(
		usize in_h,
		usize in_w,
		usize num_kernels,
		usize kernel_h,
		usize kernel_w,
		usize stride_h = 1,
		usize stride_w = 1
	);
	std::unique_ptr<Layer> conv2d_tanh_layer(
		usize in_h,
		usize in_w,
		usize num_kernels,
		usize kernel_h,
		usize kernel_w,
		usize stride_h = 1,
		usize stride_w = 1
	);
	std::unique_ptr<Layer> conv2d_no_activation_layer(
		usize in_h,
		usize in_w,
		usize num_kernels,
		usize kernel_h,
		usize kernel_w,
		usize stride_h = 1,
		usize stride_w = 1
	);
}
