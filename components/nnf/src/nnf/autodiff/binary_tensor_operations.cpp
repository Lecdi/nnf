#include <nnf/autodiff/binary_tensor_operations.hpp>

#include <cmath>
#include <stdexcept>
#include <nnf/autodiff/unary_tensor_operations.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

void nnf::autodiff::AddOperation::forward(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	Tensor &output
)
{
	output.set(input_lhs);
	output += input_rhs;
}

void nnf::autodiff::AddOperation::back(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad_lhs,
	Tensor &input_grad_rhs
)
{
	input_grad_lhs += output_grad;
	input_grad_rhs += output_grad;
}

void nnf::autodiff::SubOperation::forward(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	Tensor &output
)
{
	output.set(input_lhs);
	output -= input_rhs;
}

void nnf::autodiff::SubOperation::back(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad_lhs,
	Tensor &input_grad_rhs
)
{
	input_grad_lhs += output_grad;
	input_grad_rhs -= output_grad;
}

void nnf::autodiff::MatvecmulOperation::forward(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	Tensor &output
)
{
	output = matvecmul(input_lhs, input_rhs);
}

void nnf::autodiff::MatvecmulOperation::back(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad_lhs,
	Tensor &input_grad_rhs
)
{
	input_grad_lhs += outer_product(output_grad, input_rhs);
	input_grad_rhs += matvecmul(input_lhs.transposed(), output_grad);
}

nnf::autodiff::Conv2DOperation::Conv2DOperation(usize input_h, usize input_w, usize stride_h, usize stride_w)
	: input_h_{ input_h }, input_w_{ input_w }, stride_h_{ stride_h }, stride_w_{ stride_w } {
}

void nnf::autodiff::Conv2DOperation::forward(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	Tensor &output
)
{
	output = input_lhs.conv2d(input_rhs, stride_h_, stride_w_);
}

void nnf::autodiff::Conv2DOperation::back(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad_lhs,
	Tensor &input_grad_rhs
)
{
	if (input_lhs.rank() != 2)
		throw std::logic_error("Attempt to conv2d on tensor which is not rank 2");
	if (input_rhs.rank() != 3)
		throw std::invalid_argument("Kernels for conv2d are not rank 3");
	if (output_grad.rank() != 3)
		throw std::invalid_argument("Output grad for conv2d is not rank 3");

	usize num_kernels = input_rhs.view_dims()[0];
	usize kernel_h = input_rhs.view_dims()[1];
	usize kernel_w = input_rhs.view_dims()[2];
	usize output_grad_elems_per_channel = output_grad.num_elems() / num_kernels;

	input_grad_lhs += matmatmul(
		output_grad.reshaped({ output_grad_elems_per_channel, num_kernels }),
		input_rhs.reshaped({ num_kernels, kernel_h * kernel_w })
	).fold2d(input_h_, input_w_, kernel_h, kernel_w, stride_h_, stride_w_);

	input_grad_rhs += matmatmul(
		output_grad.reshaped({ output_grad_elems_per_channel, num_kernels }).transposed(),
		input_lhs.unfold2d(kernel_h, kernel_w, stride_h_, stride_w_)
	).reshaped({ num_kernels, kernel_h, kernel_w });
}

void nnf::autodiff::SoftmaxCrossentropyOperation::forward(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	Tensor &output
)
{
	auto target = input_rhs.maxpos();

	float expsum = 0.f;
	auto pred_data = input_lhs.view_raw_data();
	for (const usize index : input_lhs.raw_data_positions())
		expsum += std::exp(pred_data[index]);

	output = Tensor::from_data(Vector<usize>{}, { -input_lhs.at(target) + std::log(expsum) });
}

void nnf::autodiff::SoftmaxCrossentropyOperation::back(
	const Tensor &input_lhs,
	const Tensor &input_rhs,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad_lhs,
	Tensor &input_grad_rhs
)
{
	// Ignore label grads; input_grad_rhs is not calculated

	Tensor new_grad{ input_grad_lhs.view_dims() };
	SoftmaxOperation().forward(input_lhs, new_grad);

	auto target = input_rhs.maxpos();
	new_grad.at(target) -= 1.f;

	input_grad_lhs += new_grad;
}
