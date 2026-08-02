#include <nnf/autodiff/base_operations.hpp>

#include <span>
#include <stdexcept>
#include <nnf/tensor/tensor.hpp>

void nnf::autodiff::UnaryTensorOperation::forward(
	std::span<const Tensor *> inputs,
	Tensor *output
)
{
	if (inputs.size() != 1)
		throw std::logic_error("Unary tensor operation has wrong number of inputs, should be 1");
	forward(*inputs[0], *output);
}

void nnf::autodiff::UnaryTensorOperation::back(
	std::span<const Tensor *> inputs,
	const Tensor *output,
	const Tensor *output_grad,
	std::span<Tensor *> input_grads
)
{
	if (inputs.size() != 1)
		throw std::logic_error("Unary tensor operation has wrong number of inputs, should be 1");
	if (input_grads.size() != 1)
		throw std::logic_error("Unary tensor operation has wrong number of input grads, should be 1");
	back(*inputs[0], *output, *output_grad, *input_grads[0]);
}

void nnf::autodiff::BinaryTensorOperation::forward(
	std::span<const Tensor *> inputs,
	Tensor *output
)
{
	if (inputs.size() != 2)
		throw std::logic_error("Binary tensor operation has wrong number of inputs, should be 2");
	forward(*inputs[0], *inputs[1], *output);
}

void nnf::autodiff::BinaryTensorOperation::back(
	std::span<const Tensor *> inputs,
	const Tensor *output,
	const Tensor *output_grad,
	std::span<Tensor *> input_grads
)
{
	if (inputs.size() != 2)
		throw std::logic_error("Binary tensor operation has wrong number of inputs, should be 2");
	if (input_grads.size() != 2)
		throw std::logic_error("Binary tensor operation has wrong number of input grads, should be 2");
	back(*inputs[0], *inputs[1], *output, *output_grad, *input_grads[0], *input_grads[1]);
}

void nnf::autodiff::UnaryScalarOperation::forward(
	std::span<const Tensor *> inputs,
	Tensor *output
)
{
	if (inputs.size() != 1)
		throw std::logic_error("Unary scalar operation has wrong number of inputs, should be 1");
	output->set(*inputs[0]);
	output->apply_inplace([this](float input) { return forward(input); });
}

void nnf::autodiff::UnaryScalarOperation::back(
	std::span<const Tensor *> inputs,
	const Tensor *output,
	const Tensor *output_grad,
	std::span<Tensor *> input_grads
)
{
	if (inputs.size() != 1)
		throw std::logic_error("Unary scalar operation has wrong number of inputs, should be 1");
	if (input_grads.size() != 1)
		throw std::logic_error("Unary scalar operation has wrong number of input grads, should be 1");
	auto new_grad = inputs[0]->copy();
	new_grad.apply_inplace([this](float input) { return back(input); });
	for (const auto &position : new_grad.positions())
		new_grad.at(position) *= output_grad->at(position);
	*input_grads[0] += new_grad;
}
