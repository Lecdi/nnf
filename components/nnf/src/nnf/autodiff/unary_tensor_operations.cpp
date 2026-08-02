#include <nnf/autodiff/unary_tensor_operations.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <nnf/tensor/indexing.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

nnf::autodiff::AtOperation::AtOperation(TensorSingleIndexInput index)
	: index_{ std::move(index.values) } {
}

void nnf::autodiff::AtOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.at({}).set(input.at(index_));
}

void nnf::autodiff::AtOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	input_grad.at(index_) += output_grad.at({});
}

nnf::autodiff::SliceOperation::SliceOperation(TensorMultiIndexInput index)
	: index_{ std::move(index.values) } {
}

void nnf::autodiff::SliceOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.set(input(index_));
}

void nnf::autodiff::SliceOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	input_grad(index_) += output_grad;
}

nnf::autodiff::FlattenOperation::FlattenOperation(VectorInput<usize> old_dims)
	: old_dims_{ std::move(old_dims.values) } {
}

void nnf::autodiff::FlattenOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.set(input.flattened());
}

void nnf::autodiff::FlattenOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	input_grad += output_grad.reshaped(old_dims_);
}

nnf::autodiff::ReshapeOperation::ReshapeOperation(VectorInput<usize> old_dims, VectorInput<usize> new_dims)
	: old_dims_{ std::move(old_dims.values) }, new_dims_{ std::move(new_dims.values) } {
}

void nnf::autodiff::ReshapeOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.set(input.reshaped(new_dims_));
}

void nnf::autodiff::ReshapeOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	input_grad += output_grad.reshaped(old_dims_);
}

nnf::autodiff::TransposeOperation::TransposeOperation(usize dim1, usize dim2)
	: dim1_{ dim1 }, dim2_{ dim2 } {
}

void nnf::autodiff::TransposeOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.set(input.transposed(dim1_, dim2_));
}

void nnf::autodiff::TransposeOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	input_grad += output_grad.transposed(dim1_, dim2_);
}

nnf::autodiff::PermuteOperation::PermuteOperation(VectorInput<usize> dims)
	: dims_forward_{ std::move(dims.values) }, dims_inverse_{ Vector<usize>(dims_forward_.size()) }
{
	for (usize i = 0; i < dims_forward_.size(); ++i)
		dims_inverse_[dims_forward_[i]] = i;
}

void nnf::autodiff::PermuteOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.set(input);
	output.permute_inplace(dims_forward_);
}

void nnf::autodiff::PermuteOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	input_grad += output_grad.permuted(dims_inverse_);
}

nnf::autodiff::ConstMultOperation::ConstMultOperation(float scalar)
	: scalar_{ scalar } {
}

void nnf::autodiff::ConstMultOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.set(input);
	output *= scalar_;
}

void nnf::autodiff::ConstMultOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	input_grad += output_grad * scalar_;
}

nnf::autodiff::ConstDivOperation::ConstDivOperation(float scalar)
	: scalar_{ scalar } {
}

void nnf::autodiff::ConstDivOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.set(input);
	output /= scalar_;
}

void nnf::autodiff::ConstDivOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	input_grad += output_grad / scalar_;
}

void nnf::autodiff::SumOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.set(Tensor::from_data(Vector<usize>{}, { input.sum() }));
}

void nnf::autodiff::SumOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	float grad = output_grad.at({});
	for (const auto &position : input_grad.positions())
	{
		input_grad.at(position) += grad;
	}
}

void nnf::autodiff::MeanOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	output.set(Tensor::from_data(Vector<usize>{}, { input.mean() }));
}

void nnf::autodiff::MeanOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	float grad = output_grad.at({}) / input_grad.num_elems();
	for (const auto &position : input_grad.positions())
	{
		input_grad.at(position) += grad;
	}
}

void nnf::autodiff::SoftmaxOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	float max_logit = input.max();
	float denominator = 0.f;
	auto data = input.view_raw_data();
	for (const auto &index : input.raw_data_positions())
		denominator += std::exp(data[index] - max_logit);
	output.set(input);
	output.apply_inplace([denominator, max_logit](float in) { return std::exp(in - max_logit) / denominator; });
}

void nnf::autodiff::SoftmaxOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	usize num_logits = input.num_elems();

	auto output_data = output.view_raw_data();
	auto output_grad_data = output_grad.view_raw_data();
	auto input_grad_data = Vector<float>(num_logits);

	auto output_data_pos_i_it = output.raw_data_positions().begin();
	auto output_grad_data_pos_i_it = output_grad.raw_data_positions().begin();
	for (usize i = 0; i < num_logits; ++i)
	{
		auto output_data_pos_j_it = output.raw_data_positions().begin();
		auto output_grad_data_pos_j_it = output_grad.raw_data_positions().begin();
		input_grad_data[i] += output_data[*output_data_pos_i_it] * output_grad_data[*output_grad_data_pos_i_it];
		for (usize j = 0; j < num_logits; ++j)
		{
			input_grad_data[i] -= output_data[*output_data_pos_i_it]
				* output_data[*output_data_pos_j_it] * output_grad_data[*output_grad_data_pos_j_it];
			++output_data_pos_j_it;
			++output_grad_data_pos_j_it;
		}
		++output_data_pos_i_it;
		++output_grad_data_pos_i_it;
	}

	input_grad += Tensor::from_data(input_grad.view_dims(), std::move(input_grad_data));
}

nnf::autodiff::MaxPool2DOperation::MaxPool2DOperation(usize h, usize w)
	: w_{ w }, h_{ h } {
}

void nnf::autodiff::MaxPool2DOperation::forward(
	const Tensor &input,
	Tensor &output
)
{
	input_dims_ = input.get_dims();
	input_strides_ = input.get_strides();
	input_offset_ = input.offset();

	if (input_dims_.size() != 3)
		throw std::invalid_argument("Attempt to max pool 2d on tensor which is not rank 3");

	usize out_h = input_dims_[0] / h_;
	usize out_w = input_dims_[1] / w_;
	usize num_channels = input_dims_[2];

	usize input_strides_y = input_strides_[0];
	usize input_strides_x = input_strides_[1];
	usize input_strides_c = input_strides_[2];

	if (out_h == 0 || out_w == 0)
		throw std::invalid_argument("Input to max pool 2d is too small and output would have 0 size");

	auto input_data = input.view_raw_data();
	auto output_data = Vector<float>(out_h * out_w * num_channels);

	from_positions_.clear();
	from_positions_.reserve(out_h * out_w * num_channels);

	usize input_data_index = 0;
	usize output_data_index = 0;
	for (usize y_in_output = 0; y_in_output < out_h; ++y_in_output)
	{
		for (usize x_in_output = 0; x_in_output < out_w; ++x_in_output)
		{
			for (usize channel = 0; channel < num_channels; ++channel)
			{
				output_data[output_data_index] = input_data[input_data_index];
				for (usize y_in_window = 0; y_in_window < h_; ++y_in_window)
				{
					for (usize x_in_window = 0; x_in_window < w_; ++x_in_window)
					{
						if (input_data[input_data_index] >= output_data[output_data_index])
						{
							output_data[output_data_index] = input_data[input_data_index];
							from_positions_.push_back(input_data_index);
						}
						input_data_index += input_strides_x;
					}
					input_data_index -= w_ * input_strides_x;
					input_data_index += input_strides_y;
				}
				input_data_index -= h_ * input_strides_y;
				input_data_index += input_strides_c;
				++output_data_index;
			}
			input_data_index -= num_channels * input_strides_c;
			input_data_index += w_ * input_strides_x;
		}
		input_data_index -= out_w * w_ * input_strides_x;
		input_data_index += h_ * input_strides_y;
	}

	output = Tensor::from_data({ out_h, out_w, num_channels }, std::move(output_data));
}

void nnf::autodiff::MaxPool2DOperation::back(
	const Tensor &input,
	const Tensor &output,
	const Tensor &output_grad,
	Tensor &input_grad
)
{
	if (!std::equal(input_grad.view_dims().begin(), input_grad.view_dims().end(), input_dims_.begin(), input_dims_.end()))
		throw std::runtime_error(
			"[nnf::autodiff::MaxPool2DOperation] Input grad has different dimensions from input"
		);

	if (
		!std::equal(
			input.view_strides().begin(), input.view_strides().end(),
			input_strides_.begin(), input_strides_.end()
		) || input.offset() != input_offset_)
		throw std::runtime_error(
			"[nnf::autodiff::MaxPool2DOperation] Input grad has different data layout from input"
		);

	auto output_grad_raw_data = output_grad.view_raw_data();
	auto output_grad_raw_data_position_it = output_grad.raw_data_positions().begin();

	for (const usize from_position : from_positions_)
	{
		input_grad.raw_data_at(from_position) += output_grad_raw_data[*output_grad_raw_data_position_it];
		++output_grad_raw_data_position_it;
	}
}
