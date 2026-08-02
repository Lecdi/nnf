#pragma once

#include <nnf/autodiff/base_operations.hpp>
#include <nnf/tensor/indexing.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf::autodiff
{
	class AtOperation : public UnaryTensorOperation
	{
	public:
		AtOperation() = delete;
		explicit AtOperation(TensorSingleIndexInput index);

		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;

	private:
		TensorSingleIndex index_;
	};

	class SliceOperation : public UnaryTensorOperation
	{
	public:
		SliceOperation() = delete;
		explicit SliceOperation(TensorMultiIndexInput index);

		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;

	private:
		TensorMultiIndex index_;
	};

	class FlattenOperation : public UnaryTensorOperation
	{
	public:
		FlattenOperation() = delete;
		explicit FlattenOperation(VectorInput<usize> old_dims);

		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;

	private:
		Vector<usize> old_dims_;
	};

	class ReshapeOperation : public UnaryTensorOperation
	{
	public:
		ReshapeOperation() = delete;
		explicit ReshapeOperation(VectorInput<usize> old_dims, VectorInput<usize> new_dims);

		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;

	private:
		Vector<usize> old_dims_;
		Vector<usize> new_dims_;
	};

	class TransposeOperation : public UnaryTensorOperation
	{
	public:
		TransposeOperation() = delete;
		explicit TransposeOperation(usize dim1, usize dim2);

		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;

	private:
		usize dim1_;
		usize dim2_;
	};

	class PermuteOperation : public UnaryTensorOperation
	{
	public:
		PermuteOperation() = delete;
		explicit PermuteOperation(VectorInput<usize> dims);

		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;

	private:
		Vector<usize> dims_forward_;
		Vector<usize> dims_inverse_;
	};

	class ConstMultOperation : public UnaryTensorOperation
	{
	public:
		ConstMultOperation() = delete;
		explicit ConstMultOperation(float scalar);

		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;

	private:
		float scalar_;
	};

	class ConstDivOperation : public UnaryTensorOperation
	{
	public:
		ConstDivOperation() = delete;
		explicit ConstDivOperation(float scalar);

		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;

	private:
		float scalar_;
	};

	class SumOperation : public UnaryTensorOperation
	{
	public:
		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;
	};

	class MeanOperation : public UnaryTensorOperation
	{
	public:
		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;
	};

	class SoftmaxOperation : public UnaryTensorOperation
	{
	public:
		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;
	};

	class MaxPool2DOperation : public UnaryTensorOperation
	{
	public:
		MaxPool2DOperation() = delete;
		explicit MaxPool2DOperation(usize h, usize w);

		void forward(
			const Tensor &input,
			Tensor &output
		) override;
		void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) override;

	private:
		usize h_, w_;
		Vector<usize> input_dims_{};
		Vector<usize> input_strides_{};
		usize input_offset_{};
		Vector<usize> from_positions_{};
	};
}
