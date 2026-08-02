#pragma once

#include <nnf/autodiff/base_operations.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf::autodiff
{
	class AddOperation : public BinaryTensorOperation
	{
	public:
		void forward(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			Tensor &output
		) override;
		void back(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad_lhs,
			Tensor &input_grad_rhs
		) override;
	};

	class SubOperation : public BinaryTensorOperation
	{
	public:
		void forward(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			Tensor &output
		) override;
		void back(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad_lhs,
			Tensor &input_grad_rhs
		) override;
	};

	class MatvecmulOperation : public BinaryTensorOperation
	{
	public:
		void forward(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			Tensor &output
		) override;
		void back(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad_lhs,
			Tensor &input_grad_rhs
		) override;
	};

	class Conv2DOperation : public BinaryTensorOperation
	{
	public:
		Conv2DOperation() = delete;
		explicit Conv2DOperation(usize input_h, usize input_w, usize stride_h = 1, usize stride_w = 1);

		void forward(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			Tensor &output
		) override;
		void back(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad_lhs,
			Tensor &input_grad_rhs
		) override;

	private:
		usize input_h_;
		usize input_w_;
		usize stride_h_;
		usize stride_w_;
	};

	class SoftmaxCrossentropyOperation : public BinaryTensorOperation
	{
	public:
		void forward(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			Tensor &output
		) override;
		void back(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad_lhs,
			Tensor &input_grad_rhs
		) override;
	};
}
