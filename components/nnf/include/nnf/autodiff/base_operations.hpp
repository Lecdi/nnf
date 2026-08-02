#pragma once

#include <algorithm>
#include <concepts>
#include <memory>
#include <span>
#include <utility>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf::autodiff
{
	class EmptyOperation : public Operation
	{
	public:
		void forward(
			std::span<const Tensor *> inputs,
			Tensor *output
		) override
		{
		}
		void back(
			std::span<const Tensor *> inputs,
			const Tensor *output,
			const Tensor *output_grad,
			std::span<Tensor *> input_grads
		) override
		{
		}
	};

	class UnaryTensorOperation : public Operation
	{
	public:
		virtual void forward(
			const Tensor &input,
			Tensor &output
		) = 0;
		virtual void back(
			const Tensor &input,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad
		) = 0;

		void forward(
			std::span<const Tensor *> inputs,
			Tensor *output
		) override;
		void back(
			std::span<const Tensor *> inputs,
			const Tensor *output,
			const Tensor *output_grad,
			std::span<Tensor *> input_grads
		) override;
	};

	class BinaryTensorOperation : public Operation
	{
	public:
		virtual void forward(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			Tensor &output
		) = 0;
		virtual void back(
			const Tensor &input_lhs,
			const Tensor &input_rhs,
			const Tensor &output,
			const Tensor &output_grad,
			Tensor &input_grad_lhs,
			Tensor &input_grad_rhs
		) = 0;

		void forward(
			std::span<const Tensor *> inputs,
			Tensor *output
		) override;
		void back(
			std::span<const Tensor *> inputs,
			const Tensor *output,
			const Tensor *output_grad,
			std::span<Tensor *> input_grads
		) override;
	};

	class UnaryScalarOperation : public Operation
	{
	public:
		virtual float forward(float input) = 0;
		virtual float back(float output) = 0;

		void forward(
			std::span<const Tensor *> inputs,
			Tensor *output
		) override;
		void back(
			std::span<const Tensor *> inputs,
			const Tensor *output,
			const Tensor *output_grad,
			std::span<Tensor *> input_grads
		) override;
	};

	template<typename T>
	concept UnaryScalarOperationType = std::derived_from<T, UnaryScalarOperation>;

	class UnaryScalarOperator
	{
	public:
		virtual ~UnaryScalarOperator() {}

		virtual Value operator()(const Value &operand) const = 0;
	};

	template<UnaryScalarOperationType Op>
	class UnaryScalarOperatorInstance : public UnaryScalarOperator
	{
	public:
		explicit UnaryScalarOperatorInstance(Op op) : op_{ std::move(op) } {}

		Value operator()(const Value &operand) const
		{
			auto new_node = std::make_shared<Node>(
				Vector<std::shared_ptr<Node>>{operand.node},
				std::make_unique<Op>(op_),
				operand.view_dims()
			);
			return Value(new_node);
		}

	private:
		Op op_;
	};
}
