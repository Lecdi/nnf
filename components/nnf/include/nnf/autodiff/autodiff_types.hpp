#pragma once

#include <memory>
#include <span>
#include <unordered_set>
#include <nnf/tensor/indexing.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf::autodiff
{
	class Operation
	{
	public:
		virtual ~Operation() {}

		virtual void forward(
			std::span<const Tensor *> inputs,
			Tensor *output
		) = 0;
		virtual void back(
			std::span<const Tensor *> inputs,
			const Tensor *output,
			const Tensor *output_grad,
			std::span<Tensor *> input_grads
		) = 0;
	};

	struct Node
	{
		Node() = delete;
		explicit Node(
			VectorInput<std::shared_ptr<Node>> prevs,
			std::unique_ptr<Operation> op,
			VectorInput<usize> dims
		);

		void forward();
		void back();

		const Vector<std::shared_ptr<Node>> prevs;
		const std::unique_ptr<Operation> op;
		Tensor value;
		Tensor grad;
	};

	class Value
	{
	public:
		Value() = default;
		explicit Value(std::shared_ptr<Node> node);

		virtual ~Value() {}

		bool is_empty() const;
		const Tensor &value() const;
		const Tensor &grad() const;
		usize rank() const;
		VectorView<usize> view_dims() const;
		Vector<usize> get_dims() const;
		usize num_elems() const;

		Value at(TensorSingleIndexInput index) const;
		Value operator()(TensorMultiIndexInput index) const;
		Value flattened() const;
		Value reshaped(VectorInput<usize> new_dims) const;
		Value transposed(usize dim1, usize dim2) const;
		Value permuted(VectorInput<usize> dims) const;

		friend Value operator+(const Value &lhs, const Value &rhs);
		friend Value operator-(const Value &lhs, const Value &rhs);
		friend Value operator*(const Value &lhs, float scalar);
		friend Value operator*(float scalar, const Value &rhs);
		friend Value operator/(const Value &lhs, float scalar);

		friend Value matvecmul(const Value &lhs, const Value &rhs);
		Value conv2d(const Value &kernels, usize stride_h = 1, usize stride_w = 1) const;

		Value softmax() const;
		friend Value softmax_crossentropy(const Value &pred, const Value &label);
		Value max_pool_2d(usize h, usize w) const;

		Value sum() const;
		Value mean() const;

		std::shared_ptr<Node> node;
	};

	class InputValue : public Value
	{
	public:
		InputValue() = delete;
		explicit InputValue(std::shared_ptr<Node> node);
		explicit InputValue(VectorInput<usize> dims);

		static InputValue with_value(const Tensor &value);

		InputValue &input(const Tensor &value);
		InputValue &input_raw_data(VectorView<float> data);
	};

	class AutodiffEngine
	{
	public:
		explicit AutodiffEngine(const Value &value);

		bool is_compiled() const;

		AutodiffEngine &compile();
		AutodiffEngine &fill_forward();
		AutodiffEngine &set_final_grad(const Tensor &grad);
		AutodiffEngine &fill_back();

	private:
		static void build_topo_nodes(
			Node *node,
			std::unordered_set<Node *> &visited,
			Vector<Node *> &topo_nodes
		);

		std::shared_ptr<Node> node_;
		Vector<Node *> topo_nodes_;
	};
}
