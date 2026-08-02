#include <nnf/autodiff/autodiff_types.hpp>

#include <memory>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <nnf/autodiff/base_operations.hpp>
#include <nnf/autodiff/binary_tensor_operations.hpp>
#include <nnf/autodiff/unary_tensor_operations.hpp>
#include <nnf/tensor/indexing.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

nnf::autodiff::Node::Node(
	VectorInput<std::shared_ptr<Node>> prevs,
	std::unique_ptr<Operation> op,
	VectorInput<usize> dims
) : prevs{ std::move(prevs.values) }, op{ std::move(op) }, value{ dims }, grad{ dims } {
}

void nnf::autodiff::Node::forward()
{
	Vector<const Tensor *> inputs{};
	inputs.reserve(prevs.size());
	for (const auto &prev : prevs)
	{
		inputs.push_back(&prev->value);
	}
	op->forward(inputs, &value);
}

void nnf::autodiff::Node::back()
{
	Vector<const Tensor *> inputs{};
	Vector<Tensor *> input_grads{};
	inputs.reserve(prevs.size());
	input_grads.reserve(prevs.size());
	for (const auto &prev : prevs)
	{
		inputs.push_back(&prev->value);
		input_grads.push_back(&prev->grad);
	}
	op->back(inputs, &value, &grad, input_grads);
}

nnf::autodiff::Value::Value(std::shared_ptr<Node> node) : node{ node } {}

bool nnf::autodiff::Value::is_empty() const
{
	return !node;
}

const nnf::Tensor &nnf::autodiff::Value::value() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	return node->value;
}

const nnf::Tensor &nnf::autodiff::Value::grad() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	return node->grad;
}

nnf::usize nnf::autodiff::Value::rank() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	return node->value.rank();
}

nnf::VectorView<nnf::usize> nnf::autodiff::Value::view_dims() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	return node->value.view_dims();
}

nnf::Vector<nnf::usize> nnf::autodiff::Value::get_dims() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	return node->value.get_dims();
}

nnf::usize nnf::autodiff::Value::num_elems() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	return node->value.num_elems();
}

nnf::autodiff::Value nnf::autodiff::Value::at(TensorSingleIndexInput index) const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::make_unique<autodiff::AtOperation>(std::move(index.values)),
		VectorInitializer<usize>{}
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::operator()(TensorMultiIndexInput index) const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::make_unique<autodiff::SliceOperation>(std::move(index.values)),
		node->value(index.values).view_dims()
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::flattened() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::make_unique<autodiff::FlattenOperation>(view_dims()),
		VectorInitializer<usize>{num_elems()}
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::reshaped(VectorInput<usize> new_dims) const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	auto operation = std::make_unique<autodiff::ReshapeOperation>(
		view_dims(),
		new_dims.values
	);
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::move(operation),
		std::move(new_dims.values)
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::transposed(usize dim1, usize dim2) const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	auto new_dims = get_dims();
	if (dim1 >= new_dims.size() || dim2 >= new_dims.size())
		throw std::out_of_range("Dimension to transpose is >= rank of value so does not exist");
	if (dim1 == dim2)
		throw std::invalid_argument("Dimensions to transpose are the same");
	std::swap(new_dims[dim1], new_dims[dim2]);
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::make_unique<autodiff::TransposeOperation>(dim1, dim2),
		std::move(new_dims)
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::permuted(VectorInput<usize> dims) const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	auto new_dims = node->value.permuted(dims.values).get_dims();
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::make_unique<autodiff::PermuteOperation>(std::move(dims.values)),
		std::move(new_dims)
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::operator*(const Value &lhs, float scalar)
{
	if (lhs.is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{lhs.node},
		std::make_unique<autodiff::ConstMultOperation>(scalar),
		lhs.view_dims()
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::operator*(float scalar, const Value &rhs)
{
	if (rhs.is_empty()) throw std::logic_error("Value is empty");
	return rhs * scalar;
}

nnf::autodiff::Value nnf::autodiff::operator/(const Value &lhs, float scalar)
{
	if (lhs.is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{lhs.node},
		std::make_unique<autodiff::ConstDivOperation>(scalar),
		lhs.view_dims()
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::operator+(const Value &lhs, const Value &rhs)
{
	if (lhs.is_empty() || rhs.is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{lhs.node, rhs.node},
		std::make_unique<autodiff::AddOperation>(),
		lhs.view_dims()
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::operator-(const Value &lhs, const Value &rhs)
{
	if (lhs.is_empty() || rhs.is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{lhs.node, rhs.node},
		std::make_unique<autodiff::SubOperation>(),
		lhs.view_dims()
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::matvecmul(const Value &lhs, const Value &rhs)
{
	if (lhs.is_empty() || rhs.is_empty()) throw std::logic_error("Value is empty");
	if (lhs.rank() != 2)
		throw std::logic_error("Lhs of matvecmul is not rank 2");
	if (rhs.rank() != 1)
		throw std::logic_error("Rhs of matvecmul is not rank 1");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{lhs.node, rhs.node},
		std::make_unique<autodiff::MatvecmulOperation>(),
		VectorInitializer<usize>{ lhs.view_dims()[0] }
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::conv2d(const Value &kernels, usize stride_h, usize stride_w) const
{
	if (is_empty() || kernels.is_empty()) throw std::logic_error("Value is empty");
	if (rank() != 2)
		throw std::logic_error("Attempt to conv2d on value which is not rank 2");
	if (kernels.rank() != 3)
		throw std::logic_error("Kernels for conv2d are not rank 3");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node, kernels.node},
		std::make_unique<autodiff::Conv2DOperation>(view_dims()[0], view_dims()[1], stride_h, stride_w),
		VectorView<usize>(Tensor::get_conv2d_dims(
			value().view_dims()[0],
			value().view_dims()[1],
			kernels.value().view_dims()[0],
			kernels.value().view_dims()[1],
			kernels.value().view_dims()[2],
			stride_h, stride_w
		))
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::softmax() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::make_unique<autodiff::SoftmaxOperation>(),
		view_dims()
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::softmax_crossentropy(const Value &pred, const Value &label)
{
	if (pred.is_empty() || label.is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{pred.node, label.node},
		std::make_unique<autodiff::SoftmaxCrossentropyOperation>(),
		Vector<usize>{}
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::max_pool_2d(usize h, usize w) const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	if (rank() != 3) throw std::logic_error("Attempt to max pool 2d on value which is not rank 3");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::make_unique<autodiff::MaxPool2DOperation>(h, w),
		Vector<usize>{ view_dims()[0] / h, view_dims()[1] / w, view_dims()[2] }
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::sum() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::make_unique<autodiff::SumOperation>(),
		Vector<usize>{}
	);
	return Value(new_node);
}

nnf::autodiff::Value nnf::autodiff::Value::mean() const
{
	if (is_empty()) throw std::logic_error("Value is empty");
	auto new_node = std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{node},
		std::make_unique<autodiff::MeanOperation>(),
		Vector<usize>{}
	);
	return Value(new_node);
}

nnf::autodiff::InputValue::InputValue(std::shared_ptr<Node> node) : Value{ node } {}

nnf::autodiff::InputValue::InputValue(VectorInput<usize> dims)
	: Value{ std::make_shared<Node>(
		Vector<std::shared_ptr<Node>>{},
		std::make_unique<autodiff::EmptyOperation>(),
		std::move(dims.values)
	) }
{
}

nnf::autodiff::InputValue nnf::autodiff::InputValue::with_value(const Tensor &value)
{
	InputValue result{ value.view_dims() };
	result.input(value);
	return result;
}

nnf::autodiff::InputValue &nnf::autodiff::InputValue::input(const Tensor &value)
{
	node->value.set(value);
	return *this;
}

nnf::autodiff::InputValue &nnf::autodiff::InputValue::input_raw_data(VectorView<float> data)
{
	node->value.set_from_contiguous_data(data);
	return *this;
}

nnf::autodiff::AutodiffEngine::AutodiffEngine(const Value &value)
	: node_{ value.node } {
}

bool nnf::autodiff::AutodiffEngine::is_compiled() const
{
	return topo_nodes_.size() != 0;
}

nnf::autodiff::AutodiffEngine &nnf::autodiff::AutodiffEngine::compile()
{
	if (is_compiled()) throw std::logic_error("Autodiff engine compiled for second time");
	std::unordered_set<Node *> visited{};
	build_topo_nodes(node_.get(), visited, topo_nodes_);
	return *this;
}

nnf::autodiff::AutodiffEngine &nnf::autodiff::AutodiffEngine::fill_forward()
{
	if (!is_compiled()) throw std::logic_error("Use of autodiff engine before compilation");
	for (auto &node : topo_nodes_)
	{
		node->grad.make_zero();
		node->forward();
	}
	return *this;
}

nnf::autodiff::AutodiffEngine &nnf::autodiff::AutodiffEngine::set_final_grad(const Tensor &grad)
{
	if (!is_compiled()) throw std::logic_error("Use of autodiff engine before compilation");
	topo_nodes_.back()->grad.set(grad);
	return *this;
}

nnf::autodiff::AutodiffEngine &nnf::autodiff::AutodiffEngine::fill_back()
{
	if (!is_compiled()) throw std::logic_error("Use of autodiff engine before compilation");
	for (auto &node : topo_nodes_ | std::views::reverse)
	{
		node->back();
	}
	return *this;
}

void nnf::autodiff::AutodiffEngine::build_topo_nodes(
	Node *node,
	std::unordered_set<Node *> &visited,
	Vector<Node *> &topo_nodes
)
{
	if (visited.count(node)) return;
	visited.insert(node);
	for (const auto &prev : node->prevs)
		build_topo_nodes(prev.get(), visited, topo_nodes);
	topo_nodes.push_back(node);
}
