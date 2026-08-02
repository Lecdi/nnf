#include <nnf/ml/losses.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/autodiff/unary_scalar_operations.hpp>

nnf::autodiff::Value nnf::ml::MSELoss::operator()(const autodiff::Value &pred, const autodiff::Value &label) const
{
	if (!std::equal(pred.view_dims().begin(), pred.view_dims().end(), label.view_dims().begin(), label.view_dims().end()))
		throw std::invalid_argument("Dims of pred and label are not the same");
	return autodiff::square()(pred - label).mean();
}

std::unique_ptr<nnf::ml::LossFunction> nnf::ml::mse_loss()
{
	return std::make_unique<MSELoss>();
}

nnf::autodiff::Value nnf::ml::SoftmaxCrossentropyLoss::operator()(const autodiff::Value &pred, const autodiff::Value &label) const
{
	if (!std::equal(pred.view_dims().begin(), pred.view_dims().end(), label.view_dims().begin(), label.view_dims().end()))
		throw std::invalid_argument("Dims of pred and label are not the same");
	return softmax_crossentropy(pred, label);
}

std::unique_ptr<nnf::ml::LossFunction> nnf::ml::softmax_crossentropy_loss()
{
	return std::make_unique<SoftmaxCrossentropyLoss>();
}
