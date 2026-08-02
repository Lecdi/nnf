#pragma once

#include <memory>
#include <nnf/autodiff/autodiff_types.hpp>

namespace nnf::ml
{
	class LossFunction
	{
	public:
		virtual ~LossFunction() {}

		virtual autodiff::Value operator()(const autodiff::Value &pred, const autodiff::Value &label) const = 0;
	};

	class MSELoss : public LossFunction
	{
	public:
		autodiff::Value operator()(const autodiff::Value &pred, const autodiff::Value &label) const override;
	};

	std::unique_ptr<LossFunction> mse_loss();

	class SoftmaxCrossentropyLoss : public LossFunction
	{
	public:
		autodiff::Value operator()(const autodiff::Value &pred, const autodiff::Value &label) const override;
	};

	std::unique_ptr<LossFunction> softmax_crossentropy_loss();
}
