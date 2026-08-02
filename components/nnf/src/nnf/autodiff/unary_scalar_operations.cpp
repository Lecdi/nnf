#include <nnf/autodiff/unary_scalar_operations.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <nnf/autodiff/base_operations.hpp>
#include <nnf/utils/math.hpp>

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::IdentityOperation> nnf::autodiff::identity()
{
	return UnaryScalarOperatorInstance<IdentityOperation>(IdentityOperation());
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::identity_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<IdentityOperation>>(IdentityOperation());
}


float nnf::autodiff::SgnOperation::forward(float input)
{
	return nnf::sgn(input);
}

float nnf::autodiff::SgnOperation::back(float input)
{
	return 0;
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::SgnOperation> nnf::autodiff::sgn()
{
	return UnaryScalarOperatorInstance<SgnOperation>(SgnOperation());
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::sgn_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<SgnOperation>>(SgnOperation());
}

float nnf::autodiff::AbsOperation::forward(float input)
{
	return std::abs(input);
}

float nnf::autodiff::AbsOperation::back(float input)
{
	return nnf::sgn(input);
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::AbsOperation> nnf::autodiff::abs()
{
	return UnaryScalarOperatorInstance<AbsOperation>(AbsOperation());
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::abs_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<AbsOperation>>(AbsOperation());
}

nnf::autodiff::PowOperation::PowOperation(float exponent) : exponent_{ exponent } {}

float nnf::autodiff::PowOperation::forward(float input)
{
	return std::pow(input, exponent_);
}

float nnf::autodiff::PowOperation::back(float input)
{
	return exponent_ * std::pow(input, exponent_ - 1);
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::PowOperation> nnf::autodiff::pow(float exponent)
{
	return UnaryScalarOperatorInstance<PowOperation>(PowOperation(exponent));
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::PowOperation> nnf::autodiff::sqrt()
{
	return UnaryScalarOperatorInstance<PowOperation>(PowOperation(0.5f));
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::PowOperation> nnf::autodiff::square()
{
	return UnaryScalarOperatorInstance<PowOperation>(PowOperation(2.0f));
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::pow_operator(float exponent)
{
	return std::make_unique<UnaryScalarOperatorInstance<PowOperation>>(PowOperation(exponent));
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::sqrt_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<PowOperation>>(PowOperation(0.5f));
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::square_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<PowOperation>>(PowOperation(2.0f));
}

float nnf::autodiff::ExpOperation::forward(float input)
{
	return std::exp(input);
}

float nnf::autodiff::ExpOperation::back(float input)
{
	return std::exp(input);
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::ExpOperation> nnf::autodiff::exp()
{
	return UnaryScalarOperatorInstance<ExpOperation>(ExpOperation());
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::exp_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<ExpOperation>>(ExpOperation());
}

float nnf::autodiff::LogOperation::forward(float input)
{
	return std::log(input);
}

float nnf::autodiff::LogOperation::back(float input)
{
	return 1.f / input;
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::LogOperation> nnf::autodiff::log()
{
	return UnaryScalarOperatorInstance<LogOperation>(LogOperation());
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::log_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<LogOperation>>(LogOperation());
}

float nnf::autodiff::ReLUOperation::forward(float input)
{
	return std::max(input, 0.f);
}

float nnf::autodiff::ReLUOperation::back(float input)
{
	return (input > 0);
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::ReLUOperation> nnf::autodiff::relu()
{
	return UnaryScalarOperatorInstance<ReLUOperation>(ReLUOperation());
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::relu_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<ReLUOperation>>(ReLUOperation());
}

float nnf::autodiff::SigmoidOperation::forward(float input)
{
	return 1.f / (1.f + std::exp(-input));
}

float nnf::autodiff::SigmoidOperation::back(float input)
{
	return std::exp(-input) / ((1.f + std::exp(-input)) * (1.f + std::exp(-input)));
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::SigmoidOperation> nnf::autodiff::sigmoid()
{
	return UnaryScalarOperatorInstance<SigmoidOperation>(SigmoidOperation());
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::sigmoid_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<SigmoidOperation>>(SigmoidOperation());
}

float nnf::autodiff::TanhOperation::forward(float input)
{
	return std::tanh(input);
}

float nnf::autodiff::TanhOperation::back(float input)
{
	return 1 - std::tanh(input) * std::tanh(input);
}

nnf::autodiff::UnaryScalarOperatorInstance<nnf::autodiff::TanhOperation> nnf::autodiff::tanh()
{
	return UnaryScalarOperatorInstance<TanhOperation>(TanhOperation());
}

std::unique_ptr<nnf::autodiff::UnaryScalarOperator> nnf::autodiff::tanh_operator()
{
	return std::make_unique<UnaryScalarOperatorInstance<TanhOperation>>(TanhOperation());
}
