#pragma once

#include <memory>
#include <nnf/autodiff/base_operations.hpp>

namespace nnf::autodiff
{
	class IdentityOperation : public UnaryScalarOperation
	{
		float forward(float input) override { return input; }
		float back(float input) override { return 1; }
	};

	UnaryScalarOperatorInstance<IdentityOperation> identity();
	std::unique_ptr<UnaryScalarOperator> identity_operator();

	class SgnOperation : public UnaryScalarOperation
	{
		float forward(float input) override;
		float back(float input) override;
	};

	UnaryScalarOperatorInstance<SgnOperation> sgn();
	std::unique_ptr<UnaryScalarOperator> sgn_operator();

	class AbsOperation : public UnaryScalarOperation
	{
		float forward(float input) override;
		float back(float input) override;
	};

	UnaryScalarOperatorInstance<AbsOperation> abs();
	std::unique_ptr<UnaryScalarOperator> abs_operator();

	class PowOperation : public UnaryScalarOperation
	{
	public:
		PowOperation() = delete;
		explicit PowOperation(float exponent);

		float forward(float input) override;
		float back(float input) override;

	private:
		float exponent_;
	};

	UnaryScalarOperatorInstance<PowOperation> pow(float exponent);
	UnaryScalarOperatorInstance<PowOperation> sqrt();
	UnaryScalarOperatorInstance<PowOperation> square();
	std::unique_ptr<UnaryScalarOperator> pow_operator(float exponent);
	std::unique_ptr<UnaryScalarOperator> sqrt_operator();
	std::unique_ptr<UnaryScalarOperator> square_operator();

	class ExpOperation : public UnaryScalarOperation
	{
		float forward(float input) override;
		float back(float input) override;
	};

	UnaryScalarOperatorInstance<ExpOperation> exp();
	std::unique_ptr<UnaryScalarOperator> exp_operator();

	class LogOperation : public UnaryScalarOperation
	{
		float forward(float input) override;
		float back(float input) override;
	};

	UnaryScalarOperatorInstance<LogOperation> log();
	std::unique_ptr<UnaryScalarOperator> log_operator();

	class ReLUOperation : public UnaryScalarOperation
	{
		float forward(float input) override;
		float back(float input) override;
	};

	UnaryScalarOperatorInstance<ReLUOperation> relu();
	std::unique_ptr<UnaryScalarOperator> relu_operator();

	class SigmoidOperation : public UnaryScalarOperation
	{
		float forward(float input) override;
		float back(float input) override;
	};

	UnaryScalarOperatorInstance<SigmoidOperation> sigmoid();
	std::unique_ptr<UnaryScalarOperator> sigmoid_operator();

	class TanhOperation : public UnaryScalarOperation
	{
		float forward(float input) override;
		float back(float input) override;
	};

	UnaryScalarOperatorInstance<TanhOperation> tanh();
	std::unique_ptr<UnaryScalarOperator> tanh_operator();
}
