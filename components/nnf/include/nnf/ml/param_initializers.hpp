#pragma once

#include <memory>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

namespace nnf::ml
{
    class ParamInitializer
	{
	public:
		virtual ~ParamInitializer() {}

		virtual void operator()(Tensor &params) const = 0;
	};

	class NormalInitializer : public ParamInitializer
	{
	public:
		explicit NormalInitializer(float mean, float stddev);

		void operator()(Tensor &params) const override;

	private:
		float mean_;
		float stddev_;
	};

	std::unique_ptr<ParamInitializer> normal_initializer(float mean, float stddev);

	class HeKaimingInitializer : public NormalInitializer
	{
	public:
		explicit HeKaimingInitializer(usize in);
	};

	std::unique_ptr<ParamInitializer> he_kaiming_initializer(usize in);

	class XavierGlorotInitializer : public NormalInitializer
	{
	public:
		explicit XavierGlorotInitializer(usize in, usize out);
	};

	std::unique_ptr<ParamInitializer> xavier_glorot_initializer(usize in, usize out);
}
