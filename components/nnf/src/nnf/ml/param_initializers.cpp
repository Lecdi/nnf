#include <nnf/ml/param_initializers.hpp>

#include <cmath>
#include <memory>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

nnf::ml::NormalInitializer::NormalInitializer(float mean, float stddev)
    : mean_{ mean }, stddev_{ stddev }
{
}

void nnf::ml::NormalInitializer::operator()(Tensor &params) const
{
    params.make_normal(mean_, stddev_);
}

std::unique_ptr<nnf::ml::ParamInitializer> nnf::ml::normal_initializer(float mean, float stddev)
{
	return std::make_unique<NormalInitializer>(mean, stddev);
}

nnf::ml::HeKaimingInitializer::HeKaimingInitializer(usize in)
    : NormalInitializer{ 0.f, std::sqrt(2.f / static_cast<float>(in)) }
{
}

std::unique_ptr<nnf::ml::ParamInitializer> nnf::ml::he_kaiming_initializer(usize in)
{
	return std::make_unique<HeKaimingInitializer>(in);
}

nnf::ml::XavierGlorotInitializer::XavierGlorotInitializer(usize in, usize out)
    : NormalInitializer{ 0.f, std::sqrt(2.f / static_cast<float>(in + out)) }
{
}

std::unique_ptr<nnf::ml::ParamInitializer> nnf::ml::xavier_glorot_initializer(usize in, usize out)
{
	return std::make_unique<XavierGlorotInitializer>(in, out);
}
