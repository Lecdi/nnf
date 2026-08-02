#include <nnf/tensor/tensor_base.hpp>

#include <algorithm>
#include <memory>
#include <utility>
#include <nnf/tensor/indexing.hpp>
#include <nnf/tensor/views.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

nnf::TensorLike::TensorLike(std::shared_ptr<TensorData> data) : data_{ data } {}

bool nnf::TensorLike::is_empty() const
{
	return data_->values.size() == 0;
}

bool nnf::TensorLike::is_detached() const
{
	return data_.use_count() == 1;
}

nnf::TensorPositionView nnf::TensorLike::at(TensorSingleIndexInitializer index_initializer) const
{
	auto index = Vector<TensorSingleIndexArg>{ index_initializer };
	return at(index);
}

nnf::TensorLikeWithDims::TensorLikeWithDims(std::shared_ptr<TensorData> data, VectorInput<usize> dims)
	: TensorLike{ data }, dims_{ std::move(dims.values) }
{
}

nnf::usize nnf::TensorLikeWithDims::rank() const
{
	return dims_.size();
}

nnf::VectorView<nnf::usize> nnf::TensorLikeWithDims::view_dims() const
{
	return dims_;
}

nnf::Vector<nnf::usize> nnf::TensorLikeWithDims::get_dims() const
{
	return dims_;
}

nnf::TensorSingleIndexRange nnf::TensorLikeWithDims::positions() const
{
	return TensorSingleIndexRange{ dims_ };
}

bool nnf::TensorLikeWithDims::dims_compatible(VectorView<usize> dims) const
{
	return std::ranges::equal(dims, this->view_dims());
}
