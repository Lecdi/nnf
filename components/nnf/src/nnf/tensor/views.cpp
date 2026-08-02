#include <nnf/tensor/views.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <nnf/tensor/tensor_base.hpp>
#include <nnf/tensor/indexing.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

nnf::TensorPositionView::TensorPositionView(std::shared_ptr<TensorData> data, usize index)
	: data_{ data }, offset_{ index }
{
}

bool nnf::TensorPositionView::is_empty() const
{
	return data_->values.size() == 0;
}

bool nnf::TensorPositionView::is_detached() const
{
	return data_.use_count() == 1;
}

nnf::TensorPositionView::operator float() const
{
	if (is_empty()) throw std::logic_error("Attempt to convert empty TensorPositionView to float");
	return data_->values[offset_];
}

nnf::TensorPositionView &nnf::TensorPositionView::clear()
{
	data_ = std::make_shared<TensorData>();
	offset_ = 0;
	return *this;
}

nnf::TensorPositionView &nnf::TensorPositionView::detatch()
{
	if (is_detached()) return *this;
	if (is_empty()) return clear();
	data_ = std::make_shared<TensorData>(Vector<float>{ data_->values[offset_] });
	offset_ = 0;
	return *this;
}

nnf::TensorPositionView nnf::TensorPositionView::copy() const
{
	TensorPositionView result = *this;
	result.detatch();
	return result;
}

nnf::TensorPositionView &nnf::TensorPositionView::set(float other)
{
	if (is_empty()) throw std::logic_error("Attempt to set value of empty TensorPositionView");
	data_->values[offset_] = other;
	return *this;
}

nnf::TensorPositionView &nnf::TensorPositionView::apply_inplace(const std::function<float(float)> &func)
{
	set(func(*this));
	return *this;
}

nnf::TensorPositionView &nnf::TensorPositionView::operator+=(float other)
{
	if (is_empty()) throw std::logic_error("Attempt to set value of empty TensorPositionView");
	data_->values[offset_] += other;
	return *this;
}

nnf::TensorPositionView &nnf::TensorPositionView::operator-=(float other)
{
	if (is_empty()) throw std::logic_error("Attempt to set value of empty TensorPositionView");
	data_->values[offset_] -= other;
	return *this;
}

nnf::TensorPositionView &nnf::TensorPositionView::operator*=(float other)
{
	if (is_empty()) throw std::logic_error("Attempt to set value of empty TensorPositionView");
	data_->values[offset_] *= other;
	return *this;
}

nnf::TensorPositionView &nnf::TensorPositionView::operator/=(float other)
{
	if (is_empty()) throw std::logic_error("Attempt to set value of empty TensorPositionView");
	data_->values[offset_] /= other;
	return *this;
}

nnf::TensorGatherView::TensorGatherView(
	std::shared_ptr<TensorData> data,
	VectorInput<usize> original_dims,
	VectorInput<usize> strides,
	usize offset,
	usize gather_dim,
	VectorInput<usize> gather_indices
)
	: TensorLikeWithDims{ data, std::move(original_dims.values) }, strides_{ std::move(strides.values) },
	offset_{ offset }, gathered_dim_{ gather_dim }, gathered_indices_{ std::move(gather_indices.values) }
{
	if (gather_dim >= rank())
		throw std::out_of_range("Dimension to gather does not exist");
	if (gathered_indices_.empty())
		throw std::invalid_argument("No indices to gather");
	if (*std::max_element(gathered_indices_.begin(), gathered_indices_.end()) >= dims_[gathered_dim_])
		throw std::out_of_range("Some indices are out of range for the given dimension");
	dims_[gathered_dim_] = gathered_indices_.size();
}

nnf::TensorPositionView nnf::TensorGatherView::at(TensorSingleIndexView index) const
{
	if (is_empty()) throw std::logic_error("Attempt to get value from empty TensorGatherView");
	usize new_offset = offset_;
	usize dimension_position = 0;
	for (usize i = 0; i < rank(); ++i)
	{
		if (-index[i] > static_cast<ssize>(dims_[i])) throw std::out_of_range("Index out of range (too negative)");
		else if (index[i] < static_cast<ssize>(0)) dimension_position = index[i] + dims_[i];
		else if (index[i] < static_cast<ssize>(dims_[i])) dimension_position = index[i];
		else throw std::out_of_range("Index out of range (too positive)");
		if (i == gathered_dim_)
			new_offset += gathered_indices_[dimension_position] * strides_[i];
		else
			new_offset += dimension_position * strides_[i];
	}
	return TensorPositionView(data_, new_offset);
}
