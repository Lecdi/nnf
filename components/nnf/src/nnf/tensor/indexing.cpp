#include <nnf/tensor/indexing.hpp>

#include <utility>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

nnf::TensorSingleIndexIterator::TensorSingleIndexIterator(VectorInput<usize> dims)
	: dims_{ std::move(dims.values) }, current_{ Vector<ssize>(dims_.size()) }, finished_{ false }
{
}

nnf::TensorSingleIndexIterator::TensorSingleIndexIterator() : finished_{ true }
{
}

nnf::TensorSingleIndexIterator::value_type nnf::TensorSingleIndexIterator::operator*() const
{
	return current_;
}

nnf::TensorSingleIndexIterator &nnf::TensorSingleIndexIterator::operator++()
{
	if (finished_) return *this;
	for (usize i = dims_.size(); i-- > 0; )
	{
		++current_[i];
		if (current_[i] < dims_[i]) return *this;
		current_[i] = 0;
	}
	finished_ = true;
	return *this;
}

nnf::TensorSingleIndexIterator nnf::TensorSingleIndexIterator::operator++(int)
{
	TensorSingleIndexIterator copy = *this;
	++*this;
	return copy;
}

bool nnf::operator==(const TensorSingleIndexIterator &lhs, const TensorSingleIndexIterator &rhs)
{
	return lhs.finished_ == rhs.finished_;
}

nnf::TensorSingleIndexRange::TensorSingleIndexRange(VectorInput<usize> dims)
	: dims_{ std::move(dims.values) }
{
}

nnf::TensorSingleIndexIterator nnf::TensorSingleIndexRange::begin() const
{
	return TensorSingleIndexIterator{ dims_ };
}

nnf::TensorSingleIndexIterator nnf::TensorSingleIndexRange::end() const
{
	return TensorSingleIndexIterator{};
}

nnf::TensorRawDataIndexIterator::TensorRawDataIndexIterator(
	VectorInput<usize> dims, VectorInput<usize> strides, usize offset
)
	: dims_{ std::move(dims.values) }, strides_{ std::move(strides.values) }, current_raw_data_index_{ offset },
	current_position_{ Vector<usize>(dims_.size()) }, finished_{ false }
{
}

nnf::TensorRawDataIndexIterator::TensorRawDataIndexIterator() : finished_{ true }
{
}

nnf::TensorRawDataIndexIterator::value_type nnf::TensorRawDataIndexIterator::operator*() const
{
	return current_raw_data_index_;
}

nnf::TensorRawDataIndexIterator &nnf::TensorRawDataIndexIterator::operator++()
{
	if (finished_) return *this;
	for (usize i = dims_.size(); i-- > 0; )
	{
		++current_position_[i];
		current_raw_data_index_ += strides_[i];
		if (current_position_[i] < dims_[i])
			return *this;
		current_position_[i] = 0;
		current_raw_data_index_ -= dims_[i] * strides_[i];
	}
	finished_ = true;
	return *this;
}

nnf::TensorRawDataIndexIterator nnf::TensorRawDataIndexIterator::operator++(int)
{
	TensorRawDataIndexIterator copy = *this;
	++*this;
	return copy;
}

bool nnf::operator==(const TensorRawDataIndexIterator &lhs, const TensorRawDataIndexIterator &rhs)
{
	return lhs.finished_ == rhs.finished_;
}

nnf::TensorRawDataIndexRange::TensorRawDataIndexRange(
	VectorInput<usize> dims, VectorInput<usize> strides, usize offset
)
	: dims_{ std::move(dims.values) }, strides_{ std::move(strides.values) }, offset_{ offset }
{
}

nnf::TensorRawDataIndexIterator nnf::TensorRawDataIndexRange::begin() const
{
	return TensorRawDataIndexIterator{ dims_, strides_, offset_ };
}

nnf::TensorRawDataIndexIterator nnf::TensorRawDataIndexRange::end() const
{
	return TensorRawDataIndexIterator{};
}
