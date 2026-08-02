#include <nnf/tensor/tensor.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <nnf/tensor/tensor_base.hpp>
#include <nnf/tensor/indexing.hpp>
#include <nnf/tensor/views.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/math.hpp>
#include <nnf/utils/vector.hpp>

// Construction

nnf::Tensor::Tensor(VectorInput<usize> dims)
	: TensorLikeWithDims{ std::make_shared<TensorData>(), std::move(dims.values) },
	strides_{ calculate_contiguous_strides(dims_) }, offset_{ 0 }
{
	require_dims_no_zeroes();
}

nnf::Tensor nnf::Tensor::from_data(VectorInput<usize> dims, VectorInput<float> data)
{
	auto use_data = std::make_shared<TensorData>(Vector<float>(std::move(data.values)));
	Tensor result{ std::move(dims), use_data };
	if (result.data_->values.size() != result.num_elems())
		throw std::invalid_argument("Data has wrong size for tensor dimensions");
	return result;
}

nnf::Tensor nnf::Tensor::zeroes(VectorInput<usize> dims)
{
	auto use_data = std::make_shared<TensorData>(Vector<float>(calculate_num_elems(dims.values)));
	return Tensor{ std::move(dims), use_data };
}

nnf::Tensor nnf::Tensor::normals(VectorInput<usize> dims, float mean, float stddev)
{
	Tensor result{ std::move(dims) };
	result.make_normal(mean, stddev);
	return result;
}

// Inspection

bool nnf::Tensor::is_contiguous() const
{
	if (data_->values.size() != 0 && data_->values.size() != num_elems()) return false;
	if (strides_ != calculate_contiguous_strides(dims_)) return false;
	if (offset_ != 0) return false;
	return true;
}

nnf::usize nnf::Tensor::num_elems() const
{
	return calculate_num_elems(dims_);
}

nnf::Vector<nnf::usize> nnf::Tensor::get_strides() const
{
	return strides_;
}

nnf::VectorView<nnf::usize> nnf::Tensor::view_strides() const
{
	return strides_;
}

nnf::usize nnf::Tensor::offset() const
{
	return offset_;
}

nnf::Vector<float> nnf::Tensor::get_raw_data() const
{
	if (is_empty()) throw std::logic_error("Attempt to get raw data of empty tensor");
	return data_->values;
}

nnf::VectorView<float> nnf::Tensor::view_raw_data() const
{
	if (is_empty()) throw std::logic_error("Attempt to view raw data of empty tensor");
	return data_->values;
}

nnf::Vector<float> nnf::Tensor::get_contiguous_data() const
{
	if (is_empty()) throw std::logic_error("Attempt to get contiguous data of empty tensor");
	auto contiguous_data = Vector<float>(num_elems());
	usize contiguous_index = 0;
	for (const auto &position : positions())
	{
		contiguous_data[contiguous_index] = at(position);
		++contiguous_index;
	}
	return contiguous_data;
}

// Static helpers

nnf::usize nnf::Tensor::calculate_num_elems(VectorView<usize> dims)
{
	return std::accumulate(dims.begin(), dims.end(), usize(1), std::multiplies<usize>{});
}

nnf::Vector<nnf::usize> nnf::Tensor::calculate_contiguous_strides(VectorView<usize> dims)
{
	auto strides = Vector<usize>(dims.size());
	usize current_stride = 1;
	for (usize i = dims.size(); i-- > 0; )
	{
		strides[i] = current_stride;
		current_stride *= dims[i];
	}
	return strides;
}

std::array<nnf::usize, 2> nnf::Tensor::get_num_unfold2d_strides(
	usize dim0_h,
	usize dim1_w,
	usize kernel_h,
	usize kernel_w,
	usize stride_h,
	usize stride_w
)
{
	if (kernel_h >= dim0_h + 1 || kernel_w >= dim1_w + 1)
		throw std::logic_error("Tensor input to unfold2d is too small and output would be empty");
	if (kernel_h == 0 || kernel_w == 0)
		throw std::invalid_argument("Kernel size of 0 for unfold2d is invalid");
	
	usize h_strides = (dim0_h + 1 - kernel_h) / stride_h + ((dim0_h + 1 - kernel_h) % stride_h != 0);
	usize w_strides = (dim1_w + 1 - kernel_w) / stride_w + ((dim1_w + 1 - kernel_w) % stride_w != 0);
	return { h_strides, w_strides };
}

std::array<nnf::usize, 2> nnf::Tensor::get_unfold2d_dims(
	usize dim0_h,
	usize dim1_w,
	usize kernel_h,
	usize kernel_w,
	usize stride_h,
	usize stride_w
)
{
	auto [h_strides, w_strides] = get_num_unfold2d_strides(
		dim0_h,
		dim1_w,
		kernel_h,
		kernel_w,
		stride_h,
		stride_w
	);
	return { h_strides * w_strides, kernel_h * kernel_w };
}

std::array<nnf::usize, 3> nnf::Tensor::get_conv2d_dims(
	usize lhs_dim0_h,
	usize lhs_dim1_w,
	usize kernels_dim0_num_kernels,
	usize kernels_dim1_h,
	usize kernels_dim2_w,
	usize stride_h,
	usize stride_w
)
{
	auto [h_strides, w_strides] = get_num_unfold2d_strides(
		lhs_dim0_h,
		lhs_dim1_w,
		kernels_dim1_h,
		kernels_dim2_w,
		stride_h,
		stride_w
	);
	return { h_strides, w_strides, kernels_dim0_num_kernels };
}

// Data modification

nnf::Tensor &nnf::Tensor::clear()
{
	data_ = std::make_shared<TensorData>();
	strides_ = calculate_contiguous_strides(dims_);
	offset_ = 0;
	return *this;
}

nnf::Tensor &nnf::Tensor::detach()
{
	if (is_detached()) return *this;
	if (is_empty()) return clear();
	force_detach_contiguous();
	return *this;
}

nnf::Tensor nnf::Tensor::copy() const
{
	Tensor result = *this;
	result.detach();
	return result;
}

nnf::Tensor &nnf::Tensor::require_contiguous()
{
	if (is_contiguous()) return *this;
	if (!is_detached()) throw std::logic_error("Attempt to make non-detached tensor contiguous");
	force_detach_contiguous();
	return *this;
}

nnf::Tensor &nnf::Tensor::set_raw_data(VectorInput<float> data)
{
	data_->values = std::move(data.values);
	return *this;
}

nnf::Tensor &nnf::Tensor::set_from_contiguous_data(VectorView<float> data)
{
	make_zero();
	usize contiguous_index = 0;
	for (const auto &position : positions())
	{
		at(position).set(data[contiguous_index]);
		++contiguous_index;
	}
	return *this;
}

nnf::Tensor &nnf::Tensor::set_from_contiguous_data(VectorInitializer<float> data)
{
	Vector<float> contiguous_data = std::move(data);
	return set_from_contiguous_data(contiguous_data);
}

nnf::Tensor &nnf::Tensor::set(const TensorLike &other)
{
	if (!other.dims_compatible(view_dims()))
		throw std::invalid_argument("Other tensor has incorrect dimensions");
	if (is_empty())
	{
		if (is_detached()) data_->values.resize(num_elems());
		else throw std::logic_error("Attempt to set value of empty non-detached tensor");
	}
	for (const auto &position : positions())
		at(position).set(other.at(position));
	return *this;
}

nnf::Tensor &nnf::Tensor::make_zero()
{
	if (is_empty())
	{
		if (is_detached())
		{
			data_->values.resize(num_elems());
			return *this;
		}
		else throw std::logic_error("Attempt to set value of empty non-detached tensor");
	}
	for (const auto &position : positions())
		at(position).set(0.f);
	return *this;
}

nnf::Tensor &nnf::Tensor::make_normal(float mean, float stddev)
{
	if (is_empty())
	{
		if (is_detached()) data_->values.resize(num_elems());
		else throw std::logic_error("Attempt to set value of empty non-detached tensor");
	}
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<float> d{ mean, stddev };
	for (const auto &position : positions())
		at(position).set(d(gen));
	return *this;
}

nnf::Tensor &nnf::Tensor::apply_inplace(const std::function<float(float)> &func)
{
	for (const auto &position : positions())
	{
		at(position).apply_inplace(func);
	}
	return *this;
}

nnf::TensorPositionView nnf::Tensor::raw_data_at(usize raw_data_index) const
{
	return TensorPositionView(data_, raw_data_index);
}

nnf::TensorRawDataIndexRange nnf::Tensor::raw_data_positions() const
{
	return TensorRawDataIndexRange(dims_, strides_, offset_);
}

// Dimension modification

nnf::Tensor &nnf::Tensor::flatten_inplace()
{
	require_contiguous();
	dims_ = Vector<usize>({ num_elems() });
	strides_ = Vector<usize>({ 1 });
	offset_ = 0;
	return *this;
}

nnf::Tensor nnf::Tensor::flattened() const
{
	Tensor result = *this;
	result.flatten_inplace();
	return result;
}

nnf::Tensor &nnf::Tensor::reshape_inplace(VectorInput<usize> new_dims)
{
	require_contiguous();
	auto prev_num_elems = num_elems();
	dims_ = std::move(new_dims.values);
	strides_ = calculate_contiguous_strides(dims_);
	if (num_elems() != prev_num_elems)
		throw std::invalid_argument("New dimensions have different total number of elements");
	return *this;
}

nnf::Tensor nnf::Tensor::reshaped(VectorInput<usize> new_dims) const
{
	Tensor result = *this;
	result.reshape_inplace(std::move(new_dims.values));
	return result;
}

nnf::Tensor &nnf::Tensor::transpose_inplace(usize dim1, usize dim2)
{
	if (dim1 >= rank() || dim2 >= rank())
		throw std::out_of_range("Dimension to transpose is >= rank of tensor so does not exist");
	if (dim1 == dim2)
		throw std::invalid_argument("Dimensions to transpose are the same");
	std::swap(dims_[dim1], dims_[dim2]);
	std::swap(strides_[dim1], strides_[dim2]);
	return *this;
}

nnf::Tensor &nnf::Tensor::transpose_inplace()
{
	if (rank() != 2)
		throw std::logic_error("Transpose for tensor not of rank 2 must specify transpose dimensions");
	return transpose_inplace(0, 1);
}

nnf::Tensor nnf::Tensor::transposed(usize dim1, usize dim2) const
{
	Tensor result = *this;
	result.transpose_inplace(dim1, dim2);
	return result;
}

nnf::Tensor nnf::Tensor::transposed() const
{
	if (rank() != 2)
		throw std::logic_error("Transpose for tensor not of rank 2 must specify transpose dimensions");
	return transposed(0, 1);
}

nnf::Tensor &nnf::Tensor::permute_inplace(VectorInput<usize> dims)
{
	if (dims.values.size() != rank())
		throw std::invalid_argument("Permutation has wrong number of elements");
	auto seen = Vector<bool>(rank(), false);
	auto new_dims = Vector<usize>(rank());
	auto new_strides = Vector<usize>(rank());
	for (usize i = 0; i < rank(); ++i)
	{
		if (dims.values[i] >= rank())
			throw std::invalid_argument("Permutation contains index too large");
		if (seen[dims.values[i]])
			throw std::invalid_argument("Permutation contains same index twice");
		seen[dims.values[i]] = true;
		new_dims[i] = dims_[dims.values[i]];
		new_strides[i] = strides_[dims.values[i]];
	}
	dims_ = std::move(new_dims);
	strides_ = std::move(new_strides);
	return *this;
}

nnf::Tensor nnf::Tensor::permuted(VectorInput<usize> dims) const
{
	Tensor result = *this;
	result.permute_inplace(std::move(dims.values));
	return result;
}

// Indexing

nnf::TensorPositionView nnf::Tensor::at(TensorSingleIndexView index) const
{
	if (index.size() != rank()) throw std::invalid_argument("Incorrect number of indices");
	usize new_offset = offset_;
	usize dimension_position = 0;
	for (usize i = 0; i < rank(); ++i)
	{
		if (-index[i] > static_cast<ssize>(dims_[i])) throw std::out_of_range("Index out of range (too negative)");
		else if (index[i] < static_cast<ssize>(0)) dimension_position = index[i] + dims_[i];
		else if (index[i] < static_cast<ssize>(dims_[i])) dimension_position = index[i];
		else throw std::out_of_range("Index out of range (too positive)");
		new_offset += dimension_position * strides_[i];
	}
	return TensorPositionView(data_, new_offset);
}

nnf::Tensor nnf::Tensor::operator()(TensorMultiIndexView index) const
{
	auto new_dims = Vector<usize>(rank());
	auto new_strides = Vector<usize>(rank());
	usize new_offset = offset_;

	bool past_ellipsis = false;
	usize current_dim = 0;
	ProcessedVectorMultiIndex index_for_current_dim{};
	for (const auto &index_arg : index)
	{
		if (!index_arg.has_value()) // Ellipsis
		{
			if (past_ellipsis) throw std::invalid_argument("Multi-index has multiple ellipses");
			past_ellipsis = true;
			if (index.size() - 1 >= rank()) continue;
			for (usize last_dim = rank() - (index.size() - 1) + current_dim; current_dim < last_dim; ++current_dim)
			{
				new_dims[current_dim] = dims_[current_dim];
				new_strides[current_dim] = strides_[current_dim];
			}
		}
		else
		{
			if (current_dim >= rank()) throw std::out_of_range("Multi-index has too many arguments");
			index_for_current_dim = process_vector_multi_index(index_arg.value(), dims_[current_dim]);
			new_dims[current_dim] = index_for_current_dim.num_values;
			new_strides[current_dim] = index_for_current_dim.step * strides_[current_dim];
			new_offset += index_for_current_dim.start * strides_[current_dim];
			++current_dim;
		}
	}
	if (current_dim != rank())
		throw std::out_of_range("Multi-index has too few arguments and no ellipsis");
	return Tensor{ std::move(new_dims), std::move(new_strides), new_offset, data_ };
}

nnf::Tensor nnf::Tensor::operator()(TensorMultiIndexInitializer index_initializer) const
{
	auto index = Vector<TensorMultiIndexArg>{ index_initializer };
	return (*this)(index);
}

nnf::TensorGatherView nnf::Tensor::gather(usize gather_dim, VectorInput<usize> gather_indices) const
{
	return TensorGatherView(
		data_, dims_, strides_, offset_, gather_dim, std::move(gather_indices.values)
	);
}

// Operations

nnf::Tensor nnf::concat(const Tensor &lhs, const Tensor &rhs, usize concat_dim)
{
	if (lhs.rank() != rhs.rank())
		throw std::invalid_argument("Attempt to concatenate tensors of different rank");
	auto new_dims = Vector<usize>(lhs.view_dims().begin(), lhs.view_dims().end());
	TensorMultiIndex rhs_part_index{};
	rhs_part_index.reserve(lhs.rank());
	for (usize i = 0; i < lhs.rank(); ++i)
	{
		if (i == concat_dim)
		{
			new_dims[i] += rhs.view_dims()[i];
			rhs_part_index.push_back(VectorMultiIndex(lhs.view_dims()[i], 0, 1));
		}
		else
		{
			if (new_dims[i] != rhs.view_dims()[i])
				throw std::invalid_argument("Tensors to concatenate have incompatible shapes");
			rhs_part_index.push_back(VectorMultiIndex(0, 0, 1));
		}
	}
	Tensor result{ new_dims };
	for (const auto &position : lhs.positions())
		result.at(position).set(lhs.at(position));
	Tensor result_rhs_part = result(std::move(rhs_part_index));
	for (const auto &position : rhs.positions())
		result_rhs_part.at(position).set(rhs.at(position));
	return result;
}

nnf::Tensor &nnf::Tensor::operator+=(const Tensor &other)
{
	if (!other.dims_compatible(view_dims()))
		throw std::invalid_argument("Other tensor has incorrect dimensions");

	usize n = num_elems();
	auto it_this = raw_data_positions().begin();
	auto it_other = other.raw_data_positions().begin();

	for (usize i = 0; i < n; ++i)
	{
		data_->values[*it_this] += other.data_->values[*it_other];
		++it_this;
		++it_other;
	}

	return *this;
}

nnf::Tensor &nnf::Tensor::operator+=(const TensorLike &other)
{
	if (!other.dims_compatible(view_dims()))
		throw std::invalid_argument("Other tensor has incorrect dimensions");
	for (const auto &position : positions())
		at(position) += other.at(position);
	return *this;
}

nnf::Tensor nnf::operator+(const Tensor &lhs, const Tensor &rhs)
{
	auto result = lhs.copy();
	result += rhs;
	return result;
}

nnf::Tensor nnf::operator+(const Tensor &lhs, const TensorLike &rhs)
{
	auto result = lhs.copy();
	result += rhs;
	return result;
}

nnf::Tensor nnf::operator+(const TensorLike &lhs, const Tensor &rhs)
{
	auto result = rhs.copy();
	result += lhs;
	return result;
}

nnf::Tensor &nnf::Tensor::operator-=(const Tensor &other)
{
	if (!other.dims_compatible(view_dims()))
		throw std::invalid_argument("Other tensor has incorrect dimensions");

	usize n = num_elems();
	auto it_this = raw_data_positions().begin();
	auto it_other = other.raw_data_positions().begin();

	for (usize i = 0; i < n; ++i)
	{
		data_->values[*it_this] -= other.data_->values[*it_other];
		++it_this;
		++it_other;
	}

	return *this;
}

nnf::Tensor &nnf::Tensor::operator-=(const TensorLike &other)
{
	if (!other.dims_compatible(view_dims()))
		throw std::invalid_argument("Other tensor has incorrect dimensions");
	for (const auto &position : positions())
		at(position) -= other.at(position);
	return *this;
}

nnf::Tensor nnf::operator-(const Tensor &lhs, const Tensor &rhs)
{
	auto result = lhs.copy();
	result -= rhs;
	return result;
}

nnf::Tensor nnf::operator-(const Tensor &lhs, const TensorLike &rhs)
{
	auto result = lhs.copy();
	result -= rhs;
	return result;
}

nnf::Tensor nnf::operator-(const TensorLike &lhs, const Tensor &rhs)
{
	auto result = rhs.copy();
	result *= -1;
	result += lhs;
	return result;
}

nnf::Tensor &nnf::Tensor::operator*=(float scalar)
{
	for (const auto &position : positions())
		at(position) *= scalar;
	return *this;
}

nnf::Tensor nnf::operator*(const Tensor &lhs, float scalar)
{
	auto result = lhs.copy();
	result *= scalar;
	return result;
}

nnf::Tensor nnf::operator*(float scalar, const Tensor &rhs)
{
	auto result = rhs.copy();
	result *= scalar;
	return result;
}

nnf::Tensor &nnf::Tensor::operator/=(float scalar)
{
	for (const auto &position : positions())
		at(position) /= scalar;
	return *this;
}

nnf::Tensor nnf::operator/(const Tensor &lhs, float scalar)
{
	auto result = lhs.copy();
	result /= scalar;
	return result;
}

nnf::Tensor nnf::outer_product(const Tensor &lhs, const Tensor &rhs)
{
	Vector<usize> new_dims{};
	new_dims.reserve(lhs.rank() + rhs.rank());
	new_dims.insert(new_dims.end(), lhs.dims_.begin(), lhs.dims_.end());
	new_dims.insert(new_dims.end(), rhs.dims_.begin(), rhs.dims_.end());

	auto new_data = Vector<float>(Tensor::calculate_num_elems(new_dims));

	usize new_raw_data_position = 0;
	for (const usize rhs_raw_data_position : rhs.raw_data_positions())
	{
		for (const usize lhs_raw_data_position : lhs.raw_data_positions())
		{
			new_data[new_raw_data_position] =
				lhs.data_->values[lhs_raw_data_position] * rhs.data_->values[rhs_raw_data_position];
			++new_raw_data_position;
		}
	}

	return Tensor::from_data(std::move(new_dims), std::move(new_data));
}

nnf::Tensor nnf::Tensor::contracted(VectorInput<std::pair<usize, usize>> contractions) const
{
	std::unordered_set<usize> seen{};
	Vector<usize> new_dims = dims_;
	for (const auto &dims_pair : contractions.values)
	{
		if (seen.contains(dims_pair.first))
			throw std::invalid_argument("Same dim appears twice for contraction");
		if (dims_pair.first >= rank())
			throw std::out_of_range("Dim for contraction does not exist");
		seen.insert(dims_pair.first);
		new_dims[dims_pair.first] = 1;

		if (seen.contains(dims_pair.second))
			throw std::invalid_argument("Same dim appears twice for contraction");
		if (dims_pair.second >= rank())
			throw std::out_of_range("Dim for contraction does not exist");
		seen.insert(dims_pair.second);
		new_dims[dims_pair.second] = 1;
	}
	auto result = zeroes(new_dims);
	for (const auto &position : result.positions())
	{
		perform_contractions(
			TensorSingleIndex(position.begin(), position.end()), position, result, contractions.values
		);
	}
	remove_at_indices(seen, new_dims);
	result.reshape_inplace(std::move(new_dims));
	return result;
}

nnf::Tensor nnf::contract(const Tensor &lhs, const Tensor &rhs, VectorInput<std::pair<usize, usize>> contractions)
{
	Vector<std::pair<usize, usize>> dims_after_outer_product = std::move(contractions.values);
	for (auto &dims_pair : dims_after_outer_product)
	{
		dims_pair.second += lhs.rank();
	}
	return outer_product(lhs, rhs).contracted(std::move(dims_after_outer_product));
}

nnf::Tensor nnf::dot(const Tensor &lhs, const Tensor &rhs)
{
	if (lhs.rank() != 1)
		throw std::logic_error("[nnf::Tensor::dot] lhs is not rank 1");
	if (rhs.rank() != 1)
		throw std::logic_error("[nnf::Tensor::dot] rhs is not rank 1");
	if (lhs.view_dims()[0] != rhs.view_dims()[0])
		throw std::logic_error("[nnf::Tensor::dot] lhs and rhs vectors are not the same length");
	return contract(lhs, rhs, { {0, 0} });
}

nnf::Tensor nnf::matvecmul(const Tensor &lhs, const Tensor &rhs)
{
	if (lhs.rank() != 2)
		throw std::logic_error("[nnf::Tensor::matvecmul] lhs is not rank 2");
	if (rhs.rank() != 1)
		throw std::logic_error("[nnf::Tensor::matvecmul] rhs is not rank 1");
	if (lhs.view_dims()[1] != rhs.view_dims()[0])
		throw std::logic_error("[nnf::Tensor::matvecmul] lhs and rhs dimensions are not compatible");

	usize I = lhs.view_dims()[0];
	usize J = lhs.view_dims()[1];
	auto it_lhs = lhs.raw_data_positions().begin();
	auto range_rhs = rhs.raw_data_positions();

	auto result = Tensor::zeroes({ I });
	for (usize i = 0; i < I; ++i)
	{
		auto it_rhs = range_rhs.begin();
		for (usize j = 0; j < J; ++j)
		{
			result.data_->values[i] += lhs.data_->values[*it_lhs] * rhs.data_->values[*it_rhs];
			++it_lhs;
			++it_rhs;
		}
	}
	return result;
}

nnf::Tensor nnf::matmatmul(const Tensor &lhs, const Tensor &rhs)
{
	if (lhs.rank() != 2)
		throw std::logic_error("[nnf::Tensor::matmatmul] lhs is not rank 2");
	if (rhs.rank() != 2)
		throw std::logic_error("[nnf::Tensor::matmatmul] rhs is not rank 2");
	if (lhs.view_dims()[1] != rhs.view_dims()[0])
		throw std::logic_error("[nnf::Tensor::matmatmul] lhs and rhs dimensions are not compatible");

	usize I = lhs.view_dims()[0];
	usize J = lhs.view_dims()[1];
	usize K = rhs.view_dims()[1];
	auto result = Tensor::zeroes({ I, K });

	usize lhs_stride_i = lhs.view_strides()[0];
	usize lhs_stride_j = lhs.view_strides()[1];
	usize rhs_stride_j = rhs.view_strides()[0];
	usize rhs_stride_k = rhs.view_strides()[1];

	usize raw_data_position_lhs = lhs.offset_;
	usize raw_data_position_rhs = rhs.offset_;

	for (usize i = 0; i < I; ++i)
	{
		for (usize j = 0; j < J; ++j)
		{
			for (usize k = 0; k < K; ++k)
			{
				result.data_->values[i * K + k] +=
					lhs.data_->values[raw_data_position_lhs] * rhs.data_->values[raw_data_position_rhs];
				raw_data_position_rhs += rhs_stride_k;
			}
			raw_data_position_lhs += lhs_stride_j;
			raw_data_position_rhs -= K * rhs_stride_k;
			raw_data_position_rhs += rhs_stride_j;
		}
		raw_data_position_lhs -= J * lhs_stride_j;
		raw_data_position_lhs += lhs_stride_i;
		raw_data_position_rhs -= J * rhs_stride_j;
	}
	return result;
}

nnf::Tensor nnf::matmul(const Tensor &lhs, const Tensor &rhs)
{
	if (rhs.rank() == 1)
		return matvecmul(lhs, rhs);
	return matmatmul(lhs, rhs);
}

nnf::Tensor nnf::Tensor::unfold2d(
	usize kernel_h,
	usize kernel_w,
	usize stride_h,
	usize stride_w
) const
{
	if (rank() != 2)
		throw std::logic_error("[nnf::Tensor::unfold2d] tensor is not rank 2");

	auto [h_strides, w_strides] = get_num_unfold2d_strides(
		view_dims()[0], view_dims()[1], kernel_h, kernel_w, stride_h, stride_w
	);
	
	std::array<usize, 2> result_dims = get_unfold2d_dims(
		view_dims()[0], view_dims()[1], kernel_h, kernel_w, stride_h, stride_w
	);
	
	auto result_data = Vector<float>(result_dims[0] * result_dims[1]);

	usize result_data_index = 0;
	ssize row, y_base, y, col, x_base, x;
	for (row = 0; row < h_strides; ++row)
	{
		y_base = row * stride_h;
		for (col = 0; col < w_strides; ++col)
		{
			x_base = col * stride_w;
			for (y = 0; y < kernel_h; ++y)
			{
				for (x = 0; x < kernel_w; ++x)
				{
					result_data[result_data_index] = at(TensorSingleIndexInitializer{ y_base + y, x_base + x });
					++result_data_index;
				}
			}
		}
	}

	return from_data({ std::move(result_dims) }, std::move(result_data));
}

nnf::Tensor nnf::Tensor::fold2d(
	usize result_h,
	usize result_w,
	usize kernel_h,
	usize kernel_w,
	usize stride_h,
	usize stride_w
) const
{
	if (result_h == 0)
		throw std::invalid_argument("[nnf::Tensor::fold2d] height of result tensor given as 0 is invalid");
	if (result_w == 0)
		throw std::invalid_argument("[nnf::Tensor::fold2d] width of result tensor given as 0 is invalid");
	if (kernel_h == 0)
		throw std::invalid_argument("[nnf::Tensor::fold2d] height of kernel given as 0 is invalid");
	if (kernel_w == 0)
		throw std::invalid_argument("[nnf::Tensor::fold2d] width of kernel given as 0 is invalid");

	usize num_strides_h, num_strides_w;
	try
	{
		auto [H, W] = get_num_unfold2d_strides(
			result_h, result_w, kernel_h, kernel_w, stride_h, stride_w
		);
		num_strides_h = H;
		num_strides_w = W;
	}
	catch (const std::logic_error &e)
	{
		throw std::invalid_argument("[nnf::Tensor::fold2d] result dimensions are not possible for given arguments");
	}

	if (rank() != 2)
		throw std::logic_error("[nnf::Tensor::fold2d] tensor is not rank 2");
	if (view_dims()[0] != num_strides_h * num_strides_w)
		throw std::invalid_argument("[nnf::Tensor::fold2d] tensor height does not equal number of strides taken");
	if (view_dims()[1] != kernel_h * kernel_w)
		throw std::invalid_argument("[nnf::Tensor::fold2d] tensor width doesn not equal kernel height times width");

	auto result_data = Vector<float>(result_h * result_w);

	usize result_data_index = 0;
	usize this_data_index = offset_;

	usize this_stride_x_in_kernel = view_strides()[1];
	usize this_stride_y_in_kernel = view_strides()[1] * kernel_w;
	usize this_stride_x_in_stride = view_strides()[0];
	usize this_stride_y_in_stride = view_strides()[0] * num_strides_w;

	usize result_stride_x_in_kernel = 1;
	usize result_stride_y_in_kernel = result_w;
	usize result_stride_x_in_stride = stride_w;
	usize result_stride_y_in_stride = stride_h * result_w;

	for (usize y_in_stride = 0; y_in_stride < num_strides_h; ++y_in_stride)
	{
		for (usize x_in_stride = 0; x_in_stride < num_strides_w; ++x_in_stride)
		{
			for (usize y_in_kernel = 0; y_in_kernel < kernel_h; ++y_in_kernel)
			{
				for (usize x_in_kernel = 0; x_in_kernel < kernel_w; ++x_in_kernel)
				{
					result_data[result_data_index] += data_->values[this_data_index];

					result_data_index += result_stride_x_in_kernel;
					this_data_index += this_stride_x_in_kernel;
				}
				result_data_index -= kernel_w * result_stride_x_in_kernel;
				this_data_index -= kernel_w * this_stride_x_in_kernel;
				result_data_index += result_stride_y_in_kernel;
				this_data_index += this_stride_y_in_kernel;
			}
			result_data_index -= kernel_h * result_stride_y_in_kernel;
			this_data_index -= kernel_h * this_stride_y_in_kernel;
			result_data_index += result_stride_x_in_stride;
			this_data_index += this_stride_x_in_stride;
		}
		result_data_index -= num_strides_w * result_stride_x_in_stride;
		this_data_index -= num_strides_w * this_stride_x_in_stride;
		result_data_index += result_stride_y_in_stride;
		this_data_index += this_stride_y_in_stride;
	}

	return Tensor::from_data({ result_h, result_w }, std::move(result_data));
}

nnf::Tensor nnf::Tensor::conv2d(Tensor kernels, usize stride_h, usize stride_w) const
{
	if (rank() != 2)
		throw std::logic_error("[nnf::Tensor::conv2d] tensor is not rank 2");
	if (kernels.rank() != 3)
		throw std::logic_error("[nnf::Tensor::conv2d] kernels is not rank 3");

	usize num_kernels = kernels.view_dims()[0];
	usize kernel_h = kernels.view_dims()[1];
	usize kernel_w = kernels.view_dims()[2];

	Tensor result = matmatmul(
		unfold2d(kernel_h, kernel_w, stride_h, stride_w),
		kernels.reshaped({ num_kernels, kernel_h * kernel_w }).transposed()
	);

	return result.reshaped({ get_conv2d_dims(
		view_dims()[0],
		view_dims()[1],
		num_kernels,
		kernel_h,
		kernel_w,
		stride_h,
		stride_w
	) });
}

// Comparison

bool nnf::approx_eq(const Tensor &lhs, const Tensor &rhs, float epsilon)
{
	if (!std::equal(lhs.view_dims().begin(), lhs.view_dims().end(), rhs.view_dims().begin(), rhs.view_dims().end()))
		return false;
	for (const auto &position : lhs.positions())
		if (!float_approx_eq(lhs.at(position), rhs.at(position), epsilon))
			return false;
	return true;
}

// Reduction

float nnf::Tensor::sum() const
{
	return std::transform_reduce(
		positions().begin(), positions().end(),
		0.f, std::plus{},
		[this](const auto &position) { return static_cast<float>(at(position)); }
	);
}

float nnf::Tensor::mean() const
{
	return sum() / num_elems();
}

nnf::TensorSingleIndex nnf::Tensor::maxpos() const
{
	auto it = std::ranges::max_element(
		positions(),
		std::less{},
		[this](const auto &position) { return static_cast<float>(at(position)); }
	);
	auto index_view = *it;
	return TensorSingleIndex(index_view.begin(), index_view.end());
}

nnf::TensorSingleIndex nnf::Tensor::minpos() const
{
	auto it = std::ranges::max_element(
		positions(),
		std::greater{},
		[this](const auto &position) { return static_cast<float>(at(position)); }
	);
	auto index_view = *it;
	return TensorSingleIndex(index_view.begin(), index_view.end());
}

nnf::TensorPositionView nnf::Tensor::max() const
{
	return at(maxpos());
}

nnf::TensorPositionView nnf::Tensor::min() const
{
	return at(minpos());
}

// Private constructors

nnf::Tensor::Tensor(
	VectorInput<usize> dims,
	std::shared_ptr<TensorData> use_data
)
	: TensorLikeWithDims{ std::move(use_data), std::move(dims.values) },
	strides_{ calculate_contiguous_strides(dims_) }, offset_{ 0 }
{
	require_dims_no_zeroes();
}

nnf::Tensor::Tensor(
	VectorInput<usize> dims,
	VectorInput<usize> strides,
	usize offset,
	std::shared_ptr<TensorData> use_data
)
	: TensorLikeWithDims{ std::move(use_data), std::move(dims.values) },
	strides_{ std::move(strides.values) }, offset_{ offset }
{
	require_dims_no_zeroes();
}

// Private helpers

nnf::ProcessedVectorMultiIndex nnf::Tensor::process_vector_multi_index(const nnf::VectorMultiIndex &index, usize length)
{
	ProcessedVectorMultiIndex result{};
	usize end;

	if (-index.start > static_cast<ssize>(length)) throw std::out_of_range("Start index out of range (too negative)");
	else if (index.start < static_cast<ssize>(0)) result.start = index.start + length;
	else if (index.start < static_cast<ssize>(length)) result.start = index.start;
	else throw std::out_of_range("Start index out of range (too positive)");

	if (-index.end >= static_cast<ssize>(length)) throw std::out_of_range("End index out of range (too negative)");
	else if (index.end <= static_cast<ssize>(0)) end = index.end + length;
	else if (index.end <= static_cast<ssize>(length)) end = index.end;
	else throw std::out_of_range("End index out of range (too positive)");

	result.step = index.step;

	if (end <= result.start) return result;

	result.num_values = (end - result.start) / result.step + ((end - result.start) % result.step != 0);
	return result;
}

void nnf::Tensor::require_dims_no_zeroes() const
{
	if (std::any_of(dims_.begin(), dims_.end(), [](usize dim) { return dim == 0; }))
		throw std::invalid_argument("Dimension of size 0 is not allowed");
}

void nnf::Tensor::force_detach_contiguous()
{
	auto new_data = std::make_shared<TensorData>(Vector<float>(num_elems()));
	usize new_data_index = 0;
	for (const auto &position : positions())
	{
		new_data->values[new_data_index] = at(position);
		++new_data_index;
	}
	data_ = new_data;
	strides_ = calculate_contiguous_strides(dims_);
	offset_ = 0;
}

void nnf::Tensor::perform_contractions(
	TensorSingleIndex from_position,
	TensorSingleIndexView target_position,
	Tensor &target,
	Vector<std::pair<usize, usize>> contractions
) const
{
	if (contractions.empty())
	{
		target.at(target_position).set(at(from_position));
		return;
	}
	auto contraction = contractions[contractions.size() - 1];
	contractions.pop_back();
	if (dims_[contraction.first] != dims_[contraction.second])
		throw std::invalid_argument("Pair of dimensions to contract do not have same length");
	for (usize i = 0; i < dims_[contraction.first]; ++i)
	{
		from_position[contraction.first] = i;
		from_position[contraction.second] = i;
		perform_contractions(
			from_position, target_position, target, contractions
		);
	}
}
