#pragma once

#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <nnf/tensor/tensor_base.hpp>
#include <nnf/tensor/indexing.hpp>
#include <nnf/tensor/views.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf
{
	class Tensor : public TensorLikeWithDims
	{
	public:
		// Construction
		Tensor() = delete;
		explicit Tensor(VectorInput<usize> dims);
		static Tensor from_data(VectorInput<usize> dims, VectorInput<float> data);
		static Tensor zeroes(VectorInput<usize> dims);
		static Tensor normals(VectorInput<usize> dims, float mean, float stddev);

		// Inspection
		bool is_contiguous() const;
		usize num_elems() const;
		Vector<usize> get_strides() const;
		VectorView<usize> view_strides() const;
		usize offset() const;
		Vector<float> get_raw_data() const;
		VectorView<float> view_raw_data() const;
		Vector<float> get_contiguous_data() const;
		//  -  TensorLike:         bool is_empty() const;
		//  -  TensorLike:         bool is_detached() const;
		//  -  TensorLikeWithDims: bool dims_compatible(VectorView<usize> dims) const;
		//  -  TensorLikeWithDims: usize rank() const;
		//  -  TensorLikeWithDims: Vector<usize> get_dims() const;
		//  -  TensorLikeWithDims: VectorView<usize> view_dims() const;

		// Static helpers
		static usize calculate_num_elems(
			VectorView<usize> dims
		);
		static Vector<usize> calculate_contiguous_strides(
			VectorView<usize> dims
		);
		static std::array<usize, 2> get_num_unfold2d_strides(
			usize dim0_h,
			usize dim1_w,
			usize kernel_h,
			usize kernel_w,
			usize stride_h = 1,
			usize stride_w = 1
		);
		static std::array<usize, 2> get_unfold2d_dims(
			usize dim0_h,
			usize dim1_w,
			usize kernel_h,
			usize kernel_w,
			usize stride_h = 1,
			usize stride_w = 1
		);
		static std::array<usize, 3> get_conv2d_dims(
			usize lhs_dim0_h,
			usize lhs_dim1_w,
			usize kernels_dim0_num_kernels,
			usize kernels_dim1_h,
			usize kernels_dim2_w,
			usize stride_h = 1,
			usize stride_w = 1
		);

		// Data modification
		Tensor &clear();
		Tensor &detach();
		Tensor copy() const;
		Tensor &require_contiguous();
		Tensor &set_raw_data(VectorInput<float> data);
		Tensor &set_from_contiguous_data(VectorView<float> data);
		Tensor &set_from_contiguous_data(VectorInitializer<float> data);
		Tensor &set(const TensorLike &other);
		Tensor &make_zero();
		Tensor &make_normal(float mean, float stddev);
		Tensor &apply_inplace(const std::function<float(float)> &func);
		TensorPositionView raw_data_at(usize raw_data_index) const;
		TensorRawDataIndexRange raw_data_positions() const;

		// Dimension modification
		Tensor &flatten_inplace();
		Tensor flattened() const;
		Tensor &reshape_inplace(VectorInput<usize> new_dims);
		Tensor reshaped(VectorInput<usize> new_dims) const;
		Tensor &transpose_inplace(usize dim1, usize dim2);
		Tensor &transpose_inplace();
		Tensor transposed(usize dim1, usize dim2) const;
		Tensor transposed() const;
		Tensor &permute_inplace(VectorInput<usize> dims);
		Tensor permuted(VectorInput<usize> dims) const;

		// Indexing 
		TensorPositionView at(TensorSingleIndexView index) const override;
		Tensor operator()(TensorMultiIndexView index) const;
		Tensor operator()(TensorMultiIndexInitializer index_initializer) const;
		TensorGatherView gather(usize gather_dim, VectorInput<usize> gather_indices) const;
		//  -  TensorLikeWithDims: TensorSingleIndexRange positions() const;

		// Operations
		friend Tensor concat(const Tensor &lhs, const Tensor &rhs, usize concat_dim);
		Tensor &operator+=(const Tensor &other);
		Tensor &operator+=(const TensorLike &other);
		friend Tensor operator+(const Tensor &lhs, const Tensor &rhs);
		friend Tensor operator+(const Tensor &lhs, const TensorLike &rhs);
		friend Tensor operator+(const TensorLike &lhs, const Tensor &rhs);
		Tensor &operator-=(const Tensor &other);
		Tensor &operator-=(const TensorLike &other);
		friend Tensor operator-(const Tensor &lhs, const Tensor &rhs);
		friend Tensor operator-(const Tensor &lhs, const TensorLike &rhs);
		friend Tensor operator-(const TensorLike &lhs, const Tensor &rhs);
		Tensor &operator*=(float scalar);
		friend Tensor operator*(const Tensor &lhs, float scalar);
		friend Tensor operator*(float scalar, const Tensor &rhs);
		Tensor &operator/=(float scalar);
		friend Tensor operator/(const Tensor &lhs, float scalar);
		friend Tensor outer_product(const Tensor &lhs, const Tensor &rhs);
		Tensor contracted(VectorInput<std::pair<usize, usize>> contractions) const;
		friend Tensor contract(const Tensor &lhs, const Tensor &rhs, VectorInput<std::pair<usize, usize>> contractions);
		friend Tensor dot(const Tensor &lhs, const Tensor &rhs);
		friend Tensor matvecmul(const Tensor &lhs, const Tensor &rhs);
		friend Tensor matmatmul(const Tensor &lhs, const Tensor &rhs);
		friend Tensor matmul(const Tensor &lhs, const Tensor &rhs);
		Tensor unfold2d(
			usize kernel_h,
			usize kernel_w,
			usize stride_h = 1,
			usize stride_w = 1
		) const;
		Tensor fold2d(
			usize result_h,
			usize result_w,
			usize kernel_h,
			usize kernel_w,
			usize stride_h = 1,
			usize stride_w = 1
		) const;
		Tensor conv2d(
			Tensor kernels,
			usize stride_h = 1,
			usize stride_w = 1
		) const;

		// Comparison
		friend bool approx_eq(const Tensor &lhs, const Tensor &rhs, float epsilon);

		// Reduction
		float sum() const;
		float mean() const;
		TensorSingleIndex maxpos() const;
		TensorSingleIndex minpos() const;
		TensorPositionView max() const;
		TensorPositionView min() const;

	private:
		// Private constructors
		explicit Tensor(
			VectorInput<usize> dims,
			std::shared_ptr<TensorData> use_data
		);
		explicit Tensor(
			VectorInput<usize> dims,
			VectorInput<usize> strides,
			usize offset,
			std::shared_ptr<TensorData> use_data
		);

		// Private helpers
		static ProcessedVectorMultiIndex process_vector_multi_index(const VectorMultiIndex &index, usize num_elems);
		void require_dims_no_zeroes() const;
		void force_detach_contiguous();
		void perform_contractions(
			TensorSingleIndex from_position,
			TensorSingleIndexView target_position,
			Tensor &target,
			Vector<std::pair<usize, usize>> contractions
		) const;

		// Private members
		Vector<usize> strides_;
		usize offset_;

		// Protected members
		//  -  TensorLike:         std::shared_ptr<TensorData> data_;
		//  -  TensorLikeWithDims: Vector<usize> dims_;
	};

	bool approx_eq(const Tensor &lhs, const Tensor &rhs, float epsilon = 1e-5f);
}
