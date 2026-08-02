#pragma once

#include <functional>
#include <memory>
#include <nnf/tensor/tensor_base.hpp>
#include <nnf/tensor/indexing.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf
{
	class TensorPositionView
	{
	public:
		TensorPositionView() = delete;
		explicit TensorPositionView(
			std::shared_ptr<TensorData> data,
			usize index
		);

		bool is_empty() const;
		bool is_detached() const;
		operator float() const;

		TensorPositionView &clear();
		TensorPositionView &detatch();
		TensorPositionView copy() const;

		TensorPositionView &set(float other);
		TensorPositionView &apply_inplace(const std::function<float(float)> &func);
		TensorPositionView &operator+=(float other);
		TensorPositionView &operator-=(float other);
		TensorPositionView &operator*=(float other);
		TensorPositionView &operator/=(float other);

	private:
		std::shared_ptr<TensorData> data_;
		usize offset_;
	};

	class TensorGatherView : public TensorLikeWithDims
	{
	public:
		TensorGatherView() = delete;
		TensorGatherView(
			std::shared_ptr<TensorData> data,
			VectorInput<usize> original_dims,
			VectorInput<usize> strides,
			usize offset,
			usize gather_dim,
			VectorInput<usize> gather_indices
		);

		TensorPositionView at(TensorSingleIndexView index) const override;

	private:
		Vector<usize> strides_;
		usize offset_;
		usize gathered_dim_;
		Vector<usize> gathered_indices_;
	};
}
