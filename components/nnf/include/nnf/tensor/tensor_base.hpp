#pragma once

#include <memory>
#include <nnf/tensor/indexing.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf
{
	struct TensorData
	{
		Vector<float> values;
	};

	class TensorPositionView;

	class TensorLike
	{
	public:
		TensorLike() = delete;
		explicit TensorLike(std::shared_ptr<TensorData> data);

		virtual ~TensorLike() {}

		bool is_empty() const;
		bool is_detached() const;
		virtual bool dims_compatible(VectorView<usize> dims) const = 0;

		virtual TensorPositionView at(TensorSingleIndexView index) const = 0;
		TensorPositionView at(TensorSingleIndexInitializer index_initializer) const;

	protected:
		std::shared_ptr<TensorData> data_;
	};

	class TensorLikeWithDims : public TensorLike
	{
	public:
		TensorLikeWithDims() = delete;
		explicit TensorLikeWithDims(std::shared_ptr<TensorData> data, VectorInput<usize> dims_);

		bool dims_compatible(VectorView<usize> dims) const;
		usize rank() const;
		Vector<usize> get_dims() const;
		VectorView<usize> view_dims() const;
		TensorSingleIndexRange positions() const;

	protected:
		Vector<usize> dims_;
	};
}
