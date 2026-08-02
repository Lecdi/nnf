#pragma once

#include <iterator>
#include <optional>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf
{
	using VectorSingleIndex = ssize;

	using TensorSingleIndexArg = VectorSingleIndex;

	using TensorSingleIndex = Vector<TensorSingleIndexArg>;
	using TensorSingleIndexView = VectorView<TensorSingleIndexArg>;
	using TensorSingleIndexInitializer = VectorInitializer<TensorSingleIndexArg>;
	using TensorSingleIndexInput = VectorInput<TensorSingleIndexArg>;

	class VectorMultiIndex
	{
	public:
		VectorMultiIndex() = delete;
		VectorMultiIndex(ssize pos) : start{ pos }, end{ pos + 1 }, step{ 1 } {}
		VectorMultiIndex(ssize start, ssize end) : start{ start }, end{ end }, step{ 1 } {}
		VectorMultiIndex(ssize start, ssize end, usize step) : start{ start }, end{ end }, step{ step } {}

		ssize start;
		ssize end;
		usize step;
	};

	using TensorMultiIndexArg = std::optional<VectorMultiIndex>;
	constexpr const TensorMultiIndexArg ellipsis = std::nullopt;

	using TensorMultiIndex = Vector<TensorMultiIndexArg>;
	using TensorMultiIndexView = VectorView<TensorMultiIndexArg>;
	using TensorMultiIndexInitializer = VectorInitializer<TensorMultiIndexArg>;
	using TensorMultiIndexInput = VectorInput<TensorMultiIndexArg>;

	class TensorSingleIndexIterator
	{
	public:
		using value_type = TensorSingleIndexView;
		using difference_type = ssize;
		using iterator_category = std::forward_iterator_tag;
		using iterator_concept = std::forward_iterator_tag;

		TensorSingleIndexIterator();
		explicit TensorSingleIndexIterator(VectorInput<usize> dims);

		value_type operator*() const;
		TensorSingleIndexIterator &operator++();
		TensorSingleIndexIterator operator++(int);
		friend bool operator==(const TensorSingleIndexIterator &lhs, const TensorSingleIndexIterator &rhs);

	private:
		Vector<usize> dims_;
		Vector<ssize> current_;
		bool finished_;
	};

	class TensorSingleIndexRange
	{
	public:
		explicit TensorSingleIndexRange(VectorInput<usize> dims);

		TensorSingleIndexIterator begin() const;
		TensorSingleIndexIterator end() const;

	private:
		Vector<usize> dims_;
	};

	class TensorRawDataIndexIterator
	{
	public:
		using value_type = usize;
		using difference_type = ssize;
		using iterator_category = std::forward_iterator_tag;
		using iterator_concept = std::forward_iterator_tag;

		TensorRawDataIndexIterator();
		explicit TensorRawDataIndexIterator(VectorInput<usize> dims, VectorInput<usize> strides, usize offset);

		value_type operator*() const;
		TensorRawDataIndexIterator &operator++();
		TensorRawDataIndexIterator operator++(int);
		friend bool operator==(const TensorRawDataIndexIterator &lhs, const TensorRawDataIndexIterator &rhs);

	private:
		Vector<usize> dims_;
		Vector<usize> strides_;
		usize current_raw_data_index_;
		Vector<usize> current_position_;
		bool finished_;
	};

	class TensorRawDataIndexRange
	{
	public:
		explicit TensorRawDataIndexRange(VectorInput<usize> dims, VectorInput<usize> strides, usize offset);

		TensorRawDataIndexIterator begin() const;
		TensorRawDataIndexIterator end() const;

	private:
		Vector<usize> dims_;
		Vector<usize> strides_;
		usize offset_;
	};

	struct ProcessedVectorMultiIndex
	{
		usize start;
		usize num_values;
		usize step;
	};
}

namespace std::ranges {
	template<>
	inline constexpr bool enable_borrowed_range<nnf::TensorSingleIndexRange> = true;

	template<>
	inline constexpr bool enable_borrowed_range<nnf::TensorRawDataIndexRange> = true;
}
