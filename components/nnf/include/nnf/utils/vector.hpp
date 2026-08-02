#pragma once

#include <initializer_list>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>
#include <nnf/utils/base_types.hpp>

namespace nnf
{
	template <typename T>
	using Vector = std::vector<T>;

	template <typename T>
	using VectorView = std::span<const T>;

	template <typename T>
	using VectorInitializer = std::initializer_list<T>;

	template <typename T>
	class VectorInput
	{
	public:
		VectorInput() = delete;
		VectorInput(VectorInitializer<T> input) : values{ input } {}
		VectorInput(Vector<T> input) : values{ std::move(input) } {}
		VectorInput(VectorView<T> input) : values{ input.begin(), input.end() } {}

		Vector<T> values;
	};

	template <typename T>
	void remove_at_indices(const std::unordered_set<usize> &indices, Vector<T> &target)
	{
		Vector<T> result;
		result.reserve(target.size() - indices.size());
		for (usize index = 0; index < target.size(); ++index)
		{
			if (!indices.contains(index)) result.push_back(target[index]);
		}
		target = std::move(result);
	}
}
