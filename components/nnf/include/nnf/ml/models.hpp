#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/io/logging.hpp>
#include <nnf/ml/layers.hpp>
#include <nnf/ml/losses.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf::ml
{
	class Model
	{
	public:
		Model(std::unique_ptr<io::ModelLogger> logger);

		virtual ~Model() {}

		virtual bool is_compiled() const = 0;
		virtual bool is_init() const = 0;

		virtual Model &init() = 0;
		virtual Model &compile() = 0;
		virtual Model &load_from(const std::filesystem::path &) = 0;
		virtual const Model &save_to(const std::filesystem::path &) const = 0;
		virtual Model &perform_epoch(
			usize num_samples,
			VectorView<float> train_data,
			VectorView<float> train_labels,
			usize batch_size = 16,
			float learning_rate = 0.01
		) = 0;
		virtual Model &sgd(
			const Tensor &train_data,
			const Tensor &train_labels,
			usize epochs,
			usize batch_size = 16,
			float learning_rate = 0.01
		);
		virtual Tensor predict(const Tensor &input) = 0;

	protected:
		std::unique_ptr<io::ModelLogger> logger_;
	};

	class Sequential : public Model
	{
	public:
		Sequential() = delete;
		explicit Sequential(
			usize input_elems,
			usize output_elems,
			VectorInput<std::unique_ptr<Layer>> layers,
			std::unique_ptr<LossFunction> loss,
			std::unique_ptr<io::ModelLogger> logger
		);

		bool is_compiled() const override;
		bool is_init() const override;

		Sequential &init() override;
		Sequential &compile() override;
		Sequential &load_from(const std::filesystem::path &) override;
		const Sequential &save_to(const std::filesystem::path &) const override;
		Sequential &perform_epoch(
			usize num_samples,
			VectorView<float> train_data,
			VectorView<float> train_labels,
			usize batch_size = 16,
			float learning_rate = 0.01
		) override;
		Tensor predict(const Tensor &input) override;
		float eval_regressor(
			const Tensor &test_data,
			const Tensor &test_labels,
			float acceptable_difference
		);
		float eval_regressor(
			const Tensor &test_data,
			const Tensor &test_labels,
			std::function<bool(float, float)> verifier
		);
		float eval_classifier(
			const Tensor &test_data,
			const Tensor &test_labels
		);

	private:
		usize input_elems_;
		usize output_elems_;
		bool init_;
		autodiff::InputValue input_value_;
		autodiff::InputValue label_value_;
		autodiff::Value out_value_;
		autodiff::Value loss_value_;
		Vector<std::unique_ptr<Layer>> layers_;
		std::unique_ptr<LossFunction> loss_function_;
		std::optional<autodiff::AutodiffEngine> engine_;
	};
}
