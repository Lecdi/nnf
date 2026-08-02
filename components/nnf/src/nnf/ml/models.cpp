#include <nnf/ml/models.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/io/logging.hpp>
#include <nnf/ml/layers.hpp>
#include <nnf/ml/losses.hpp>
#include <nnf/tensor/indexing.hpp>
#include <nnf/tensor/tensor_base.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/math.hpp>
#include <nnf/utils/vector.hpp>

nnf::ml::Model::Model(std::unique_ptr<io::ModelLogger> logger)
	: logger_{ std::move(logger) }
{
}

nnf::ml::Model &nnf::ml::Model::sgd(
	const Tensor &train_data,
	const Tensor &train_labels,
	usize epochs,
	usize batch_size,
	float learning_rate
)
{
	if (train_data.rank() < 1)
		throw std::invalid_argument("Training data has no dimensions");
	if (train_labels.rank() < 1)
		throw std::invalid_argument("Training labels has no dimensions");
	if (train_data.view_dims()[0] != train_labels.view_dims()[0])
		throw std::logic_error("Training data and labels have different length (contain different number of samples)");
	
	usize num_samples = train_labels.view_dims()[0];
	logger_->start_training(epochs, num_samples, batch_size);

	const auto train_data_raw = train_data.get_contiguous_data();
	const auto train_labels_raw = train_labels.get_contiguous_data();

	for (usize i = 0; i < epochs; ++i)
		perform_epoch(num_samples, train_data_raw, train_labels_raw, batch_size, learning_rate);

	logger_->complete_training();

	return *this;
}

nnf::ml::Sequential::Sequential(
	usize input_elems,
	usize output_elems,
	VectorInput<std::unique_ptr<Layer>> layers,
	std::unique_ptr<LossFunction> loss,
	std::unique_ptr<io::ModelLogger> logger
)
	: input_elems_{ input_elems }, output_elems_{ output_elems },
	input_value_{ {input_elems} }, label_value_{ {output_elems} }, init_{ false },
	layers_{ std::move(layers.values) }, loss_function_{ std::move(loss) },
	Model{ std::move(logger) }
{
	autodiff::Value prev_value = input_value_;
	for (auto &layer : layers_)
	{
		layer->set_prev_value(prev_value);
		prev_value = layer->out_value;
	}
	out_value_ = prev_value;
	if (out_value_.view_dims().size() != 1 || out_value_.view_dims()[0] != output_elems)
		throw std::invalid_argument("Output of last layer has incorrect dimensions");
	loss_value_ = (*loss_function_)(out_value_, label_value_);
}

bool nnf::ml::Sequential::is_compiled() const
{
	return engine_.has_value();
}

bool nnf::ml::Sequential::is_init() const
{
	return init_;
}

nnf::ml::Sequential &nnf::ml::Sequential::init()
{
	for (auto &layer : layers_)
	{
		layer->init_params();
	}
	init_ = true;
	return *this;
}

nnf::ml::Sequential &nnf::ml::Sequential::compile()
{
	engine_ = autodiff::AutodiffEngine(loss_value_);
	engine_.value().compile();
	return *this;
}

nnf::ml::Sequential &nnf::ml::Sequential::load_from(const std::filesystem::path &savefile)
{
	std::ifstream stream{ savefile, std::ios::binary };
	if (!stream)
		throw std::ios_base::failure("Failed to open file: " + savefile.string());

	for (auto &layer : layers_)
	{
		layer->read_params_from(stream);
	}

	if (stream.fail())
		throw std::ios_base::failure("Failure while reading file: " + savefile.string());
	if (stream.eof())
		throw std::runtime_error("Model file is wrong format (too short for number of params): " + savefile.string());
	char dummy;
	stream.read(&dummy, 1);
	if (!stream.eof())
		throw std::runtime_error("Model file is wrong format (too long for number of params): " + savefile.string());

	init_ = true;
	return *this;
}

const nnf::ml::Sequential &nnf::ml::Sequential::save_to(const std::filesystem::path &savefile) const
{
	std::filesystem::create_directories(savefile.parent_path());

	std::ofstream stream{ savefile, std::ios::binary };
	if (!stream)
		throw std::ios_base::failure("Failed to open file: " + savefile.string());

	for (auto &layer : layers_)
	{
		layer->write_params_to(stream);
	}

	if (stream.fail())
		throw std::ios_base::failure("Failure while writing to file: " + savefile.string());

	return *this;
}

nnf::ml::Sequential &nnf::ml::Sequential::perform_epoch(
	usize num_samples,
	VectorView<float> train_data,
	VectorView<float> train_labels,
	usize batch_size,
	float learning_rate
)
{
	if (!is_compiled())
		throw std::logic_error("Attempt to train non-compiled model");
	if (!is_init())
		throw std::logic_error("Attempt to train non-initialized model");

	if (train_data.size() != num_samples * input_elems_)
		throw std::invalid_argument("Train data has incorrect dimensions");
	if (train_labels.size() != num_samples * output_elems_)
		throw std::invalid_argument("Train labels have incorrect dimensions");

	logger_->start_epoch();

	float learning_rate_per_sample = learning_rate / static_cast<float>(batch_size);

	for (const auto &layer : layers_)
		layer->reset_grad();

	auto sample_index_order = Vector<usize>(num_samples);
	std::iota(sample_index_order.begin(), sample_index_order.end(), 0);
	std::mt19937 rng(std::random_device{}());
	std::shuffle(sample_index_order.begin(), sample_index_order.end(), rng);

	float total_loss = 0.f;

	auto loss_grad = Tensor::from_data(Vector<usize>{}, { 1.f });
	usize batch_index = 0;
	for (usize sample_index : sample_index_order)
	{
		if (batch_index >= batch_size)
		{
			for (const auto &layer : layers_)
			{
				layer->update_params(learning_rate_per_sample);
				layer->reset_grad();
			}
			batch_index = 0;
			logger_->complete_batch();
		}

		input_value_.input_raw_data({
			train_data.begin() + sample_index * input_elems_,
			train_data.begin() + (sample_index + 1) * input_elems_
		});
		label_value_.input_raw_data({
			train_labels.begin() + sample_index * output_elems_,
			train_labels.begin() + (sample_index + 1) * output_elems_
		});
		engine_->fill_forward();
		engine_->set_final_grad(loss_grad);
		engine_->fill_back();
		total_loss += loss_value_.value().at({});

		for (const auto &layer : layers_)
			layer->accumulate_grad();

		++batch_index;
	}

	for (const auto &layer : layers_)
		layer->update_params(learning_rate_per_sample);

	logger_->complete_epoch(total_loss / static_cast<float>(num_samples));

	return *this;
}

nnf::Tensor nnf::ml::Sequential::predict(const Tensor &input)
{
	if (!is_compiled())
		throw std::logic_error("Attempt to predict with non-compiled model");
	if (!is_init())
		throw std::logic_error("Attempt to predict with non-initialized model");

	label_value_.node->value.make_zero();
	input_value_.input(input);
	engine_->fill_forward();

	return out_value_.value();
}

float nnf::ml::Sequential::eval_regressor(
	const Tensor &test_data,
	const Tensor &test_labels,
	float acceptable_difference
)
{
	return eval_regressor(test_data, test_labels,
		[acceptable_difference](float pred, float label) {
			return float_approx_eq(pred, label, acceptable_difference);
		}
	);
}

float nnf::ml::Sequential::eval_regressor(
	const Tensor &test_data,
	const Tensor &test_labels,
	std::function<bool(float, float)> verifier
)
{
	if (!is_compiled())
		throw std::logic_error("Attempt to evaluate non-compiled model");
	if (!is_init())
		throw std::logic_error("Attempt to evaluate non-initialized model");
	if (test_data.rank() != 2 || test_labels.rank() != 2)
		throw std::invalid_argument("Evaluation data has incorrect format");

	usize num_samples = test_data.view_dims()[0];
	if (test_labels.view_dims()[0] != num_samples)
		throw std::invalid_argument("Evaluation data and labels have different length");
	if (test_labels.view_dims()[1] != out_value_.view_dims()[0])
		throw std::invalid_argument("Evaluation labels do not have compatible dimensions with model output");

	if (out_value_.num_elems() != 1)
		throw std::logic_error("Output has more than one element so not a valid regressor");

	usize num_correct = 0;
	for (ssize sample_number = 0; sample_number < num_samples; ++sample_number)
	{
		auto pred = predict(test_data({ {sample_number}, ellipsis }).copy().reshaped({ {input_elems_} })).at({ {0} });
		auto target = test_labels.at({ {sample_number, 0} });
		if (verifier(pred, target))
			++num_correct;
	}

	return static_cast<float>(num_correct) / static_cast<float>(num_samples);
}

float nnf::ml::Sequential::eval_classifier(
	const Tensor &test_data,
	const Tensor &test_labels
)
{
	if (!is_compiled())
		throw std::logic_error("Attempt to evaluate non-compiled model");
	if (!is_init())
		throw std::logic_error("Attempt to evaluate non-initialized model");
	if (test_data.rank() != 2 || test_labels.rank() != 2)
		throw std::invalid_argument("Evaluation data has incorrect format");

	usize num_samples = test_data.view_dims()[0];
	if (test_labels.view_dims()[0] != num_samples)
		throw std::invalid_argument("Evaluation data and labels have different length");
	if (test_labels.view_dims()[1] != out_value_.view_dims()[0])
		throw std::invalid_argument("Evaluation labels do not have compatible dimensions with model output");

	if (out_value_.num_elems() <= 1)
		throw std::logic_error("Output has only one element so not a valid classifier");

	usize num_correct = 0;
	for (usize sample_number = 0; sample_number < num_samples; ++sample_number)
	{
		auto pred = predict(test_data({ {sample_number}, ellipsis }).copy().reshaped({{input_elems_}})).maxpos()[0];
		auto target = test_labels({ {sample_number}, ellipsis }).maxpos()[1];
		auto a = pred + target;
		if (pred == target)
			++num_correct;
	}

	return static_cast<float>(num_correct) / static_cast<float>(num_samples);
}
