#include <nnf/io/logging.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <nnf/utils/base_types.hpp>

std::unique_ptr<nnf::io::ModelLogger> nnf::io::no_model_logger()
{
	return std::make_unique<NoModelLogger>();
}

nnf::io::StdoutModelLogger::StdoutModelLogger(usize epoch_width)
	: epoch_width_{ epoch_width }
{
}

nnf::io::StdoutModelLogger &nnf::io::StdoutModelLogger::start_training(
	usize num_epochs,
	usize num_samples,
	usize default_batch_size
)
{
	status_ = StdoutModelLoggerStatus::TRAINING;
	num_epochs_ = num_epochs;
	completed_epochs_ = 0;
	num_samples_ = num_samples;
	default_batch_size_ = default_batch_size;
	completed_samples_ = 0;
	epoch_width_position_ = 0;

	std::cout
		<< "STARTED TRAINING ["
		<< num_epochs_
		<< " epochs] ["
		<< num_samples_
		<< " samples]\n"
		<< std::endl;

	return *this;
}

nnf::io::StdoutModelLogger &nnf::io::StdoutModelLogger::start_epoch()
{
	if (status_ != StdoutModelLoggerStatus::TRAINING)
		throw std::logic_error("Attempt to log epoch start but not currently training");

	std::cout
		<< "Epoch "
		<< completed_epochs_ + 1
		<< " / "
		<< num_epochs_
		<< " : [";

	return *this;
}

nnf::io::StdoutModelLogger &nnf::io::StdoutModelLogger::complete_epoch(float loss)
{
	if (status_ != StdoutModelLoggerStatus::TRAINING)
		throw std::logic_error("Attempt to log epoch completion but not currently training");

	if (epoch_width_position_ < epoch_width_)
		std::cout << std::string(epoch_width_ - epoch_width_position_, '-');

	completed_epochs_ += 1;
	completed_samples_ = 0;
	epoch_width_position_ = 0;

	std::cout
		<< "] Loss "
		<< loss
		<< std::endl;

	return *this;
}

nnf::io::StdoutModelLogger &nnf::io::StdoutModelLogger::complete_batch(usize batch_size)
{
	if (status_ != StdoutModelLoggerStatus::TRAINING)
		throw std::logic_error("Attempt to log batch completion but not currently training");

	if (batch_size == 0) completed_samples_ += default_batch_size_;
	else completed_samples_ += batch_size;

	usize new_epoch_width_position = epoch_width_ * completed_samples_ / num_samples_;

	if (new_epoch_width_position > epoch_width_position_ && new_epoch_width_position <= epoch_width_)
		std::cout << std::string(new_epoch_width_position - epoch_width_position_, '-');

	epoch_width_position_ = new_epoch_width_position;
	return *this;
}

nnf::io::StdoutModelLogger &nnf::io::StdoutModelLogger::complete_training()
{
	if (status_ != StdoutModelLoggerStatus::TRAINING)
		throw std::logic_error("Attempt to log training completion but not currently training");

	std::cout << "\nCOMPLETED TRAINING\n\n" << std::endl;

	status_ = StdoutModelLoggerStatus::EMPTY;
	return *this;
}

std::unique_ptr<nnf::io::ModelLogger> nnf::io::stdout_model_logger()
{
	return std::make_unique<StdoutModelLogger>();
}
