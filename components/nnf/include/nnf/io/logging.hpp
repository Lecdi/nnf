#pragma once

#include <memory>
#include <nnf/utils/base_types.hpp>

namespace nnf::io
{
	class ModelLogger
	{
	public:
		virtual ~ModelLogger() {}

		virtual ModelLogger &start_training(usize num_epochs, usize num_samples, usize default_batch_size) = 0;
		virtual ModelLogger &start_epoch() = 0;
		virtual ModelLogger &complete_epoch(float loss) = 0;
		virtual ModelLogger &complete_batch(usize batch_size = 0) = 0;
		virtual ModelLogger &complete_training() = 0;
	};

	class NoModelLogger : public ModelLogger
	{
	public:
		NoModelLogger &start_training(usize num_epochs, usize num_samples, usize default_batch_size) override { return *this; }
		NoModelLogger &start_epoch() override { return *this; }
		NoModelLogger &complete_epoch(float loss) override { return *this; }
		NoModelLogger &complete_batch(usize batch_size = 0) override { return *this; }
		NoModelLogger &complete_training() override { return *this; }
	};

	std::unique_ptr<ModelLogger> no_model_logger();

	enum class StdoutModelLoggerStatus
	{
		EMPTY,
		TRAINING
	};

	class StdoutModelLogger : public ModelLogger
	{
	public:
		StdoutModelLogger(usize epoch_width = 40);

		StdoutModelLogger &start_training(usize num_epochs, usize num_samples, usize default_batch_size) override;
		StdoutModelLogger &start_epoch() override;
		StdoutModelLogger &complete_epoch(float loss) override;
		StdoutModelLogger &complete_batch(usize batch_size = 0) override;
		StdoutModelLogger &complete_training() override;

	private:
		usize epoch_width_;

		StdoutModelLoggerStatus status_ = StdoutModelLoggerStatus::EMPTY;

		usize num_epochs_ = 0;
		usize completed_epochs_ = 0;
		usize num_samples_ = 0;
		usize default_batch_size_ = 0;
		usize completed_samples_ = 0;
		usize epoch_width_position_ = 0;
	};

	std::unique_ptr<ModelLogger> stdout_model_logger();
}
