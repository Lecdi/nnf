#include <algorithm>
#include <filesystem>
#include <iostream>
#include <nnf/tensor/indexing.hpp>
#include <mnist/construct_model.hpp>
#include <mnist/load_data.hpp>

constexpr bool  START_FROM_SAVED  = false;
constexpr bool  SAVE_MODEL        = true;
constexpr int   MAX_SAMPLES       = 60000;
constexpr int   NUM_EPOCHS        = 1;
constexpr int   BATCH_SIZE        = 64;
constexpr float LEARNING_RATE     = 0.001f;
constexpr bool  TEST_SAME_SAMPLES = false;

int main(int argc, char **argv)
{
    auto model = mnist::construct_model();
    
    auto [train_x, train_y, test_x, test_y] = mnist::load_mnist(TRAIN_DATA_DIR);

    if constexpr (TEST_SAME_SAMPLES)
    {
        test_x = train_x;
        test_y = train_y;
    }
    
    int train_num_samples = std::min(MAX_SAMPLES, 60000);
    int test_num_samples = std::min(MAX_SAMPLES, 10000);
    nnf::TensorMultiIndex train_index{ {{ 0, train_num_samples }}, nnf::ellipsis };
    nnf::TensorMultiIndex test_index{ {{ 0, test_num_samples }}, nnf::ellipsis };

    model->compile();

    auto save_path = std::filesystem::path(OUTPUT_DIR) / OUTPUT_FILENAME;

    if constexpr (START_FROM_SAVED)
    {
        model->load_from(save_path);
    }
    else
    {
        model->init();
    }

    model->sgd(train_x(train_index), train_y(train_index), NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);

    if constexpr (SAVE_MODEL)
    {
        model->save_to(save_path);
        std::cout
            << "Model saved to "
            << save_path
            << "\n\n"
            << std::endl;
    }
    
    std::cout
        << "Evaluating model on testing data...\n\n"
        << std::endl;

    auto eval_percentage = 100 * model->eval_classifier(test_x(test_index), test_y(test_index));
    std::cout
        << "Percentage of samples correctly identified: "
        << eval_percentage
        << std::endl;
}
