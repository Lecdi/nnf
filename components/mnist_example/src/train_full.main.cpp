#include <filesystem>
#include <iostream>
#include <mnist/construct_model.hpp>
#include <mnist/load_data.hpp>

int main(int argc, char **argv)
{
    auto model = mnist::construct_model();
    
    auto [train_x, train_y, test_x, test_y] = mnist::load_mnist(TRAIN_DATA_DIR);

    model->compile();
    model->init();

    model->sgd(train_x, train_y, 30, 64, 0.001f);
    model->sgd(train_x, train_y, 30, 64, 0.0001f);
    
    auto save_path = std::filesystem::path(OUTPUT_DIR) / OUTPUT_FILENAME;
    model->save_to(save_path);

    std::cout
        << "Model saved to "
        << save_path
        << "\n\n\nEvaluating model on testing data...\n\n"
        << std::endl;

    auto eval_percentage = 100 * model->eval_classifier(test_x, test_y);
    std::cout
        << "Percentage of samples correctly identified: "
        << eval_percentage
        << std::endl;
}
