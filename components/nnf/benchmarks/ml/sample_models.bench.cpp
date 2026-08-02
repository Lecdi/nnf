#include <algorithm>
#include <memory>
#include <utility>
#include <benchmark/benchmark.h>
#include <nnf/io/logging.hpp>
#include <nnf/ml/layers.hpp>
#include <nnf/ml/losses.hpp>
#include <nnf/ml/models.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

static void bench_nn_model(benchmark::State &state)
{
    const nnf::usize num_samples = state.range(0);

    nnf::Vector<std::unique_ptr<nnf::ml::Layer>> layers{};
    layers.push_back(nnf::ml::dense_relu_layer(28 * 28, 64));
    layers.push_back(nnf::ml::dense_relu_layer(64, 64));
    layers.push_back(nnf::ml::dense_no_activation_layer(64, 10));

    nnf::ml::Sequential model{
        28 * 28,
        10,
        std::move(layers),
        nnf::ml::softmax_crossentropy_loss(),
        nnf::io::no_model_logger()
    };

    model.compile();
    model.init();

    auto train_x = nnf::Tensor::normals({ num_samples, 28 * 28 }, 0.5, 0.25);
    auto train_y = (nnf::Tensor::normals({ num_samples, 10 }, 0.5, 0.25));
    train_x.apply_inplace([](float value) { return std::min(std::max(value, 0.f), 1.f); });
    train_y.apply_inplace([](float value) { return std::min(std::max(value, 0.f), 1.f); });

    for (auto _ : state)
    {
        model.sgd(train_x, train_y, 1);
    }
}

static void bench_cnn_model(benchmark::State &state)
{
    const nnf::usize num_samples = state.range(0);

    nnf::Vector<std::unique_ptr<nnf::ml::Layer>> layers{};
    layers.push_back(nnf::ml::reshape_layer({ 28, 28 }));
    layers.push_back(nnf::ml::conv2d_relu_layer(28, 28, 16, 3, 3));
    layers.push_back(nnf::ml::max_pool_2d_layer(2, 2));
    layers.push_back(nnf::ml::flatten_layer());
    layers.push_back(nnf::ml::dense_relu_layer(13 * 13 * 16, 128));
    layers.push_back(nnf::ml::dense_no_activation_layer(128, 10));

    nnf::ml::Sequential model{
        28 * 28,
        10,
        std::move(layers),
        nnf::ml::softmax_crossentropy_loss(),
        nnf::io::no_model_logger()
    };

    model.compile();
    model.init();

    auto train_x = nnf::Tensor::normals({ num_samples, 28 * 28 }, 0.5, 0.25);
    auto train_y = (nnf::Tensor::normals({ num_samples, 10 }, 0.5, 0.25));
    train_x.apply_inplace([](float value) { return std::min(std::max(value, 0.f), 1.f); });
    train_y.apply_inplace([](float value) { return std::min(std::max(value, 0.f), 1.f); });

    for (auto _ : state)
    {
        model.sgd(train_x, train_y, 1);
    }
}

BENCHMARK(bench_nn_model)
->Arg(16);

BENCHMARK(bench_cnn_model)
->Arg(16);
