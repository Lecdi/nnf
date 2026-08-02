#include <benchmark/benchmark.h>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

static void bench_tensor_conv2d(benchmark::State &state)
{
    const nnf::usize m = state.range(0);
    const nnf::usize n = state.range(1);
    const nnf::usize c = state.range(2);
    const nnf::usize k = state.range(3);
    auto a = nnf::Tensor::normals({ {m, n} }, 0, 1);
    auto b = nnf::Tensor::normals({ {k, k, c} }, 0, 1);

    for (auto _ : state)
    {
        auto c = a.conv2d(b);
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(bench_tensor_conv2d)
->Args({ 16, 16, 8, 3 })
->Args({ 32, 32, 16, 5 })
->Args({ 32, 32, 32, 5 });
