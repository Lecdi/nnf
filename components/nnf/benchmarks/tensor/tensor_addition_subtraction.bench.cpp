#include <benchmark/benchmark.h>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

static void bench_tensor_add_matrices(benchmark::State &state)
{
    const nnf::usize n = state.range(0);
    auto a = nnf::Tensor::normals({ {n, n} }, 0, 1);
    auto b = nnf::Tensor::normals({ {n, n} }, 0, 1);

    for (auto _ : state)
    {
        auto c = a + b;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(bench_tensor_add_matrices)
->Arg({ 32 })
->Arg({ 256 })
->Arg({ 2048 });

static void bench_tensor_add_vectors(benchmark::State &state)
{
    const nnf::usize n = state.range(0);
    auto a = nnf::Tensor::normals({ {n} }, 0, 1);
    auto b = nnf::Tensor::normals({ {n} }, 0, 1);

    for (auto _ : state)
    {
        auto c = a + b;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(bench_tensor_add_vectors)
->Arg({ 32 })
->Arg({ 256 })
->Arg({ 2048 });
