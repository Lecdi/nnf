#include <benchmark/benchmark.h>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

static void bench_tensor_matmatmul(benchmark::State &state)
{
    const nnf::usize n = state.range(0);
    auto a = nnf::Tensor::normals({ {n, n} }, 0, 1);
    auto b = nnf::Tensor::normals({ {n, n} }, 0, 1);

    for (auto _ : state)
    {
        auto c = matmatmul(a, b);
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(bench_tensor_matmatmul)
->Arg({ 16 })
->Arg({ 32 })
->Arg({ 128 })
->Arg({ 256 });
