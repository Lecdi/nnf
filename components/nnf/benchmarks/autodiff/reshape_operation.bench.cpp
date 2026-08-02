#include <benchmark/benchmark.h>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

static void bench_reshape_op_forward(benchmark::State &state)
{
    const nnf::usize dim = state.range(0);

    auto in = nnf::autodiff::InputValue({ dim * dim });
    auto out = in.reshaped({ dim, dim });

    in.input(nnf::Tensor::normals({ dim * dim }, 0, 1));

    for (auto _ : state)
    {
        out.node->forward();
    }
}

BENCHMARK(bench_reshape_op_forward)
->Arg({ 28 })
->Arg({ 128 });

static void bench_reshape_op_back(benchmark::State &state)
{
    const nnf::usize dim = state.range(0);

    auto in = nnf::autodiff::InputValue({ dim * dim });
    auto out = in.reshaped({ dim, dim });

    in.input(nnf::Tensor::normals({ dim * dim }, 0, 1));

    out.node->forward();

    in.node->grad.make_zero();
    out.node->grad.make_normal(0, 1);

    for (auto _ : state)
    {
        out.node->back();
    }
}

BENCHMARK(bench_reshape_op_back)
->Arg({ 28 })
->Arg({ 128 });
