#include <benchmark/benchmark.h>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

static void bench_flatten_op_forward(benchmark::State &state)
{
    const nnf::usize in_dim1 = state.range(0);
    const nnf::usize in_dim2 = state.range(1);
    const nnf::usize in_dim3 = state.range(2);

    auto in_input = nnf::autodiff::InputValue({ in_dim1, in_dim2, in_dim3 });
    auto out = in_input.flattened();

    in_input.input(nnf::Tensor::normals({ in_dim1, in_dim2, in_dim3 }, 0, 1));

    for (auto _ : state)
    {
        out.node->forward();
    }
}

BENCHMARK(bench_flatten_op_forward)
->Args({ 12, 12, 16 });

static void bench_flatten_op_back(benchmark::State &state)
{
    const nnf::usize in_dim1 = state.range(0);
    const nnf::usize in_dim2 = state.range(1);
    const nnf::usize in_dim3 = state.range(2);

    auto in = nnf::autodiff::InputValue({ in_dim1, in_dim2, in_dim3 });
    auto out = in.flattened();

    in.input(nnf::Tensor::normals({ in_dim1, in_dim2, in_dim3 }, 0, 1));

    out.node->forward();

    in.node->grad.make_zero();
    out.node->grad.make_normal(0, 1);

    for (auto _ : state)
    {
        out.node->back();
    }
}

BENCHMARK(bench_flatten_op_back)
->Args({ 12, 12, 16 });
