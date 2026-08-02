#include <benchmark/benchmark.h>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/autodiff/unary_scalar_operations.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

static void bench_relu_op_forward(benchmark::State &state)
{
    const nnf::usize dim = state.range(0);

    auto in = nnf::autodiff::InputValue({ dim });
    auto out = nnf::autodiff::relu()(in);

    in.input(nnf::Tensor::normals({ dim }, 0, 1));

    for (auto _ : state)
    {
        out.node->forward();
    }
}

BENCHMARK(bench_relu_op_forward)
->Arg({ 10 })
->Arg({ 128 })
->Arg({ 2308 });

static void bench_relu_op_back(benchmark::State &state)
{
    const nnf::usize dim = state.range(0);

    auto in = nnf::autodiff::InputValue({ dim });
    auto out = nnf::autodiff::relu()(in);

    in.input(nnf::Tensor::normals({ dim }, 0, 1));

    out.node->forward();

    in.node->grad.make_zero();
    out.node->grad.make_normal(0, 1);

    for (auto _ : state)
    {
        out.node->back();
    }
}

BENCHMARK(bench_relu_op_back)
->Arg({ 10 })
->Arg({ 128 })
->Arg({ 2308 });
