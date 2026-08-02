#include <benchmark/benchmark.h>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

static void bench_matvecmul_op_forward(benchmark::State &state)
{
    const nnf::usize out_dim = state.range(0);
    const nnf::usize in_dim = state.range(1);

    auto in_input = nnf::autodiff::InputValue({ in_dim });
    auto in_mat = nnf::autodiff::InputValue({ out_dim, in_dim });
    auto out = matvecmul(in_mat, in_input);

    in_input.input(nnf::Tensor::normals({ in_dim }, 0, 1));
    in_mat.input(nnf::Tensor::normals({ out_dim, in_dim }, 0, 1));

    for (auto _ : state)
    {
        out.node->forward();
    }
}

BENCHMARK(bench_matvecmul_op_forward)
->Args({ 10, 128 })
->Args({ 128, 2308 });

static void bench_matvecmul_op_back(benchmark::State &state)
{
    const nnf::usize out_dim = state.range(0);
    const nnf::usize in_dim = state.range(1);

    auto in_input = nnf::autodiff::InputValue({ in_dim });
    auto in_mat = nnf::autodiff::InputValue({ out_dim, in_dim });
    auto out = matvecmul(in_mat, in_input);

    in_input.input(nnf::Tensor::normals({ in_dim }, 0, 1));
    in_mat.input(nnf::Tensor::normals({ out_dim, in_dim }, 0, 1));

    out.node->forward();

    in_input.node->grad.make_zero();
    in_mat.node->grad.make_zero();
    out.node->grad.make_normal(0, 1);

    for (auto _ : state)
    {
        out.node->back();
    }
}

BENCHMARK(bench_matvecmul_op_back)
->Args({ 10, 128 })
->Args({ 128, 2308 });
