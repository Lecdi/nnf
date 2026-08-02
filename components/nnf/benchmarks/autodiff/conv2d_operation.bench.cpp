#include <benchmark/benchmark.h>
#include <nnf/autodiff/autodiff_types.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>

static void bench_conv2d_op_forward(benchmark::State &state)
{
    const nnf::usize input_square_length = state.range(0);
    const nnf::usize num_kernels = state.range(1);
    const nnf::usize kernel_square_length = state.range(2);
    const nnf::usize stride = state.range(3);

    auto in_input = nnf::autodiff::InputValue({ input_square_length, input_square_length });
    auto in_kernels = nnf::autodiff::InputValue({ num_kernels, kernel_square_length, kernel_square_length });
    auto out = in_input.conv2d(in_kernels, stride, stride);

    in_input.input(nnf::Tensor::normals({ input_square_length, input_square_length }, 0, 1));
    in_kernels.input(nnf::Tensor::normals({ num_kernels, kernel_square_length, kernel_square_length }, 0, 1));

    for (auto _ : state)
    {
        out.node->forward();
    }
}

BENCHMARK(bench_conv2d_op_forward)
->Args({ 28, 16, 5, 2 });

static void bench_conv2d_op_back(benchmark::State &state)
{
    const nnf::usize input_square_length = state.range(0);
    const nnf::usize num_kernels = state.range(1);
    const nnf::usize kernel_square_length = state.range(2);
    const nnf::usize stride = state.range(3);

    auto in_input = nnf::autodiff::InputValue({ input_square_length, input_square_length });
    auto in_kernels = nnf::autodiff::InputValue({ num_kernels, kernel_square_length, kernel_square_length });
    auto out = in_input.conv2d(in_kernels, stride, stride);

    in_input.input(nnf::Tensor::normals({ input_square_length, input_square_length }, 0, 1));
    in_kernels.input(nnf::Tensor::normals({ num_kernels, kernel_square_length, kernel_square_length }, 0, 1));

    out.node->forward();

    in_input.node->grad.make_zero();
    in_kernels.node->grad.make_zero();
    out.node->grad.make_normal(0, 1);

    for (auto _ : state)
    {
        out.node->back();
    }
}

BENCHMARK(bench_conv2d_op_back)
->Args({ 28, 16, 5, 2 });
