cmake_minimum_required(VERSION 3.23)
project(nnf VERSION 0.0.0)

platformer_add_library(nnf STATIC)
target_sources(nnf PUBLIC
    FILE_SET headers
    TYPE HEADERS
    BASE_DIRS "include"
    FILES
        "include/nnf/autodiff/autodiff_types.hpp"
        "include/nnf/autodiff/base_operations.hpp"
        "include/nnf/autodiff/binary_tensor_operations.hpp"
        "include/nnf/autodiff/unary_scalar_operations.hpp"
        "include/nnf/autodiff/unary_tensor_operations.hpp"
        "include/nnf/io/logging.hpp"
        "include/nnf/io/serialization.hpp"
        "include/nnf/ml/layers.hpp"
        "include/nnf/ml/losses.hpp"
        "include/nnf/ml/models.hpp"
        "include/nnf/ml/param_initializers.hpp"
        "include/nnf/tensor/tensor_base.hpp"
        "include/nnf/tensor/indexing.hpp"
        "include/nnf/tensor/tensor.hpp"
        "include/nnf/tensor/views.hpp"
        "include/nnf/utils/base_types.hpp"
        "include/nnf/utils/math.hpp"
        "include/nnf/utils/vector.hpp"
)
target_sources(nnf PRIVATE
    "src/nnf/autodiff/autodiff_types.cpp"
    "src/nnf/autodiff/base_operations.cpp"
    "src/nnf/autodiff/binary_tensor_operations.cpp"
    "src/nnf/autodiff/unary_scalar_operations.cpp"
    "src/nnf/autodiff/unary_tensor_operations.cpp"
    "src/nnf/io/logging.cpp"
    "src/nnf/ml/layers.cpp"
    "src/nnf/ml/losses.cpp"
    "src/nnf/ml/models.cpp"
    "src/nnf/ml/param_initializers.cpp"
    "src/nnf/tensor/tensor_base.cpp"
    "src/nnf/tensor/indexing.cpp"
    "src/nnf/tensor/tensor.cpp"
    "src/nnf/tensor/views.cpp"
)

if(BUILD_TESTS)
    platformer_add_executable(nnf_tests CLI VERSION "${PROJECT_VERSION}")
    target_link_libraries(nnf_tests PRIVATE nnf)
    target_link_libraries(nnf_tests PRIVATE Catch2::Catch2WithMain)
    target_sources(nnf_tests PRIVATE
        "tests/tensor_unit_tests/tensor_construction.test.cpp"
        "tests/tensor_unit_tests/tensor_data_modification.test.cpp"
        "tests/tensor_unit_tests/tensor_dimension_modification.test.cpp"
        "tests/tensor_unit_tests/tensor_indexing.test.cpp"
        "tests/tensor_unit_tests/tensor_operations.test.cpp"
        "tests/tensor_unit_tests/tensor_reduction.test.cpp"
        "tests/tensor_unit_tests/tensor_static_helpers.test.cpp"
    )
endif()

if(BUILD_BENCHMARKS)
    platformer_add_executable(nnf_benchmarks CLI VERSION "${PROJECT_VERSION}")
    target_link_libraries(nnf_benchmarks PRIVATE nnf)
    target_link_libraries(nnf_benchmarks PRIVATE benchmark::benchmark_main)
    target_sources(nnf_benchmarks PRIVATE
        "benchmarks/autodiff/conv2d_operation.bench.cpp"
        "benchmarks/autodiff/flatten_operation.bench.cpp"
        "benchmarks/autodiff/matvecmul_operation.bench.cpp"
        "benchmarks/autodiff/relu_operation.bench.cpp"
        "benchmarks/autodiff/reshape_operation.bench.cpp"
        "benchmarks/ml/sample_models.bench.cpp"
        "benchmarks/tensor/tensor_addition_subtraction.bench.cpp"
        "benchmarks/tensor/tensor_conv2d.bench.cpp"
        "benchmarks/tensor/tensor_matmul.bench.cpp"
    )
endif()

platformer_add_cmake_package(nnf
    TARGETS nnf
    VERSION "${PROJECT_VERSION}"
    DESCRIPTION "Neural network framework"
    NAMESPACE nnf
)
