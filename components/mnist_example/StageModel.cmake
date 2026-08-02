cmake_minimum_required(VERSION 3.23)

message(STATUS "Staging model from \"${MNIST_MODEL}\"")

if(NOT EXISTS "${MNIST_MODEL}")
    message(FATAL_ERROR "Model file not found; must train the model before building the tester app")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" "-E" "make_directory" "${MNIST_MODEL_STAGED_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
    COMMAND "${CMAKE_COMMAND}" "-E" "copy" "${MNIST_MODEL}" "${MNIST_MODEL_STAGED}"
    COMMAND_ERROR_IS_FATAL ANY
)
