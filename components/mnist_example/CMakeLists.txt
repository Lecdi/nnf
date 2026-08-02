cmake_minimum_required(VERSION 3.23)
project(mnist_example VERSION 0.0.0)

set(MNIST_MODEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/mnist_model")
set(MNIST_MODEL_FILENAME "model.bin")
set(MNIST_MODEL "${MNIST_MODEL_OUTPUT_DIR}/${MNIST_MODEL_FILENAME}")
set(MNIST_APP_TO_MODEL_DIR_RELATIVE_PATH "../share/mnist_app/model")
set(MNIST_APP_TO_MODEL_RELATIVE_PATH "${MNIST_APP_TO_MODEL_DIR_RELATIVE_PATH}/${MNIST_MODEL_FILENAME}")
set(MNIST_MODEL_STAGED_DIR "$<TARGET_FILE_DIR:mnist_app>/${MNIST_APP_TO_MODEL_DIR_RELATIVE_PATH}")
set(MNIST_MODEL_STAGED "${MNIST_MODEL_STAGED_DIR}/${MNIST_MODEL_FILENAME}")
set(MNIST_TRAIN_DATA_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/mnist")
set(MNIST_TRAIN_DATA_FILES
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

file(MAKE_DIRECTORY "${MNIST_TRAIN_DATA_DIRECTORY}")
foreach(filename IN LISTS MNIST_TRAIN_DATA_FILES)
    if(NOT EXISTS "${MNIST_TRAIN_DATA_DIRECTORY}/${filename}")
        file(DOWNLOAD
            "https://storage.googleapis.com/cvdf-datasets/mnist/${filename}"
            "${MNIST_TRAIN_DATA_DIRECTORY}/${filename}"
            SHOW_PROGRESS
        )
    endif()
endforeach()

platformer_add_library(mnist_core STATIC
    PUBLIC_INCLUDE_DIRS "src"
)
target_link_libraries(mnist_core PUBLIC
    imgui_sdl2
    ZLIB::ZLIB
    nnf::nnf
)
target_sources(mnist_core PRIVATE
    "src/mnist/app.hpp"
    "src/mnist/app.cpp"
    "src/mnist/construct_model.hpp"
    "src/mnist/construct_model.cpp"
    "src/mnist/load_data.hpp"
    "src/mnist/load_data.cpp"
)
target_compile_definitions(mnist_core PUBLIC
    "MODEL_RELATIVE_PATH=\"${MNIST_APP_TO_MODEL_RELATIVE_PATH}\""
    "OUTPUT_DIR=\"${MNIST_MODEL_OUTPUT_DIR}\""
    "OUTPUT_FILENAME=\"${MNIST_MODEL_FILENAME}\""
    "TRAIN_DATA_DIR=\"${MNIST_TRAIN_DATA_DIRECTORY}\""
)

platformer_add_executable(mnist_train_full CLI VERSION "${PROJECT_VERSION}")
target_link_libraries(mnist_train_full PRIVATE
    mnist_core
)
target_sources(mnist_train_full PRIVATE
    "src/train_full.main.cpp"
)

platformer_add_executable(mnist_train_partial CLI VERSION "${PROJECT_VERSION}")
target_link_libraries(mnist_train_partial PRIVATE
    mnist_core
)
target_sources(mnist_train_partial PRIVATE
    "src/train_partial.main.cpp"
)

add_custom_target(mnist_stage_model
    COMMAND
        "${CMAKE_COMMAND}"
        "-DMNIST_MODEL=${MNIST_MODEL}"
        "-DMNIST_MODEL_STAGED_DIR=${MNIST_MODEL_STAGED_DIR}"
        "-DMNIST_MODEL_STAGED=${MNIST_MODEL_STAGED}"
        "-P"
        "${CMAKE_CURRENT_SOURCE_DIR}/StageModel.cmake"
    VERBATIM
)

platformer_add_executable(mnist_app GUI
    DISPLAY_NAME "MNIST Model Tester"
    VERSION "${PROJECT_VERSION}"
    DESCRIPTION "MNIST Model Tester"
    VENDOR "Lecdi"
    ICONS "resources/mnist_app.icns" "resources/mnist_app.ico" "resources/mnist_app.png"
)
target_link_libraries(mnist_app PRIVATE
    mnist_core
    SDL2::SDL2main
)
target_sources(mnist_app PRIVATE
    "src/app.main.cpp"
)
add_dependencies(mnist_app mnist_stage_model)

platformer_add_application_package(mnist_app
    TARGETS mnist_app
    DISPLAY_NAME "MNIST Model Tester"
    VERSION "${PROJECT_VERSION}"
    DESCRIPTION "MNIST Model Tester"
    VENDOR "Lecdi"
    LICENSE "resources/LICENSE.txt"
    ICONS "resources/mnist_app.icns" "resources/mnist_app.ico" "resources/mnist_app.png"
    USE_COMPONENT "mnist_app"
)
install(
    FILES "${MNIST_MODEL}"
    DESTINATION "bin/${MNIST_APP_TO_MODEL_DIR_RELATIVE_PATH}"
    COMPONENT "mnist_app"
)
