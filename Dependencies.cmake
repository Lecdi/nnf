cmake_minimum_required(VERSION 3.23)

include(FetchContent)

if(COMPONENT_MNIST_EXAMPLE)
    if(USE_SYSTEM_INSTALLATIONS)
        find_package(ZLIB QUIET)
    endif()
    if(NOT ZLIB_FOUND)
        set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
        set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(zlib
            GIT_REPOSITORY "https://github.com/madler/zlib.git"
            GIT_TAG "v1.3.1"
        )
        FetchContent_MakeAvailable(zlib)
        if(NOT TARGET ZLIB::ZLIB)
            add_library(ZLIB::ZLIB ALIAS zlibstatic)
        endif()
    endif()
endif()

if(BUILD_TESTS)
    if(USE_SYSTEM_INSTALLATIONS)
        find_package(Catch2 3 CONFIG QUIET)
    endif()
    if(NOT Catch2_FOUND)
        FetchContent_Declare(Catch2
            GIT_REPOSITORY "https://github.com/catchorg/Catch2.git"
            GIT_TAG "v3.5.4"
        )
        FetchContent_MakeAvailable(Catch2)
    endif()
endif()

if(BUILD_BENCHMARKS)
    if(USE_SYSTEM_INSTALLATIONS)
        find_package(benchmark CONFIG QUIET)
    endif()
    if(NOT benchmark_FOUND)
        FetchContent_Declare(benchmark
            GIT_REPOSITORY "https://github.com/google/benchmark.git"
            GIT_TAG "v1.9.4"
        )
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(benchmark)
    endif()
endif()

if(COMPONENT_MNIST_EXAMPLE)
    if(USE_SYSTEM_INSTALLATIONS)
        find_package(SDL2 2.28 CONFIG QUIET)
        if(SDL2_FOUND)
            set(SDL2_LIB "SDL2::SDL2")
        endif()
    endif()
    if(NOT SDL2_FOUND)
        FetchContent_Declare(SDL2
            GIT_REPOSITORY "https://github.com/libsdl-org/SDL.git"
            GIT_TAG "release-2.28.5"
        )
        FetchContent_MakeAvailable(SDL2)
        set(SDL2_LIB "SDL2::SDL2-static")
    endif()
    FetchContent_Declare(imgui
        GIT_REPOSITORY "https://github.com/ocornut/imgui.git"
        GIT_TAG "v1.91.0"
    )
    FetchContent_MakeAvailable(imgui)
    add_library(imgui_sdl2 STATIC
        "${imgui_SOURCE_DIR}/imgui.cpp"
        "${imgui_SOURCE_DIR}/imgui_draw.cpp"
        "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer2.cpp"
    )
    target_include_directories(imgui_sdl2 PUBLIC "${imgui_SOURCE_DIR}")
    target_link_libraries(imgui_sdl2 PUBLIC "${SDL2_LIB}")
endif()

FetchContent_Declare(platformer
    GIT_REPOSITORY "https://github.com/Lecdi/platformer.git"
    GIT_TAG "main"
)
FetchContent_MakeAvailable(platformer)
