cmake_minimum_required(VERSION 3.10)
project(LLMInference)

set(CMAKE_CXX_STANDARD 17)
set(LLAMA_BUILD_COMMON On)
set(GGML_VULKAN On)

# For clangd intellisense to work properly
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)
# Ensure submodules are populated even if the externals directory is removed
include(FetchContent)
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        52d42686560a9e8f441f9b9780c8890c37d2802d
    SOURCE_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp"
)
FetchContent_MakeAvailable(llama)

add_executable(main src/main.cpp src/LLMEngine.cpp) 

target_include_directories(main PRIVATE 
    include 
    "externals/llama.cpp/include" 
    "externals/llama.cpp/common"
)

target_link_libraries(main PRIVATE llama readline)
