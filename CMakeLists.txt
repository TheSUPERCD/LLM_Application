cmake_minimum_required(VERSION 3.10)
project(LLMInference)

set(CMAKE_CXX_STANDARD 17)
set(LLAMA_BUILD_COMMON On)

# For clangd intellisense to work properly
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp")

add_executable(main src/main.cpp src/LLMEngine.cpp) 

target_include_directories(main PRIVATE 
    include 
    "externals/llama.cpp/include" 
    "externals/llama.cpp/common"
)

target_link_libraries(main PRIVATE llama readline)
