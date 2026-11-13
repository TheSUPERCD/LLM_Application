cmake_minimum_required(VERSION 3.16)
project(LLMInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set common flags and optimizations
add_compile_options(-Wall -Wextra -Wpedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -mtune=native -fno-math-errno
                      -fno-semantic-interposition)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
    add_compile_options(-mprefer-vector-width=256)
  endif()
endif()

set(LLAMA_BUILD_COMMON ON)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build server" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "Build mtmd" FORCE)
set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan backend")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
endif()

# Clangd Intellisense export
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/compile_commands.json")
  execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_BINARY_DIR}/compile_commands.json
          ${CMAKE_SOURCE_DIR}/compile_commands.json
  )
endif()

# ---------------------------------------------------------------------------
# llama.cpp dependency
#
# Acquisition is NON-DESTRUCTIVE: an existing externals/llama.cpp checkout is
# never deleted, reset or re-cloned. The repository is only cloned when the
# directory is missing entirely (e.g. after a fresh `git clone` of this project
# without initialising the submodule).
#
# Local modifications to llama.cpp are stored as patch files in patches/ and
# applied automatically (and idempotently) after acquisition. Because the patch
# files are committed to this repository, the modifications survive a fresh
# clone on any machine.
#
# Workflow when changing llama.cpp:
#   1. edit files under externals/llama.cpp
#   2. run ./scripts/update-llama-patch.sh   (regenerates the patch)
#   3. commit the updated patch file
# ---------------------------------------------------------------------------
set(LLAMA_CPP_GIT_REPOSITORY "https://github.com/ggml-org/llama.cpp.git" CACHE STRING
    "Git repository used to clone llama.cpp when externals/llama.cpp is missing")
set(LLAMA_CPP_GIT_TAG "52d42686560a9e8f441f9b9780c8890c37d2802d" CACHE STRING
    "Git tag/commit that externals/llama.cpp is pinned to")
set(LLAMA_CPP_APPLY_PATCHES ON CACHE BOOL
    "Apply patches/llama.cpp-*.patch to the llama.cpp checkout")

set(LLAMA_CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp")
set(LLAMA_CPP_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/llama-build")
set(LLAMA_CPP_PATCH_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/patches")

find_package(Git REQUIRED)

# 1. Acquire the source tree only if it is missing. Never touch an existing one.
if(NOT EXISTS "${LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt")
  message(STATUS "llama.cpp not found in externals/ - cloning ${LLAMA_CPP_GIT_REPOSITORY} @ ${LLAMA_CPP_GIT_TAG}")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/externals")
  execute_process(
      COMMAND "${GIT_EXECUTABLE}" clone "${LLAMA_CPP_GIT_REPOSITORY}" "${LLAMA_CPP_SOURCE_DIR}"
      RESULT_VARIABLE LLAMA_CPP_CLONE_RESULT)
  if(NOT LLAMA_CPP_CLONE_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Failed to clone llama.cpp from ${LLAMA_CPP_GIT_REPOSITORY} into ${LLAMA_CPP_SOURCE_DIR}.\n"
        "Clone it manually, or pass -DLLAMA_CPP_GIT_REPOSITORY=<url> to use a different remote.")
  endif()
  execute_process(
      COMMAND "${GIT_EXECUTABLE}" checkout "${LLAMA_CPP_GIT_TAG}"
      WORKING_DIRECTORY "${LLAMA_CPP_SOURCE_DIR}"
      RESULT_VARIABLE LLAMA_CPP_CHECKOUT_RESULT)
  if(NOT LLAMA_CPP_CHECKOUT_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to check out llama.cpp at ${LLAMA_CPP_GIT_TAG}")
  endif()
else()
  message(STATUS "llama.cpp found in externals/ - using existing checkout (no re-clone)")
endif()

# 2. Apply the project's local patches. Idempotent: a patch that is already
#    applied is detected via a reverse dry-run and skipped.
if(LLAMA_CPP_APPLY_PATCHES AND EXISTS "${LLAMA_CPP_PATCH_DIR}")
  file(GLOB LLAMA_CPP_PATCHES CONFIGURE_DEPENDS "${LLAMA_CPP_PATCH_DIR}/llama.cpp-*.patch")
  list(SORT LLAMA_CPP_PATCHES)
  foreach(LLAMA_CPP_PATCH IN LISTS LLAMA_CPP_PATCHES)
    get_filename_component(LLAMA_CPP_PATCH_NAME "${LLAMA_CPP_PATCH}" NAME)

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --check --reverse "${LLAMA_CPP_PATCH}"
        WORKING_DIRECTORY "${LLAMA_CPP_SOURCE_DIR}"
        RESULT_VARIABLE LLAMA_CPP_PATCH_ALREADY_APPLIED
        OUTPUT_QUIET ERROR_QUIET)
    if(LLAMA_CPP_PATCH_ALREADY_APPLIED EQUAL 0)
      message(STATUS "llama.cpp patch already applied: ${LLAMA_CPP_PATCH_NAME}")
      continue()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --check "${LLAMA_CPP_PATCH}"
        WORKING_DIRECTORY "${LLAMA_CPP_SOURCE_DIR}"
        RESULT_VARIABLE LLAMA_CPP_PATCH_CHECK
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT LLAMA_CPP_PATCH_CHECK EQUAL 0)
      message(WARNING
          "llama.cpp patch '${LLAMA_CPP_PATCH_NAME}' cannot be applied to "
          "${LLAMA_CPP_SOURCE_DIR} (the checkout has diverged from the pinned "
          "commit, or the patch is only partially applied). The build will "
          "continue with the current working tree. To restore the expected "
          "state, run 'git -C externals/llama.cpp checkout -- .' and re-run CMake.")
      continue()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply "${LLAMA_CPP_PATCH}"
        WORKING_DIRECTORY "${LLAMA_CPP_SOURCE_DIR}"
        RESULT_VARIABLE LLAMA_CPP_PATCH_RESULT
        ERROR_VARIABLE LLAMA_CPP_PATCH_ERROR)
    if(NOT LLAMA_CPP_PATCH_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to apply llama.cpp patch '${LLAMA_CPP_PATCH_NAME}':\n${LLAMA_CPP_PATCH_ERROR}")
    endif()
    message(STATUS "Applied llama.cpp patch: ${LLAMA_CPP_PATCH_NAME}")
  endforeach()
endif()

# 3. Build llama.cpp from the (possibly patched) source tree.
add_subdirectory("${LLAMA_CPP_SOURCE_DIR}" "${LLAMA_CPP_BINARY_DIR}")

# LLM Engine Library
add_library(llm_engine src/LLMEngine.cpp)
target_include_directories(llm_engine PUBLIC include)
target_link_libraries(llm_engine PUBLIC llama mtmd)

# CLI Application
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE llm_engine readline)

# Test Suite
enable_testing()
add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE llm_engine)
add_test(NAME test_engine COMMAND test_engine)
set_tests_properties(test_engine PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
