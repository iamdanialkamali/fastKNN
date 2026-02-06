cmake_minimum_required(VERSION 3.18)
project(fastknn VERSION 1.0.0 LANGUAGES C CXX)

# Options for feature flags
option(FASTKNN_ENABLE_OPENMP "Enable OpenMP backend" ON)
option(FASTKNN_ENABLE_MPI "Enable MPI backends" ON)
option(FASTKNN_ENABLE_CUDA "Enable CUDA backends" ON)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies
set(FASTKNN_HAS_OPENMP FALSE)
set(FASTKNN_HAS_MPI FALSE)
set(FASTKNN_HAS_CUDA FALSE)

if(FASTKNN_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        set(FASTKNN_HAS_OPENMP TRUE)
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING "OpenMP requested but not found - OpenMP backend disabled")
    endif()
endif()

if(FASTKNN_ENABLE_MPI)
    find_package(MPI)
    if(MPI_C_FOUND)
        set(FASTKNN_HAS_MPI TRUE)
        message(STATUS "MPI enabled")
    else()
        message(WARNING "MPI requested but not found - MPI backends disabled")
    endif()
endif()

if(FASTKNN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(FASTKNN_HAS_CUDA TRUE)
        set(CMAKE_CUDA_STANDARD 11)
        # Set CUDA architectures supported by CUDA 12.0+
        # 75: Turing (RTX 20xx), 80: Ampere (A100, RTX 30xx)
        # 86: Ampere (RTX 30xx mobile), 89: Ada (RTX 40xx), 90: Hopper (H100)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
        message(STATUS "CUDA enabled")
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(WARNING "CUDA requested but not found - CUDA backends disabled")
    endif()
endif()

# Library sources
set(FASTKNN_CORE_SOURCES
    src/core/core.c
    src/core/io.c
    src/core/distance.c
    src/core/timer.c
    src/core/dispatcher.c
)

set(FASTKNN_BACKEND_SOURCES
    src/backend/serial.c
)

# Add OpenMP backend if available
if(FASTKNN_HAS_OPENMP)
    list(APPEND FASTKNN_BACKEND_SOURCES src/backend/openmp.c)
endif()

# Add MPI backends if available
if(FASTKNN_HAS_MPI)
    list(APPEND FASTKNN_BACKEND_SOURCES
        src/backend/mpi_train.c
        src/backend/mpi_test.c
    )
else()
    # Add MPI stubs when MPI is not available
    list(APPEND FASTKNN_BACKEND_SOURCES
        src/backend/mpi_stubs.c
    )
endif()

# Add CUDA backends if available
if(FASTKNN_HAS_CUDA)
    list(APPEND FASTKNN_BACKEND_SOURCES
        src/backend/cuda.cu
        src/backend/cuda_stubs.cu
    )
    # Add CUDA+MPI hybrid if both are available
    if(FASTKNN_HAS_MPI)
        list(APPEND FASTKNN_BACKEND_SOURCES
            src/backend/cuda_mpi.cu
        )
        message(STATUS "CUDA+MPI hybrid backend enabled")
    endif()
else()
    # Add CUDA stubs when CUDA is not available
    list(APPEND FASTKNN_BACKEND_SOURCES
        src/backend/cuda_stubs.c
    )
endif()

# Option to build shared library for Python bindings
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Create the library
if(BUILD_SHARED_LIBS)
    add_library(fastknn SHARED ${FASTKNN_CORE_SOURCES} ${FASTKNN_BACKEND_SOURCES})
else()
    add_library(fastknn ${FASTKNN_CORE_SOURCES} ${FASTKNN_BACKEND_SOURCES})
endif()

# Set include directories
target_include_directories(fastknn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add compile definitions
if(FASTKNN_HAS_OPENMP)
    target_compile_definitions(fastknn PRIVATE FASTKNN_ENABLE_OPENMP)
    target_link_libraries(fastknn PUBLIC OpenMP::OpenMP_C)
endif()

if(FASTKNN_HAS_MPI)
    target_compile_definitions(fastknn PRIVATE FASTKNN_ENABLE_MPI)
    target_include_directories(fastknn PRIVATE ${MPI_C_INCLUDE_DIRS})
    target_link_libraries(fastknn PUBLIC ${MPI_C_LIBRARIES})
endif()

if(FASTKNN_HAS_CUDA)
    target_compile_definitions(fastknn PRIVATE FASTKNN_ENABLE_CUDA)
    set_target_properties(fastknn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Link math library
target_link_libraries(fastknn PUBLIC m)

# Install targets
install(TARGETS fastknn
    EXPORT FastKNNTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/fastknn
    DESTINATION include
)

# Install CMake package config
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FastKNNConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/FastKNNConfig.cmake
    INSTALL_DESTINATION lib/cmake/FastKNN
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/FastKNNConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/FastKNNConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/FastKNNConfigVersion.cmake
    DESTINATION lib/cmake/FastKNN
)

install(EXPORT FastKNNTargets
    FILE FastKNNTargets.cmake
    NAMESPACE fastknn::
    DESTINATION lib/cmake/FastKNN
)

# Summary
message(STATUS "==================================================")
message(STATUS "FastKNN Configuration Summary")
message(STATUS "==================================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Library type: ${BUILD_SHARED_LIBS}")
message(STATUS "OpenMP backend: ${FASTKNN_HAS_OPENMP}")
message(STATUS "MPI backends: ${FASTKNN_HAS_MPI}")
message(STATUS "CUDA backends: ${FASTKNN_HAS_CUDA}")
message(STATUS "==================================================")
