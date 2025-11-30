cmake_minimum_required(VERSION 3.26)

project(QUBOBruteForcing CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(CUDAToolkit REQUIRED)
    

include_directories("${CUDAToolkit_INCLUDE_DIRS}")

option(WITH_PRODUCTION_COMPILATION "Turn off all debug infos and apply every optimization" 0)
option(WITH_PROFILING "Generate the files with cuda lineinfos" 0)
option(WITH_DEBUG_COMPILATION_CUDA "Generate Cuda Debug Code and use additional error checking in CUDA_CALL macro" 0)
option(WITH_ACTIVATE_CUDA_COMPILATION_INFO "activate PTX compilation info" 1)
option(WITH_OPENMP "Use openmp parallellization for the cpu code" 0)

if(WITH_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if(WITH_ACTIVATE_CUDA_COMPILATION_INFO)
    set(custom_flags "${custom_flags} -Xptxas -v")
endif()

if(UNIX)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC compile option
endif()

if(WITH_PRODUCTION_COMPILATION)
    set(WITH_PROFILING OFF)
    set(Architectures "50;61-real;75-real;80;86-real;89-real;90")
else()
    set(Architectures "native")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${Architectures})    
endif()
if(WITH_PROFILING)
    set(custom_flags "${custom_flags} --generate-line-info ")
endif()

if(WITH_DEBUG_COMPILATION_CUDA)
    set(custom_flags "${custom_flags} -g -G")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)
endif()

message(STATUS "NVCC Flags: ${custom_flags}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${custom_flags}")

add_executable(${PROJECT_NAME}
    cpu_brute_force.h
    cuda_debug.h
    cuda_timer.h
    cuda_util.h
    datatypes.h
    gpu_brute_force.h
    gpu_brute_force.cu
    main.cpp
    matrix.h
    matrix_reader.h
    qubo_brute_forcer.h 
    qubo_energy.h
    state_vector.h
    timer.h
)

message(STATUS "CUDA Architectures: ${Architectures}")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${Architectures}")


target_link_libraries(${PROJECT_NAME} 
    PRIVATE
    CUDA::cudart
)

if(WITH_OPENMP)
    target_link_libraries(${PROJECT_NAME} 
        PRIVATE
        OpenMP::OpenMP_CXX
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_OPENMP)
endif()


if(UNIX)
set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        LINK_FLAGS "-Wl,-rpath,$ORIGIN/")
endif()


install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION bin
)

install(DIRECTORY ../data/
    DESTINATION data
)

include(InstallRequiredSystemLibraries)

if(WIN32)
    # To install other/non-system 3rd party required libraries:
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleDependencies.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/BundleDependencies.cmake
        @ONLY
    )

    install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/BundleDependencies.cmake)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING "Install path" FORCE)
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

#copy data files to build folder
file(COPY ${CMAKE_SOURCE_DIR}/../data_dense DESTINATION ${CMAKE_BINARY_DIR})
