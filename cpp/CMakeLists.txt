cmake_minimum_required(VERSION 3.12)
project(BiRefNet)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add source files
set(SOURCES
	main.cpp
	utils.cpp
	birefnet.cpp
    # ... add more source files here if needed
)

# Add headers
set(HEADERS
	utils.h
	birefnet.h
    # ... add more header files here if needed
)

# Find and include CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find and include OpenCV
set(OpenCV_DIR "D:\\opencv\\build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "D:\\0_dp\\library\\TensorRT-10.3.0.26")

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

# Read TensorRT version
file(STRINGS "${TENSORRT_DIR}/include/NvInferVersion.h" tensorrt_version REGEX "#define NV_TENSORRT_MAJOR +[0-9]+")
string(REGEX MATCH "[0-9]+" tensorrt_version_major ${tensorrt_version})

# Check TensorRT version and set library accordingly
if (WIN32)
    if (tensorrt_version_major EQUAL 10)
        set(TENSORRT_LIBS nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
    else()
        set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
    endif()
else()
    if (tensorrt_version_major EQUAL 10)
        set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
    else()
        set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
    endif()
endif()

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)
