cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(dino)

find_package(OpenCV REQUIRED)

set(Torch_DIR "libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${TORCH_INCLUDE_DIRS} )

add_executable(dino main.cpp)
target_link_libraries(dino ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET dino PROPERTY CXX_STANDARD 17)


