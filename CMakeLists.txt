cmake_minimum_required(VERSION 3.0.0)
project(ceres_solver_sample VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS 
  ${CMAKE_CXX_FLAGS} "-Wall -std=c++14")

# EIGEN DEPENDENCY
set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/../../3rdParty/eigen334/install/share/eigen3")
find_package(Eigen3 REQUIRED CONFIG PATHS ${Eigen3_DIR} NO_DEFAULT_PATH)

# CERES DEPENDENCY
set(CERES_LIB_DIR "${CMAKE_SOURCE_DIR}/../../3rdParty/google_ceres_solver/install/lib")
set(CERES_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../3rdParty/google_ceres_solver/install/include")

# Necessary Libs
set(NECESSARY_LIBS
  ceres glog lapack gflags blas ccolamd
  pthread csparse cholmod cxsparse
)

message("NECESSARY_LIBS : "  ${NECESSARY_LIBS})

include_directories(${CERES_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
link_directories(${CERES_LIB_DIR})

add_subdirectory(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

