
# Intro1
add_executable(intro1 intro1.cpp)
target_link_libraries(
  intro1
  ${NECESSARY_LIBS}
)

# Intro2
add_executable(intro2 intro2.cpp)
target_link_libraries(
  intro2
  ${NECESSARY_LIBS}
)

# Bundle Adjustment
add_executable(bundle_adjustment bundle_adjustment.cpp)
target_link_libraries(
  bundle_adjustment
  ${NECESSARY_LIBS}
)

# Original Example
include_directories(${MATPLOTLIBCPP_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIR})
add_executable(matplotlib_sample matplotlib_sample.cpp)
target_link_libraries(
  matplotlib_sample
  ${NECESSARY_LIBS}
  ${PYTHON_LIBRARY}
)

add_executable(ceres_original_example ceres_original_example.cpp)
target_link_libraries(
  ceres_original_example
  ${NECESSARY_LIBS}
  ${PYTHON_LIBRARY}
)