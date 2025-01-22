set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/aabb.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/thrust_helper.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

add_definitions(-DDIMENSION=${DIMENSION})
add_definitions(-DDATA_WIDTH=${DATA_WIDTH})

OPTIX_add_sample_executable( dbscan target_name
  dbscan.cpp
  dbscan.cu
  aabb.cu
  thrust_helper.cu
  dbscan.h
  state.h
  timer.h
  func.h
  util.cpp
  optix.cpp
  grid.cpp
  # OPTIONS -rdc true
  )

# target_link_libraries( ${target_name}
#   ${CUDA_LIBRARIES}
# )
  
# 使用 std::parallel_sort
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
if (OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
target_link_libraries( ${target_name}
 ${CUDA_LIBRARIES}
 OpenMP::OpenMP_CXX
 TBB::tbb
)