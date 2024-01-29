set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/aabb.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

add_definitions(-DDIMENSION=${DIMENSION})
add_definitions(-DDATA_WIDTH=${DATA_WIDTH})

OPTIX_add_sample_executable( dbscan target_name
  dbscan.cpp
  dbscan.cu
  aabb.cu
  dbscan.h
  state.h
  timer.h
  util.cpp
  optix.cpp
  # OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
