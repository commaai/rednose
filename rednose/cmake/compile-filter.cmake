# Pretty ugly but meant to be an easy way to build rednose filters. Process is as follows:
#   1) Use python filter defintion to generate C++ source
#   2) Compile that C++ source into a shared library
function(build_rednose_filter REDNOSE_ROOT FILTER_NAME FILTER_SOURCE GENERATED_FOLDER)

  # 1) code gen
  message(STATUS "Code-generating filter ${FILTER_NAME} from ${FILTER_SOURCE} into ${GENERATED_FOLDER}")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/${FILTER_SOURCE} t ${GENERATED_FOLDER}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # 2) compile lib
  add_library(${FILTER_NAME} SHARED
    ${GENERATED_FOLDER}/${FILTER_NAME}.cpp
    ${REDNOSE_ROOT}/rednose/helpers/ekf_load.cc
    ${REDNOSE_ROOT}/rednose/helpers/ekf_sym.cc
  )

  target_include_directories(${FILTER_NAME}
    PRIVATE
    ${REDNOSE_ROOT}
    ${REDNOSE_ROOT}/rednose
    ${REDNOSE_ROOT}/rednose/helpers
  )

  set_target_properties(${FILTER_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${GENERATED_FOLDER}
    OUTPUT_NAME ${FILTER_NAME}
  )

  # Link against dl for dynamic loading
  target_link_libraries(${FILTER_NAME} PRIVATE ${CMAKE_DL_LIBS})

  # On ELF platforms, catch unresolved symbols at link time (optional but helpful)
  if(UNIX AND NOT APPLE)
    target_link_options(${FILTER_NAME} PRIVATE "-Wl,--no-undefined")
  endif()
endfunction()
