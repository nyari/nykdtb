add_library(nykdtb_lib)

file(GLOB_RECURSE NYKDTB_LIB_UNIT_FILES "*.cpp")

target_sources(nykdtb_lib
  PRIVATE ${NYKDTB_LIB_UNIT_FILES}
)

file(GLOB_RECURSE NYKDTB_LIB_PUBLIC_HEADERS "nykdtb/*.hpp")

target_sources(nykdtb_lib
  PUBLIC
  FILE_SET nykdtb_lib_public_headers_file_set
  TYPE HEADERS
  FILES ${NYKDTB_LIB_PUBLIC_HEADERS}
)

install(
  TARGETS nykdtb_lib
  FILE_SET nykdtb_lib_public_headers_file_set
  DESTINATION "nykdtb"
)

set_property(TARGET nykdtb_lib PROPERTY CXX_STANDARD 17)