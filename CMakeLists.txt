cmake_minimum_required(VERSION 3.0)
project(Little_CMS VERSION 2)

set(CMAKE_CXX_STANDARD 11)
include(GNUInstallDirs)

# Set optimization flags.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

include_directories(include)
add_library(lcms2 SHARED
        src/cmsalpha.c
        src/cmscam02.c
        src/cmscgats.c
        src/cmscnvrt.c
        src/cmserr.c
        src/cmsgamma.c
        src/cmsgmt.c
        src/cmshalf.c
        src/cmsintrp.c
        src/cmsio0.c
        src/cmsio1.c
        src/cmslut.cc
        src/cmsmd5.c
        src/cmsmtrx.c
        src/cmsnamed.c
        src/cmsopt.c
        src/cmspack.c
        src/cmspcs.c
        src/cmsplugin.c
        src/cmsps2.c
        src/cmssamp.c
        src/cmssm.c
        src/cmstypes.c
        src/cmsvirt.c
        src/cmswtpnt.c
        src/cmsxform.cc
        src/lcms2_internal.h)
set_target_properties(lcms2 PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER include/lcms2.h)

install(
        TARGETS lcms2
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
