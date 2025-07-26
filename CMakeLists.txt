cmake_minimum_required(VERSION 3.14)
project(hmm_demo CXX)

# Use C++17 throughout the project.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow the user to specify the location of a pre checked‑out mlpack tree.
set(MLPACK_SRC_DIR "" CACHE PATH "Path to the mlpack source tree")

# If MLPACK_SRC_DIR is set we simply add its include directories.  The
# Hidden Markov Model is implemented entirely in header files so linking
# against the mlpack library is optional for this example.  If the user
# provides the mlpack source inside the project (e.g. by cloning it into
# the 'mlpack' directory) then we will add it as a subdirectory and link
# against the built library.
if(MLPACK_SRC_DIR)
    message(STATUS "Using mlpack from ${MLPACK_SRC_DIR}")
    include_directories(${MLPACK_SRC_DIR}/src)
    include_directories(${MLPACK_SRC_DIR}/src/mlpack)
else()
    if(EXISTS ${CMAKE_SOURCE_DIR}/mlpack/CMakeLists.txt)
        message(STATUS "Found mlpack source in ${CMAKE_SOURCE_DIR}/mlpack; adding as subdirectory.")
        add_subdirectory(${CMAKE_SOURCE_DIR}/mlpack mlpack-build EXCLUDE_FROM_ALL)
        include_directories(${CMAKE_SOURCE_DIR}/mlpack/src)
        include_directories(${CMAKE_SOURCE_DIR}/mlpack/src/mlpack)
        set(MLPACK_LIB mlpack)
    else()
        message(FATAL_ERROR "mlpack source tree not found.  Please set MLPACK_SRC_DIR or clone mlpack into the 'mlpack' subdirectory.")
    endif()
endif()

# Create the executable from our source files.
add_executable(hmm_demo src/main.cpp)

# Link against the mlpack library if it exists.  This is not strictly
# necessary for the HMM example because the template code is header‑only,
# but linking ensures that any other mlpack functionality used in the
# future will resolve symbols correctly.
if(TARGET mlpack)
    target_link_libraries(hmm_demo PRIVATE mlpack)
endif()

# On Windows we rely on the build environment to provide OpenBLAS and other
# dependencies via a package manager such as vcpkg.  See the
# `.github/workflows/build.yml` file for an example.
