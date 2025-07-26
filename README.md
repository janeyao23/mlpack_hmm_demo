# mlpack Hidden Markov Model (HMM) demo

This repository contains a simple C++ program that demonstrates how to use the [mlpack](https://www.mlpack.org) library to build and work with a discrete Hidden Markov Model (HMM). The program constructs a two‑state HMM, prints its initial parameters, runs the Viterbi algorithm to predict hidden states for a sequence of observations, computes the log‑likelihood of the observations, and then retrains the model using the Baum‑Welch algorithm to update the parameters.

## Why build mlpack from source?

The mlpack project is under active development, and new releases appear frequently. To ensure reproducible builds, this repository does not depend on a pre‑installed version of mlpack. Instead, the provided GitHub Actions workflow checks out a specific tagged release of mlpack (version 4.6.2, released **May 22 2025**) and builds it together with the demo program. If you wish to use a different version of mlpack, change the `MLPACK_VERSION` variable in `.github/workflows/build.yml`.

## Building locally with CMake

You can build the demo manually on Windows (or any other platform) as follows:

1. Clone this repository and clone the mlpack source into a subdirectory named `mlpack` at the top level of the repository, checking out the desired tag (e.g. `mlpack-4.6.2`).
2. Ensure you have a C++17 compiler and mlpack’s dependencies installed: Armadillo, Boost, ensmallen and OpenBLAS. On Windows you can use [vcpkg](https://github.com/microsoft/vcpkg) to install `armadillo`, `boost-program-options`, `boost-serialization` and other required packages.
3. Run CMake to configure a 32‑bit build and specify the toolchain file provided by vcpkg. For example:

   ```sh
   cmake -S . -B build -A Win32 \
         -DMLPACK_SRC_DIR="${PWD}/mlpack" \
         -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" \
         -DMLPACK_BUILD_TESTS=OFF -DMLPACK_BUILD_BENCHMARKS=OFF
   cmake --build build --config Release --parallel
   ```

This will produce `hmm_demo.exe` in the `build/Release` directory.

## Continuous integration

The `.github/workflows/build.yml` workflow demonstrates how to automate the build on GitHub using Actions. The job runs on `windows-latest`, clones mlpack at the specified tag, installs dependencies with vcpkg and builds the demo in 32‑bit mode. The workflow ensures that the mlpack version used is the exact tagged release by checking out the tag `4.6.2` (May 22 2025) during the build.

## Notes

* The program uses `mlpack::HMM<mlpack::distribution::DiscreteDistribution>` from mlpack’s C++ API to build and manipulate the HMM. The HMM class exposes the initial state distribution, transition matrix and emission distributions via the `Initial()`, `Transition()` and `Emission()` member functions.
* The example uses a discrete observation alphabet of size 2 and defines emission probabilities directly in the source code. You can modify the probabilities, initial vector or transition matrix to explore different HMM behaviours.
        
