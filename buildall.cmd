@echo off

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

rd /s /q build 2>nul
mkdir build 2>nul
cd build

cmake .. -G "Ninja Multi-Config" ^
-DGGML_BUILD_TESTS=OFF ^
-DGGML_BUILD_EXAMPLES=OFF ^
-DGGML_HIP=OFF ^
-DGGML_SYCL=OFF ^
-DGGML_NATIVE=OFF ^
-DGGML_AVX=OFF ^
-DGGML_AVX2=OFF ^
-DGGML_AVX512=OFF ^
-DGGML_BMI2=ON ^
-DGGML_FMA=OFF ^
-DGGML_F16C=OFF ^
-DGGML_SSE42=OFF ^
-DGGML_CPU_ALL_VARIANTS=ON ^
-DGGML_BACKEND_DL=ON ^
-DGGML_CUDA=ON ^
-DGGML_CUDA_GRAPHS=ON ^
-DGGML_VULKAN=ON
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..
