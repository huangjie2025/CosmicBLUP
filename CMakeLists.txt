cmake_minimum_required(VERSION 3.15)
project(CosmicBLUP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

configure_file(
    src/config/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h
    @ONLY
)

find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(zstd REQUIRED)
find_package(OpenMP QUIET)
find_package(BLAS QUIET)
find_package(LAPACK QUIET)

add_library(cosmicblup_core STATIC
    src/core/a_matrix.cpp
    src/core/adaptive_trace_estimator.cpp
    src/core/bgen_reader.cpp
    src/core/dense_solver.cpp
    src/core/design.cpp
    src/core/eigen_vce.cpp
    src/core/grm_builder.cpp
    src/core/matrix_inversion.cpp
    src/core/matrix_io.cpp
    src/core/mme_builder.cpp
    src/core/pgen_reader.cpp
    src/core/plink_reader.cpp
    src/core/qc_engine.cpp
    src/core/qc_utils.cpp
    src/core/slq_estimator.cpp
    src/core/stcg_vce.cpp
    src/core/upper_packed_binary.cpp
    src/core/v_matrix.cpp
    src/core/vce.cpp
)
target_include_directories(cosmicblup_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
target_link_libraries(cosmicblup_core PUBLIC Eigen3::Eigen ZLIB::ZLIB)

if(TARGET zstd::libzstd)
    target_link_libraries(cosmicblup_core PUBLIC zstd::libzstd)
elseif(TARGET zstd::libzstd_static)
    target_link_libraries(cosmicblup_core PUBLIC zstd::libzstd_static)
elseif(TARGET zstd::libzstd_shared)
    target_link_libraries(cosmicblup_core PUBLIC zstd::libzstd_shared)
else()
    message(FATAL_ERROR "zstd package found but no supported CMake target is available")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(cosmicblup_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(BLAS_FOUND)
    target_link_libraries(cosmicblup_core PUBLIC ${BLAS_LIBRARIES})
    target_compile_definitions(cosmicblup_core PUBLIC EIGEN_USE_BLAS)
endif()
if(LAPACK_FOUND)
    target_link_libraries(cosmicblup_core PUBLIC ${LAPACK_LIBRARIES})
    target_compile_definitions(cosmicblup_core PUBLIC EIGEN_USE_LAPACKE)
endif()

add_executable(cosmicblup
    src/app/main.cpp
    src/app/blup_app.cpp
    src/app/cli.cpp
    src/app/runtime.cpp
    src/io/phenotype_loader.cpp
    src/io/relationship_builder.cpp
    src/model/model_pipeline.cpp
    src/model/random_regression.cpp
    src/variance/variance_estimator.cpp
    src/solve/mixed_model_solver.cpp
    src/solve/result_writer.cpp
    src/tasks/auxiliary_tasks.cpp
)

target_include_directories(cosmicblup PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
    ${CMAKE_CURRENT_SOURCE_DIR}/src/app
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/variance
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solve
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tasks
)
target_link_libraries(cosmicblup PRIVATE cosmicblup_core)

if(MSVC)
    target_compile_options(cosmicblup PRIVATE /O2 /W4)
else()
    target_compile_options(cosmicblup PRIVATE -O3 -Wall -Wextra)
endif()

include(GNUInstallDirs)
install(TARGETS cosmicblup RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES README.md LICENSE CHANGELOG.md THIRD_PARTY_NOTICES.md VERSION DESTINATION .)
install(DIRECTORY docs/ DESTINATION docs)
install(DIRECTORY examples/ DESTINATION examples)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)

    function(cosmicblup_add_test name timeout)
        add_test(
            NAME ${name}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_case.py
                --case ${name}
                --bin $<TARGET_FILE:cosmicblup>
                --source ${CMAKE_CURRENT_SOURCE_DIR}
                --work ${CMAKE_CURRENT_BINARY_DIR}/Testing/work/${name}
        )
        set_tests_properties(${name} PROPERTIES
            TIMEOUT ${timeout}
            ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
        )
    endfunction()

    cosmicblup_add_test(cli_contract 30)
    cosmicblup_add_test(relationship_io 60)
    cosmicblup_add_test(singletrait_models 120)
    cosmicblup_add_test(repeatability 120)
    cosmicblup_add_test(random_regression 180)
    cosmicblup_add_test(multitrait 180)
    cosmicblup_add_test(auxiliary_tasks 120)
endif()
