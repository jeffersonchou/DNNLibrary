if (BUILD_BIN)
    add_executable(nnapi_benchmark
        nnapi_benchmark.cpp)

    target_link_libraries(nnapi_benchmark
        dnnlibrary)

    add_executable(dnn_infer_simple
        stb_image.h
        dnn_infer_simple.cpp)

    target_link_libraries(dnn_infer_simple
        dnnlibrary)

    add_executable(dnn_simplest
        dnn_simplest.cpp)

    target_link_libraries(dnn_simplest
        dnnlibrary)
endif()
