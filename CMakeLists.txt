cmake_minimum_required(VERSION 3.30)


### Command line arguments
option(MLP_XMOS "MLP_XMOS")


if (${MLP_XMOS})

    ##
    # XMOS private build (tests)
    ##

    message("---")
    message("- XMOS test build")
    message("---")

    set(CMAKE_CXX_STANDARD 14)

    project("MLP_XMOS")

    set(COMMON_FLAGS "-target=XK-EVK-XU316 -Wall -DMLP_VERBOSE -D__XS3A__ -fxscope ${CMAKE_CURRENT_SOURCE_DIR}/config.xscope")
    set(CMAKE_C_FLAGS ${COMMON_FLAGS})
    set(CMAKE_CXX_FLAGS ${COMMON_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS "-fxscope -Wl --gc-sections -report")

    set(XMOS_XE "MLP_XMOS")

    set(XMOS_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/MLP.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/Probe.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/FuncLearnTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/xmos_par_main.c"
        #"${CMAKE_CURRENT_SOURCE_DIR}/tests/_TestMain.cpp"
    )
    
    set(XMOS_INCLUDES
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/deps"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests")

    set(XMOS_CXX_COMPILE_OPTIONS)

    set(XMOS_C_COMPILE_OPTIONS) 

    add_executable(${XMOS_XE} ${XMOS_SOURCES})
    target_include_directories(${XMOS_XE} PUBLIC ${XMOS_INCLUDES})
    #target_compile_features(${XMOS_XE} PUBLIC cxx_std_14)
    target_compile_options(${XMOS_XE} PUBLIC -Wall -DMLP_VERBOSE -D__XS3A__)
    #target_link_libraries(${XMOS_XE} PUBLIC
    #    "/home/virgult/Frameworks/XMOS/XTC/15.2.1/target/lib/xs3a/libxscope.a")

elseif (CMAKE_SYSTEM_NAME STREQUAL "XCORE_XS3A")
    ##
    # XMOS build
    ##

    message("--- MLP ---")

    set(MLP_HEADERS
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/deps
    )
    set(MLP_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/MLP.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/FuncLearnTest.cpp
    )

    # Append in local scope
    list(APPEND MEML_HEADERS ${MLP_HEADERS})
    list(APPEND MEML_SOURCES ${MLP_SOURCES})

    # Propagate in parent scope
    set(MEML_HEADERS ${MEML_HEADERS} PARENT_SCOPE)
    set(MEML_SOURCES ${MEML_SOURCES} PARENT_SCOPE)

else ()

    ##
    # PC build
    ##
    cmake_policy(SET CMP0048 NEW)

    project(MLP VERSION 1.0)

    set(SHARED_LIB_NAME ${PROJECT_NAME})
    set(STATIC_LIB_NAME ${PROJECT_NAME}_static)
    set(MLP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/MLP.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/src/Probe.cpp")
    set(MLP_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/MLP.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/Layer.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/Node.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/Sample.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/Utils.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/Loss.h")

    # FuncLearnTest
    set(FUNCLEARN_EXEC_NAME FuncLearnTest)
    set(FUNCLEARN_EXEC_MAIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/FuncLearnTest.cpp")
    set(FUNCLEARN_DEFINES FUNCLEARN_MAIN MLP_VERBOSE)
    # LayerTest
    set(LAYER_EXEC_NAME LayerTest)
    set(LAYER_EXEC_MAIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/LayerTest.cpp")
    set(LAYER_DEFINES LAYERTEST_MAIN)
    # LossTest
    set(LOSS_EXEC_NAME LossTest)
    set(LOSS_EXEC_MAIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/LossTest.cpp")
    set(LOSS_DEFINES LOSSTEST_MAIN)
    # MLPTest
    set(MLP_EXEC_NAME MLPTest)
    set(MLP_EXEC_MAIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/MLPTest.cpp")
    set(MLP_DEFINES MLPTEST_MAIN MLP_VERBOSE)
    # Unit test main
    set(UNITTESTS_EXEC_NAME UnitTests)
    set(UNITTESTS_EXEC_MAIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/_TestMain.cpp")
    set(UNITTESTS_DEFINES)

    #### STATIC LIBRARY
    add_library(${STATIC_LIB_NAME} STATIC ${MLP_HEADER_FILES} ${MLP_SOURCE_FILES})
    set_target_properties(${STATIC_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
    set_target_properties(${STATIC_LIB_NAME} PROPERTIES SOVERSION 1)
    set_target_properties(${STATIC_LIB_NAME} PROPERTIES PUBLIC_HEADER "${MLP_HEADER_FILES}")
    target_include_directories(${STATIC_LIB_NAME} PRIVATE ".")
    target_include_directories(${STATIC_LIB_NAME} PRIVATE "src")
    target_include_directories(${STATIC_LIB_NAME} PRIVATE "deps")


    #### SHARED LIBRARY
    add_library(${SHARED_LIB_NAME} SHARED ${MLP_HEADER_FILES} ${MLP_SOURCE_FILES})
    set_target_properties(${SHARED_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
    set_target_properties(${SHARED_LIB_NAME} PROPERTIES SOVERSION 1)
    set_target_properties(${SHARED_LIB_NAME} PROPERTIES PUBLIC_HEADER "${MLP_HEADER_FILES}")
    target_include_directories(${SHARED_LIB_NAME} PRIVATE ".")
    target_include_directories(${SHARED_LIB_NAME} PRIVATE "src")
    target_include_directories(${SHARED_LIB_NAME} PRIVATE "deps")

    #### RUN TESTS

    ###### FuncLearnTest
    set(CMAKE_DEBUG_POSTFIX _d)
    add_executable(${FUNCLEARN_EXEC_NAME} ${MLP_SOURCE_FILES} ${FUNCLEARN_EXEC_MAIN_PATH})
    target_compile_definitions(${FUNCLEARN_EXEC_NAME} PUBLIC
            ${FUNCLEARN_DEFINES})
    target_compile_options(${FUNCLEARN_EXEC_NAME} PUBLIC
            "-Wall")
    set_target_properties(${FUNCLEARN_EXEC_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    target_include_directories(${FUNCLEARN_EXEC_NAME} PUBLIC ".")
    target_include_directories(${FUNCLEARN_EXEC_NAME} PUBLIC "src")
    target_include_directories(${FUNCLEARN_EXEC_NAME} PUBLIC "deps")
    target_include_directories(${FUNCLEARN_EXEC_NAME} PUBLIC "tests")

    ###### LayerTest
    set(CMAKE_DEBUG_POSTFIX _d)
    add_executable(${LAYER_EXEC_NAME} ${MLP_SOURCE_FILES} ${LAYER_EXEC_MAIN_PATH})
    target_compile_definitions(${LAYER_EXEC_NAME} PUBLIC
            ${LAYER_DEFINES})
    target_compile_options(${LAYER_EXEC_NAME} PUBLIC
            "-Wall")
    set_target_properties(${LAYER_EXEC_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    target_include_directories(${LAYER_EXEC_NAME} PUBLIC ".")
    target_include_directories(${LAYER_EXEC_NAME} PUBLIC "src")
    target_include_directories(${LAYER_EXEC_NAME} PUBLIC "deps")
    target_include_directories(${LAYER_EXEC_NAME} PUBLIC "tests")

    ###### LossTest
    set(CMAKE_DEBUG_POSTFIX _d)
    add_executable(${LOSS_EXEC_NAME} ${MLP_SOURCE_FILES} ${LOSS_EXEC_MAIN_PATH})
    target_compile_definitions(${LOSS_EXEC_NAME} PUBLIC
            ${LOSS_DEFINES})
    target_compile_options(${LOSS_EXEC_NAME} PUBLIC
            "-Wall")
    set_target_properties(${LOSS_EXEC_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    target_include_directories(${LOSS_EXEC_NAME} PUBLIC ".")
    target_include_directories(${LOSS_EXEC_NAME} PUBLIC "src")
    target_include_directories(${LOSS_EXEC_NAME} PUBLIC "deps")
    target_include_directories(${LOSS_EXEC_NAME} PUBLIC "tests")

    ###### MLPTest
    set(CMAKE_DEBUG_POSTFIX _d)
    add_executable(${MLP_EXEC_NAME} ${MLP_SOURCE_FILES} ${MLP_EXEC_MAIN_PATH})
    target_compile_definitions(${MLP_EXEC_NAME} PUBLIC
            ${MLP_DEFINES})
    target_compile_options(${MLP_EXEC_NAME} PUBLIC
            "-Wall")
    set_target_properties(${MLP_EXEC_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    target_include_directories(${MLP_EXEC_NAME} PUBLIC ".")
    target_include_directories(${MLP_EXEC_NAME} PUBLIC "src")
    target_include_directories(${MLP_EXEC_NAME} PUBLIC "deps")
    target_include_directories(${MLP_EXEC_NAME} PUBLIC "tests")

    ###### UnitTest MAIN
    set(CMAKE_DEBUG_POSTFIX _d)
    add_executable(${UNITTESTS_EXEC_NAME} ${MLP_SOURCE_FILES} ${UNITTESTS_EXEC_MAIN_PATH})
    target_compile_definitions(${UNITTESTS_EXEC_NAME} PUBLIC
            ${UNITTESTS_DEFINES})
    target_compile_options(${UNITTESTS_EXEC_NAME} PUBLIC
            "-Wall")
    set_target_properties(${UNITTESTS_EXEC_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    target_include_directories(${UNITTESTS_EXEC_NAME} PUBLIC ".")
    target_include_directories(${UNITTESTS_EXEC_NAME} PUBLIC "src")
    target_include_directories(${UNITTESTS_EXEC_NAME} PUBLIC "deps")
    target_include_directories(${UNITTESTS_EXEC_NAME} PUBLIC "tests")

    if(UNIX) ## Install shared lib
        include(GNUInstallDirs)
        install(TARGETS ${SHARED_LIB_NAME}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif(UNIX)

endif ()  # ${CMAKE_SYSTEM_NAME} STREQUAL XCORE_XS3A