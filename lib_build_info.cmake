#Name
set(LIB_NAME lib_mlp)
set(LIB_VERSION 0.0.1)
#set(LIB_ARCHIVE_ARCHS xs3a)
## Compiler/linker options
#set(LIB_ARCHIVE_INCLUDES api)
#set(LIB_ARCHIVE_CXX_SRCS
#        "src/MLP.cpp")

#set(LIB_ARCHIVE_COMPILER_FLAGS -O0 -g -std=c++14)
#set(LIB_ARCHIVE_DEPENDENT_MODULES "")
# Impose C++14 for files that need it

set(LIB_INCLUDES api)
set(LIB_CXX_SRCS "src/MLP.cpp" "src/Dataset.cpp")
set(LIB_COMPILER_FLAGS -O3 -std=c++14)

#XMOS_REGISTER_STATIC_LIB()
XMOS_REGISTER_MODULE()
