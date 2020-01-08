# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
set(CMAKE_BUILD_TYPE RELEASE)

#include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi)

include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/include/ncnn)

set(CMAKE_STATIC_LINKER_FLAGS "-lm  -pthread -fopenmp -lstdc++")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -Wno-unused-result  -Wfatal-errors -fPIC -fno-rtti -fno-exceptions")

add_library (libncnn STATIC IMPORTED)
set_target_properties(libncnn PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/src/main/jniLibs/armeabi-v7a/libncnn.a)

set(lib_src ${CMAKE_SOURCE_DIR}/src/main/cpp/yolov3-tiny-jni.cpp)

add_library( # Sets the name of the library.
        yolov3_tiny_jni

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${lib_src})

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log
        android)
#find_library( # Sets the name of the path variable.
        #JniGraphics

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        #jnigraphics)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        yolov3_tiny_jni
        libncnn
        android
        jnigraphics
        #vulkan
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})
