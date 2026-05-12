cmake_minimum_required(VERSION 3.20)
project(k4a_imviewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
  add_compile_options(/utf-8 /W3)
endif()

set(K4A_ROOT "C:/Program Files/Azure Kinect SDK v1.4.2/sdk" CACHE PATH "Azure Kinect SDK root")
set(OpenCV_DIR "D:/2026/Dev/cursor/opencv/opencv/build" CACHE PATH "OpenCVConfig.cmake directory")

list(APPEND CMAKE_PREFIX_PATH "${OpenCV_DIR}")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

add_library(glad STATIC "${CMAKE_SOURCE_DIR}/third_party/glad/src/gl.c")
target_include_directories(glad PUBLIC "${CMAKE_SOURCE_DIR}/third_party/glad/include")

include(FetchContent)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG docking
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

set(_imgui_src
  "${imgui_SOURCE_DIR}/imgui.cpp"
  "${imgui_SOURCE_DIR}/imgui_draw.cpp"
  "${imgui_SOURCE_DIR}/imgui_tables.cpp"
  "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
  "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
  "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
)
add_library(imgui STATIC ${_imgui_src})
target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends")
target_link_libraries(imgui PUBLIC glfw glad)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_CUSTOM)

if(MSVC)
  set(_gl_force "${CMAKE_SOURCE_DIR}/include/gl_force_include.h")
  set_source_files_properties(
    "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
    PROPERTIES COMPILE_OPTIONS "/FI${_gl_force}"
  )
endif()

add_executable(k4a_imviewer src/k4a_imviewer.cpp src/rvm_floor_fx.cpp src/viewer_settings_io.cpp)
target_include_directories(k4a_imviewer PRIVATE "${K4A_ROOT}/include" "${glm_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/src")

option(RVM_USE_ONNXRUNTIME "Use ONNX Runtime backend for RVM" ON)
set(ONNXRUNTIME_VERSION "1.20.1" CACHE STRING "ONNX Runtime version")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime (contains include/ and lib/)")
if(RVM_USE_ONNXRUNTIME)
  if(NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
  endif()
  if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    set(_ort_zip "${CMAKE_BINARY_DIR}/_deps/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
    set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
    message(STATUS "Downloading ONNX Runtime from ${_ort_url}")
    file(DOWNLOAD "${_ort_url}" "${_ort_zip}" SHOW_PROGRESS STATUS _ort_dl_status TLS_VERIFY ON)
    list(GET _ort_dl_status 0 _ort_dl_code)
    if(NOT _ort_dl_code EQUAL 0)
      message(WARNING "ONNX Runtime download failed: ${_ort_dl_status}. Falling back to OpenCV DNN only.")
      set(RVM_USE_ONNXRUNTIME OFF)
    else()
      file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party")
      execute_process(
        COMMAND "${CMAKE_COMMAND}" -E tar xzf "${_ort_zip}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party"
        RESULT_VARIABLE _ort_extract_result
      )
      if(NOT _ort_extract_result EQUAL 0)
        message(WARNING "ONNX Runtime extract failed (${_ort_extract_result}). Falling back to OpenCV DNN only.")
        set(RVM_USE_ONNXRUNTIME OFF)
      endif()
    endif()
  endif()

  if(RVM_USE_ONNXRUNTIME)
    if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h" AND
       EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib" AND
       EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
      target_compile_definitions(k4a_imviewer PRIVATE RVM_USE_ONNXRUNTIME=1)
      target_include_directories(k4a_imviewer PRIVATE "${ONNXRUNTIME_ROOT}/include")
      target_link_libraries(k4a_imviewer PRIVATE "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
      add_custom_command(
        TARGET k4a_imviewer POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
                "$<TARGET_FILE_DIR:k4a_imviewer>/onnxruntime.dll"
      )
      message(STATUS "RVM ONNX Runtime enabled: ${ONNXRUNTIME_ROOT}")
    else()
      message(WARNING "ONNX Runtime files not found in ${ONNXRUNTIME_ROOT}. Falling back to OpenCV DNN only.")
    endif()
  endif()
endif()

target_link_libraries(
  k4a_imviewer
  PRIVATE
    glad
    imgui
    glfw
    ${OpenCV_LIBS}
    "${K4A_ROOT}/windows-desktop/amd64/release/lib/k4a.lib"
)
if(WIN32)
  target_link_libraries(k4a_imviewer PRIVATE opengl32)
endif()
