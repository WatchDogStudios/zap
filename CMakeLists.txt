cmake_minimum_required(VERSION 3.21)
project(zap VERSION 1.1.0 LANGUAGES C)

# header-only: target_link_libraries(your_game PRIVATE zap::zap)
add_library(zap INTERFACE)
add_library(zap::zap ALIAS zap)
target_include_directories(zap INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_C_STANDARD_REQUIRED ON)
  find_package(Threads REQUIRED)

  add_executable(zap_cli zap.c)
  set_target_properties(zap_cli PROPERTIES OUTPUT_NAME zap)
  add_executable(zap_bench bench.c)
  add_executable(zap_tex_bench tex_bench.c)
  add_executable(zap_video_bench video_bench.c)

  foreach(t zap_cli zap_bench zap_tex_bench zap_video_bench)
    target_link_libraries(${t} PRIVATE zap Threads::Threads)
    if(MSVC)
      target_compile_definitions(${t} PRIVATE _CRT_SECURE_NO_WARNINGS)
      target_compile_options(${t} PRIVATE /W4)
    else()
      target_compile_options(${t} PRIVATE -Wall -Wextra -pedantic)
      target_link_libraries(${t} PRIVATE m)
    endif()
  endforeach()

  # D3D11 + Dear ImGui viewer (Windows): texture and video split-view comparisons
  if(WIN32)
    option(ZAP_BUILD_VIEWER "Build the D3D11 + Dear ImGui viewer sample (downloads Dear ImGui, LZ4 and zstd)" ON)
    if(ZAP_BUILD_VIEWER)
      enable_language(CXX)
      include(FetchContent)
      FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.92.7 GIT_SHALLOW TRUE)
      FetchContent_MakeAvailable(imgui)
      add_executable(zap_viewer samples/dx11/viewer.cpp
        ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp)
      target_include_directories(zap_viewer PRIVATE ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
      target_compile_features(zap_viewer PRIVATE cxx_std_17)
      target_link_libraries(zap_viewer PRIVATE zap d3d11 dxgi d3dcompiler windowscodecs mfplat mfreadwrite mfuuid shell32 ole32)

      # Package tab compares against LZ4 and zstd (compiled into the viewer only; the zap headers stay dependency-free)
      FetchContent_Declare(lz4 URL https://github.com/lz4/lz4/releases/download/v1.10.0/lz4-1.10.0.tar.gz
        URL_HASH SHA256=537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE SOURCE_SUBDIR none)
      FetchContent_Declare(zstd URL https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
        URL_HASH SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE SOURCE_SUBDIR none)
      FetchContent_MakeAvailable(lz4 zstd)
      file(GLOB zap_zstd_src ${zstd_SOURCE_DIR}/lib/common/*.c ${zstd_SOURCE_DIR}/lib/compress/*.c ${zstd_SOURCE_DIR}/lib/decompress/*.c)
      add_library(zap_viewer_codecs STATIC ${lz4_SOURCE_DIR}/lib/lz4.c ${lz4_SOURCE_DIR}/lib/lz4hc.c ${zap_zstd_src})
      target_include_directories(zap_viewer_codecs PUBLIC ${lz4_SOURCE_DIR}/lib ${zstd_SOURCE_DIR}/lib)
      target_compile_definitions(zap_viewer_codecs PRIVATE ZSTD_DISABLE_ASM)
      target_link_libraries(zap_viewer PRIVATE zap_viewer_codecs)
    endif()
  endif()

  enable_testing()
  add_test(NAME selftest COMMAND zap_bench)
  add_test(NAME tex_selftest COMMAND zap_tex_bench)
  add_test(NAME video_selftest COMMAND zap_video_bench)
endif()
