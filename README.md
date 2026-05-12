# k4a_imviewer

Azure Kinect DK용 뷰어: 컬러·깊이·(선택) RVM 바닥/그림자/반사 합성.

## 요구 사항

- Windows, [Azure Kinect Sensor SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)
- [OpenCV](https://opencv.org/) (빌드에 `OpenCVConfig.cmake` 필요)
- CMake 3.20+

`CMakeLists.txt`의 캐시 변수:

- `K4A_ROOT` — 기본 `C:/Program Files/Azure Kinect SDK v1.4.2/sdk`
- `OpenCV_DIR` — OpenCV 빌드 출력의 `OpenCVConfig.cmake`가 있는 디렉터리

## 빌드

```powershell
cmake -S . -B build -DOpenCV_DIR="C:/path/to/opencv/build"
cmake --build build --config Release
```

실행 파일: `build/Release/k4a_imviewer.exe` (구성에 따라 경로는 다를 수 있음)

## RVM (선택)

`models/`에 ONNX를 두고 앱에서 경로를 지정합니다. 기본 `.gitignore`는 대용량 가중치를 제외합니다.

ONNX Runtime 바이너리는 저장소에 넣지 않습니다. CMake 구성 시 `RVM_USE_ONNXRUNTIME=ON`(기본)이면 `third_party/` 아래로 자동 다운로드됩니다(인터넷 필요).

## 라이선스

Kinect SDK·OpenCV·FetchContent로 가져오는 서드파티는 각 라이선스를 따릅니다. 이 저장소의 앱 소스 라이선스는 별도로 명시하지 않았다면 저장소 소유자에게 문의하세요.
