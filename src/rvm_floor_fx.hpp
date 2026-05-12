#pragma once

/** Azure Kinect + RVM 바닥/그림자/반사 합성 API (구현은 rvm_floor_fx.cpp). */

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <array>
#include <memory>
#include <string>

#include <k4a/k4a.hpp>

#ifdef RVM_USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace rvm_floor {

/** UI/INI에서 넘기는 바닥·태양·그림자·평면 추정 파라미터 묶음 */
struct Params
{
    bool enabled = true;
    /** 바닥 위 소프트 접촉 그림자(가우시안) 강도 */
    float shadow_strength = 0.38f;
    float reflection_strength = 0.38f;
    float downsample_ratio = 0.35f;
    std::string onnx_path;

    bool floor_grid_show = true;
    int floor_grid_spacing = 28;

    /** 태양 고도(도): 90에 가까울수록 그림자 짧음 */
    float sun_altitude_deg = 48.f;
    /** 태양 방위(도): 그림자가 반대 방향으로 길어짐 (이미지 X=우, Y=아래) */
    float sun_azimuth_deg = 230.f;

    bool floor_anchor_valid = false;
    float floor_anchor_u = 0.5f;
    float floor_anchor_v = 0.82f;

    /** 3D 캐스트 그림자 합성 강도(0..1) */
    float cast_shadow_alpha = 0.72f;
    /** 캐스트 그림자 맵 가우시안 시그마(픽셀 근사) */
    float cast_shadow_blur = 10.f;
    /** 3D 투영 그림자용 카메라 intrinsics (pixel) */
    bool cam_intrin_valid = false;
    float cam_fx = 0.f, cam_fy = 0.f, cam_cx = 0.f, cam_cy = 0.f;
    /** 3D 투영 그림자 샘플링 step (픽셀) */
    int shadow_ray_step = 2;

    bool reflection_on = true;
    /** 수직 반사 기준선 (0..1, 이미지 높이 비율) */
    float reflection_plane_y = 0.68f;

    /** 평면 추정을 발 근처 ROI로 제한(반사 안정화) */
    bool plane_use_foot_roi = true;
    /** 발 ROI 반경 (화면폭 비율) */
    float plane_foot_roi_radius = 0.24f;
    /** 평면 추정 시간 평활(EMA) */
    bool plane_temporal_smooth_on = true;
    /** 현재 프레임 가중치(0..1): 낮을수록 더 안정적 */
    float plane_temporal_alpha = 0.22f;
};

/** RVM ONNX / ONNX Runtime 상태 보관 */
struct Engine
{
    cv::dnn::Net net;
    bool net_loaded = false;
    bool ort_loaded = false;
    std::string backend_name;
    std::string last_error;
    std::vector<cv::Mat> rec{4};
    cv::Size rvm_input{256, 256};
    int rvm_dtype = CV_32F;
#ifdef RVM_USE_ONNXRUNTIME
    std::unique_ptr<Ort::Env> ort_env;
    std::unique_ptr<Ort::Session> ort_session;
    std::array<std::vector<float>, 4> ort_rec;
    std::array<std::vector<int64_t>, 4> ort_rec_shape;
    std::array<std::string, 6> ort_input_names;
    std::array<std::string, 6> ort_output_names;
#endif
};

bool try_load_onnx(Engine& e, const std::string& path);

/** Recurrent RVM step: alpha resized to `color_bgr.size()`. */
bool infer_alpha(Engine& e, const cv::Mat& color_bgr, cv::Mat& alpha_f32_hw, float downsample_ratio);

/** Depth-assisted matte when RVM is unavailable or fails (0..1 float). */
void infer_alpha_depth_fallback(const cv::Mat& color_bgr,
                                const cv::Mat& depth_u16,
                                cv::Mat& alpha_f32_hw);

/**
 * color_bgr, depth_u16: same size (depth-aligned color recommended).
 * alpha_f32_hw: CV_32F, same size, 0..1 person.
 */
cv::Mat compose_floor_with_fx(const cv::Mat& color_bgr,
                                const cv::Mat& depth_u16,
                                const cv::Mat& alpha_f32_hw,
                                const Params& p);

/**
 * 바닥 마스크 + 알파/깊이 기반 3D 캐스트 그림자 가중치 (CV_32F, 0..1, color와 동일 크기).
 * intrinsics/평면 실패 시 빈 Mat — 호출부에서 2D 워프 폴백 가능.
 */
cv::Mat compute_cast_shadow_weight(const cv::Mat& color_bgr,
                                    const cv::Mat& depth_u16,
                                    const cv::Mat& alpha_f32_hw,
                                    const Params& p);

/**
 * 바닥 마스크 위에 사람 매트+깊이로 추정한 바닥 평면 거울상(태양 무관)을 블렌딩.
 * io_bgr는 in-place 수정. strength 0..1 권장(예: floor_refl).
 */
void apply_depth_mirror_person_reflection(cv::Mat& io_bgr,
                                          const cv::Mat& color_bgr,
                                          const cv::Mat& depth_u16,
                                          const cv::Mat& alpha_f32_hw,
                                          const Params& p,
                                          float strength);

/**
 * compose_floor_with_fx에 포함된 것과 동일한 2D 바닥 반사(reflection_plane_y 수평 거울 + 알파 가중).
 * Raw color MR 등 compose 결과를 쓰지 않는 경로에서 FLOOR 합성과 동일한 반사를 입힐 때 사용.
 */
void apply_compose_style_floor_reflection(cv::Mat& io_bgr,
                                          const cv::Mat& color_bgr,
                                          const cv::Mat& depth_u16,
                                          const cv::Mat& alpha_f32_hw,
                                          const Params& p);

/** k4a images -> aligned BGR + depth CV_16UC1; returns false if alignment fails. */
bool color_depth_aligned_mats(const k4a::transformation& tr,
                                const k4a::image& depth,
                                const k4a::image& color,
                                cv::Mat& out_bgr,
                                cv::Mat& out_depth_u16);

} // namespace rvm_floor
