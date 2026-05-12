// Azure Kinect — ImGui viewer (layout inspired by Microsoft k4aviewer).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <k4a/k4a.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "rvm_floor_fx.hpp"
#include "viewer_settings_io.hpp"

namespace {

constexpr int kView2D = 0;
constexpr int kView3D = 1;

struct UiConfig
{
    bool depth_on = true;
    bool color_on = true;
    bool imu_on = false;
    bool mic_ui = true;
    bool disable_led = false;
    int depth_mode = 1;
    int color_format = 0;
    int color_res = 2;
    int fps = 2;
    int view_mode = kView2D;
    int cloud_style = 0;
    float cloud_point_px = 2.0f;
    float orbit_dist = 2.2f;
    bool floor_fx_enabled = true;
    float floor_shadow = 0.55f;
    float floor_refl = 0.38f;
    float floor_ds_ratio = 0.35f;
    bool show_ir = true;
    bool show_depth = true;
    bool show_color = true;
    bool show_imu_panel = true;
    bool show_floor_panel = true;
    int floor_update_stride = 2;
    bool person_segment_depth_only = true; // fast mode: disable RVM inference
    bool floor_grid_show = true;
    int floor_grid_spacing = 28;
    // Shadow defaults (safe-visible)
    float sun_altitude_deg = 34.f;
    float sun_azimuth_deg = 205.f;
    float floor_anchor_u = 0.5f;
    float floor_anchor_v = 0.82f;
    bool floor_anchor_valid = false;
    float cast_shadow_alpha = 0.86f;
    float cast_shadow_blur = 14.f;
    bool reflection_on = true;
    float reflection_plane_y = 0.68f;
    bool reflection_single_pass_only = true;
    bool reflection_undistort_input = true;
    bool plane_use_foot_roi = true;
    float plane_foot_roi_radius = 0.24f;
    bool plane_temporal_smooth_on = true;
    float plane_temporal_alpha = 0.22f;
    int rvm_bg_mode = 2; // 0:black, 1:color, 2:shadow/reflection fx
    bool rvm_floor_overlay_on = true;
    bool rvm_shadow_overlay_on = true;
    bool rvm_reflection_overlay_on = false;
    int color_view_mode = 0; // 0:raw, 1:human matte, 2:human matte + depth shadow
    int color_depth_view_mode = 0; // 0:off, 1:overlay, 2:depth only
    float color_depth_overlay_alpha = 0.45f;
    bool rgb_depth_hole_fill = true;
    int rgb_depth_fill_iters = 2;
    int rgb_depth_fill_kernel = 5;
    bool mr_menu_enabled = true;
    int mr_stage = 4; // 1..8, default = occlusion stage
    bool mr_occlusion_enabled = true;
    bool mr_occlusion_debug = true;
    bool mr_shadow_enabled = false;
    bool mr_reflection_enabled = false;
    float mr_depth_edge_smooth = 1.25f;
    float mr_target_fps = 30.0f;
    float mr_virtual_depth_mm = 2200.0f;
    float mr_occlusion_overlay_alpha = 0.55f;
    /** true면 Occlusion 디버그를 가상 오브젝트 주변 ROI에만 표시(전체 화면 붉은 막 방지) */
    bool mr_occlusion_debug_local_only = true;
    float mr_occlusion_debug_roi_mul = 3.0f;
    bool mr_virtual_object_enabled = true;
    float mr_object_u = 0.50f;
    float mr_object_v = 0.62f;
    float mr_object_radius_px = 90.0f;
    float mr_object_depth_mm = 2300.0f;
    int mr_object_shape = 0; // 0:disc, 1:box
    float mr_object_shadow_alpha = 0.42f;
    float mr_object_reflection_alpha = 0.26f;
    bool mr_show_pick_xyz = true;
    bool mr_use_world_transform = false;
    float mr_obj_tx_m = 0.0f;
    float mr_obj_ty_m = 0.0f;
    float mr_obj_tz_m = 2.3f;
    float mr_obj_scale = 1.0f;
    float mr_obj_rot_deg = 0.0f;
    bool mr_ground_snap_on_click = true;
    float mr_ground_snap_offset_m = 0.0f;
};

const char* kDepthLabels[] = { "NFOV Binned", "NFOV Unbinned", "WFOV Binned", "WFOV Unbinned", "Passive IR" };
const k4a_depth_mode_t kDepthEnums[] = { K4A_DEPTH_MODE_NFOV_2X2BINNED,
                                         K4A_DEPTH_MODE_NFOV_UNBINNED,
                                         K4A_DEPTH_MODE_WFOV_2X2BINNED,
                                         K4A_DEPTH_MODE_WFOV_UNBINNED,
                                         K4A_DEPTH_MODE_PASSIVE_IR };

const char* kColorFmtLabels[] = { "BGRA", "MJPG", "NV12", "YUY2" };
const k4a_image_format_t kColorFmtEnums[] = { K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                              K4A_IMAGE_FORMAT_COLOR_MJPG,
                                              K4A_IMAGE_FORMAT_COLOR_NV12,
                                              K4A_IMAGE_FORMAT_COLOR_YUY2 };

const char* kRes169Labels[] = { "720p", "1080p", "1440p", "2160p" };
const k4a_color_resolution_t kRes169Enums[] = { K4A_COLOR_RESOLUTION_720P,
                                                K4A_COLOR_RESOLUTION_1080P,
                                                K4A_COLOR_RESOLUTION_1440P,
                                                K4A_COLOR_RESOLUTION_2160P };
const char* kRes43Labels[] = { "1536p", "3072p" };
const k4a_color_resolution_t kRes43Enums[] = { K4A_COLOR_RESOLUTION_1536P, K4A_COLOR_RESOLUTION_3072P };

const char* kFpsLabels[] = { "5 FPS", "15 FPS", "30 FPS" };
const k4a_fps_t kFpsEnums[] = { K4A_FRAMES_PER_SECOND_5, K4A_FRAMES_PER_SECOND_15, K4A_FRAMES_PER_SECOND_30 };

k4a_color_resolution_t map_color_resolution(const UiConfig& u)
{
    if (u.color_res < 4)
        return kRes169Enums[u.color_res];
    return kRes43Enums[u.color_res - 4];
}

/**
 * Azure Kinect: NV12/YUY2 + 30 FPS는 720p에서만 허용되는 조합이 많고,
 * 1080p/1440p/2160p/4:3 해상도와 같이 쓰면 start_cameras()가 실패한다.
 * UI 값을 BGRA로 바꿔 재시도 가능하게 한다.
 */
void fix_unsupported_color_combo(UiConfig& u, std::string* out_note)
{
    if (out_note)
        out_note->clear();
    if (!u.color_on)
        return;

    const int fmt = std::clamp(u.color_format, 0, 3);
    const int fps = std::clamp(u.fps, 0, 2);
    const int res = std::clamp(u.color_res, 0, 5);
    const bool thirty_fps = (fps == 2);
    const bool nv12_or_yuy2 = (fmt == 2 || fmt == 3);
    const bool not_720p_169 = (res != 0);
    if (thirty_fps && nv12_or_yuy2 && not_720p_169)
    {
        u.color_format = 0;
        if (out_note)
            *out_note =
                "NV12/YUY2 + 30 FPS는 720p만 지원됩니다. 컬러 포맷을 BGRA로 바꿨습니다. (Microsoft HW spec)";
    }
}

void put_kv(viewer_io::KVMap& m, const char* k, bool v)
{
    m[k] = v ? "1" : "0";
}
void put_kv(viewer_io::KVMap& m, const char* k, int v)
{
    m[k] = std::to_string(v);
}
void put_kv(viewer_io::KVMap& m, const char* k, float v)
{
    std::ostringstream o;
    o << std::setprecision(8) << v;
    m[k] = o.str();
}
void put_kv(viewer_io::KVMap& m, const char* k, const char* v)
{
    m[k] = v;
}

bool get_kv_b(const viewer_io::KVMap& m, const char* k, bool def)
{
    const auto it = m.find(k);
    if (it == m.end())
        return def;
    const std::string& s = it->second;
    return s == "1" || s == "true" || s == "True" || s == "yes";
}
int get_kv_i(const viewer_io::KVMap& m, const char* k, int def)
{
    const auto it = m.find(k);
    if (it == m.end())
        return def;
    try
    {
        return std::stoi(it->second);
    }
    catch (...)
    {
        return def;
    }
}
float get_kv_f(const viewer_io::KVMap& m, const char* k, float def)
{
    const auto it = m.find(k);
    if (it == m.end())
        return def;
    try
    {
        return std::stof(it->second);
    }
    catch (...)
    {
        return def;
    }
}

void ui_to_kv(const UiConfig& u, const char* rvm_onnx, viewer_io::KVMap& m)
{
    put_kv(m, "depth_on", u.depth_on);
    put_kv(m, "color_on", u.color_on);
    put_kv(m, "imu_on", u.imu_on);
    put_kv(m, "mic_ui", u.mic_ui);
    put_kv(m, "disable_led", u.disable_led);
    put_kv(m, "depth_mode", u.depth_mode);
    put_kv(m, "color_format", u.color_format);
    put_kv(m, "color_res", u.color_res);
    put_kv(m, "fps", u.fps);
    put_kv(m, "view_mode", u.view_mode);
    put_kv(m, "cloud_style", u.cloud_style);
    put_kv(m, "cloud_point_px", u.cloud_point_px);
    put_kv(m, "orbit_dist", u.orbit_dist);
    put_kv(m, "floor_fx_enabled", u.floor_fx_enabled);
    put_kv(m, "floor_shadow", u.floor_shadow);
    put_kv(m, "floor_refl", u.floor_refl);
    put_kv(m, "floor_ds_ratio", u.floor_ds_ratio);
    put_kv(m, "show_ir", u.show_ir);
    put_kv(m, "show_depth", u.show_depth);
    put_kv(m, "show_color", u.show_color);
    put_kv(m, "show_imu_panel", u.show_imu_panel);
    put_kv(m, "show_floor_panel", u.show_floor_panel);
    put_kv(m, "floor_update_stride", u.floor_update_stride);
    put_kv(m, "person_segment_depth_only", u.person_segment_depth_only);
    put_kv(m, "floor_grid_show", u.floor_grid_show);
    put_kv(m, "floor_grid_spacing", u.floor_grid_spacing);
    put_kv(m, "sun_altitude_deg", u.sun_altitude_deg);
    put_kv(m, "sun_azimuth_deg", u.sun_azimuth_deg);
    put_kv(m, "floor_anchor_u", u.floor_anchor_u);
    put_kv(m, "floor_anchor_v", u.floor_anchor_v);
    put_kv(m, "floor_anchor_valid", u.floor_anchor_valid);
    put_kv(m, "cast_shadow_alpha", u.cast_shadow_alpha);
    put_kv(m, "cast_shadow_blur", u.cast_shadow_blur);
    put_kv(m, "reflection_on", u.reflection_on);
    put_kv(m, "reflection_plane_y", u.reflection_plane_y);
    put_kv(m, "reflection_single_pass_only", u.reflection_single_pass_only);
    put_kv(m, "reflection_undistort_input", u.reflection_undistort_input);
    put_kv(m, "plane_use_foot_roi", u.plane_use_foot_roi);
    put_kv(m, "plane_foot_roi_radius", u.plane_foot_roi_radius);
    put_kv(m, "plane_temporal_smooth_on", u.plane_temporal_smooth_on);
    put_kv(m, "plane_temporal_alpha", u.plane_temporal_alpha);
    put_kv(m, "rvm_bg_mode", u.rvm_bg_mode);
    put_kv(m, "rvm_floor_overlay_on", u.rvm_floor_overlay_on);
    put_kv(m, "rvm_shadow_overlay_on", u.rvm_shadow_overlay_on);
    put_kv(m, "rvm_reflection_overlay_on", u.rvm_reflection_overlay_on);
    put_kv(m, "color_view_mode", u.color_view_mode);
    put_kv(m, "color_depth_view_mode", u.color_depth_view_mode);
    put_kv(m, "color_depth_overlay_alpha", u.color_depth_overlay_alpha);
    put_kv(m, "rgb_depth_hole_fill", u.rgb_depth_hole_fill);
    put_kv(m, "rgb_depth_fill_iters", u.rgb_depth_fill_iters);
    put_kv(m, "rgb_depth_fill_kernel", u.rgb_depth_fill_kernel);
    put_kv(m, "mr_menu_enabled", u.mr_menu_enabled);
    put_kv(m, "mr_stage", u.mr_stage);
    put_kv(m, "mr_occlusion_enabled", u.mr_occlusion_enabled);
    put_kv(m, "mr_occlusion_debug", u.mr_occlusion_debug);
    put_kv(m, "mr_shadow_enabled", u.mr_shadow_enabled);
    put_kv(m, "mr_reflection_enabled", u.mr_reflection_enabled);
    put_kv(m, "mr_depth_edge_smooth", u.mr_depth_edge_smooth);
    put_kv(m, "mr_target_fps", u.mr_target_fps);
    put_kv(m, "mr_virtual_depth_mm", u.mr_virtual_depth_mm);
    put_kv(m, "mr_occlusion_overlay_alpha", u.mr_occlusion_overlay_alpha);
    put_kv(m, "mr_occlusion_debug_local_only", u.mr_occlusion_debug_local_only);
    put_kv(m, "mr_occlusion_debug_roi_mul", u.mr_occlusion_debug_roi_mul);
    put_kv(m, "mr_virtual_object_enabled", u.mr_virtual_object_enabled);
    put_kv(m, "mr_object_u", u.mr_object_u);
    put_kv(m, "mr_object_v", u.mr_object_v);
    put_kv(m, "mr_object_radius_px", u.mr_object_radius_px);
    put_kv(m, "mr_object_depth_mm", u.mr_object_depth_mm);
    put_kv(m, "mr_object_shape", u.mr_object_shape);
    put_kv(m, "mr_object_shadow_alpha", u.mr_object_shadow_alpha);
    put_kv(m, "mr_object_reflection_alpha", u.mr_object_reflection_alpha);
    put_kv(m, "mr_show_pick_xyz", u.mr_show_pick_xyz);
    put_kv(m, "mr_use_world_transform", u.mr_use_world_transform);
    put_kv(m, "mr_obj_tx_m", u.mr_obj_tx_m);
    put_kv(m, "mr_obj_ty_m", u.mr_obj_ty_m);
    put_kv(m, "mr_obj_tz_m", u.mr_obj_tz_m);
    put_kv(m, "mr_obj_scale", u.mr_obj_scale);
    put_kv(m, "mr_obj_rot_deg", u.mr_obj_rot_deg);
    put_kv(m, "mr_ground_snap_on_click", u.mr_ground_snap_on_click);
    put_kv(m, "mr_ground_snap_offset_m", u.mr_ground_snap_offset_m);
    put_kv(m, "rvm_onnx", rvm_onnx ? rvm_onnx : "");
}

void kv_to_ui(const viewer_io::KVMap& m, UiConfig& u, char* rvm_onnx, size_t rvm_cap)
{
    u.depth_on = get_kv_b(m, "depth_on", u.depth_on);
    u.color_on = get_kv_b(m, "color_on", u.color_on);
    u.imu_on = get_kv_b(m, "imu_on", u.imu_on);
    u.mic_ui = get_kv_b(m, "mic_ui", u.mic_ui);
    u.disable_led = get_kv_b(m, "disable_led", u.disable_led);
    u.depth_mode = std::clamp(get_kv_i(m, "depth_mode", u.depth_mode), 0, 4);
    u.color_format = std::clamp(get_kv_i(m, "color_format", u.color_format), 0, 3);
    u.color_res = std::clamp(get_kv_i(m, "color_res", u.color_res), 0, 5);
    u.fps = std::clamp(get_kv_i(m, "fps", u.fps), 0, 2);
    u.view_mode = std::clamp(get_kv_i(m, "view_mode", u.view_mode), 0, 1);
    u.cloud_style = std::clamp(get_kv_i(m, "cloud_style", u.cloud_style), 0, 2);
    u.cloud_point_px = get_kv_f(m, "cloud_point_px", u.cloud_point_px);
    u.orbit_dist = get_kv_f(m, "orbit_dist", u.orbit_dist);
    u.floor_fx_enabled = get_kv_b(m, "floor_fx_enabled", u.floor_fx_enabled);
    u.floor_shadow = get_kv_f(m, "floor_shadow", u.floor_shadow);
    u.floor_refl = get_kv_f(m, "floor_refl", u.floor_refl);
    u.floor_ds_ratio = get_kv_f(m, "floor_ds_ratio", u.floor_ds_ratio);
    u.show_ir = get_kv_b(m, "show_ir", u.show_ir);
    u.show_depth = get_kv_b(m, "show_depth", u.show_depth);
    u.show_color = get_kv_b(m, "show_color", u.show_color);
    u.show_imu_panel = get_kv_b(m, "show_imu_panel", u.show_imu_panel);
    u.show_floor_panel = get_kv_b(m, "show_floor_panel", u.show_floor_panel);
    u.floor_update_stride = std::clamp(get_kv_i(m, "floor_update_stride", u.floor_update_stride), 1, 8);
    u.person_segment_depth_only = get_kv_b(m, "person_segment_depth_only", u.person_segment_depth_only);
    u.floor_grid_show = get_kv_b(m, "floor_grid_show", u.floor_grid_show);
    u.floor_grid_spacing = std::clamp(get_kv_i(m, "floor_grid_spacing", u.floor_grid_spacing), 4, 200);
    u.sun_altitude_deg = std::clamp(get_kv_f(m, "sun_altitude_deg", u.sun_altitude_deg), 5.f, 89.f);
    u.sun_azimuth_deg = std::clamp(get_kv_f(m, "sun_azimuth_deg", u.sun_azimuth_deg), 0.f, 360.f);
    u.floor_anchor_u = std::clamp(get_kv_f(m, "floor_anchor_u", u.floor_anchor_u), 0.f, 1.f);
    u.floor_anchor_v = std::clamp(get_kv_f(m, "floor_anchor_v", u.floor_anchor_v), 0.f, 1.f);
    u.floor_anchor_valid = get_kv_b(m, "floor_anchor_valid", u.floor_anchor_valid);
    u.cast_shadow_alpha = std::clamp(get_kv_f(m, "cast_shadow_alpha", u.cast_shadow_alpha), 0.f, 1.f);
    u.cast_shadow_blur = std::clamp(get_kv_f(m, "cast_shadow_blur", u.cast_shadow_blur), 0.f, 80.f);
    u.reflection_on = get_kv_b(m, "reflection_on", u.reflection_on);
    u.reflection_plane_y = std::clamp(get_kv_f(m, "reflection_plane_y", u.reflection_plane_y), 0.1f, 0.95f);
    u.reflection_single_pass_only = get_kv_b(m, "reflection_single_pass_only", u.reflection_single_pass_only);
    u.reflection_undistort_input = get_kv_b(m, "reflection_undistort_input", u.reflection_undistort_input);
    u.plane_use_foot_roi = get_kv_b(m, "plane_use_foot_roi", u.plane_use_foot_roi);
    u.plane_foot_roi_radius = std::clamp(get_kv_f(m, "plane_foot_roi_radius", u.plane_foot_roi_radius), 0.08f, 0.50f);
    u.plane_temporal_smooth_on = get_kv_b(m, "plane_temporal_smooth_on", u.plane_temporal_smooth_on);
    u.plane_temporal_alpha = std::clamp(get_kv_f(m, "plane_temporal_alpha", u.plane_temporal_alpha), 0.02f, 1.0f);
    u.rvm_bg_mode = std::clamp(get_kv_i(m, "rvm_bg_mode", u.rvm_bg_mode), 0, 2);
    u.rvm_floor_overlay_on = get_kv_b(m, "rvm_floor_overlay_on", u.rvm_floor_overlay_on);
    u.rvm_shadow_overlay_on = get_kv_b(m, "rvm_shadow_overlay_on", u.rvm_shadow_overlay_on);
    u.rvm_reflection_overlay_on = get_kv_b(m, "rvm_reflection_overlay_on", u.rvm_reflection_overlay_on);
    u.color_view_mode = std::clamp(get_kv_i(m, "color_view_mode", u.color_view_mode), 0, 2);
    u.color_depth_view_mode = std::clamp(get_kv_i(m, "color_depth_view_mode", u.color_depth_view_mode), 0, 2);
    u.color_depth_overlay_alpha = std::clamp(get_kv_f(m, "color_depth_overlay_alpha", u.color_depth_overlay_alpha), 0.f, 1.f);
    u.rgb_depth_hole_fill = get_kv_b(m, "rgb_depth_hole_fill", u.rgb_depth_hole_fill);
    u.rgb_depth_fill_iters = std::clamp(get_kv_i(m, "rgb_depth_fill_iters", u.rgb_depth_fill_iters), 0, 8);
    u.rgb_depth_fill_kernel = std::clamp(get_kv_i(m, "rgb_depth_fill_kernel", u.rgb_depth_fill_kernel), 3, 11);
    u.mr_menu_enabled = get_kv_b(m, "mr_menu_enabled", u.mr_menu_enabled);
    u.mr_stage = std::clamp(get_kv_i(m, "mr_stage", u.mr_stage), 1, 8);
    u.mr_occlusion_enabled = get_kv_b(m, "mr_occlusion_enabled", u.mr_occlusion_enabled);
    u.mr_occlusion_debug = get_kv_b(m, "mr_occlusion_debug", u.mr_occlusion_debug);
    u.mr_shadow_enabled = get_kv_b(m, "mr_shadow_enabled", u.mr_shadow_enabled);
    u.mr_reflection_enabled = get_kv_b(m, "mr_reflection_enabled", u.mr_reflection_enabled);
    u.mr_depth_edge_smooth = std::clamp(get_kv_f(m, "mr_depth_edge_smooth", u.mr_depth_edge_smooth), 0.f, 6.f);
    u.mr_target_fps = std::clamp(get_kv_f(m, "mr_target_fps", u.mr_target_fps), 5.f, 120.f);
    u.mr_virtual_depth_mm = std::clamp(get_kv_f(m, "mr_virtual_depth_mm", u.mr_virtual_depth_mm), 500.f, 8000.f);
    u.mr_occlusion_overlay_alpha =
        std::clamp(get_kv_f(m, "mr_occlusion_overlay_alpha", u.mr_occlusion_overlay_alpha), 0.f, 1.f);
    u.mr_occlusion_debug_local_only = get_kv_b(m, "mr_occlusion_debug_local_only", u.mr_occlusion_debug_local_only);
    u.mr_occlusion_debug_roi_mul =
        std::clamp(get_kv_f(m, "mr_occlusion_debug_roi_mul", u.mr_occlusion_debug_roi_mul), 1.f, 12.f);
    u.mr_virtual_object_enabled = get_kv_b(m, "mr_virtual_object_enabled", u.mr_virtual_object_enabled);
    u.mr_object_u = std::clamp(get_kv_f(m, "mr_object_u", u.mr_object_u), 0.f, 1.f);
    u.mr_object_v = std::clamp(get_kv_f(m, "mr_object_v", u.mr_object_v), 0.f, 1.f);
    u.mr_object_radius_px = std::clamp(get_kv_f(m, "mr_object_radius_px", u.mr_object_radius_px), 8.f, 420.f);
    u.mr_object_depth_mm = std::clamp(get_kv_f(m, "mr_object_depth_mm", u.mr_object_depth_mm), 500.f, 8000.f);
    u.mr_object_shape = std::clamp(get_kv_i(m, "mr_object_shape", u.mr_object_shape), 0, 1);
    u.mr_object_shadow_alpha = std::clamp(get_kv_f(m, "mr_object_shadow_alpha", u.mr_object_shadow_alpha), 0.f, 1.f);
    u.mr_object_reflection_alpha =
        std::clamp(get_kv_f(m, "mr_object_reflection_alpha", u.mr_object_reflection_alpha), 0.f, 1.f);
    u.mr_show_pick_xyz = get_kv_b(m, "mr_show_pick_xyz", u.mr_show_pick_xyz);
    u.mr_use_world_transform = get_kv_b(m, "mr_use_world_transform", u.mr_use_world_transform);
    u.mr_obj_tx_m = std::clamp(get_kv_f(m, "mr_obj_tx_m", u.mr_obj_tx_m), -6.f, 6.f);
    u.mr_obj_ty_m = std::clamp(get_kv_f(m, "mr_obj_ty_m", u.mr_obj_ty_m), -4.f, 4.f);
    u.mr_obj_tz_m = std::clamp(get_kv_f(m, "mr_obj_tz_m", u.mr_obj_tz_m), 0.3f, 10.f);
    u.mr_obj_scale = std::clamp(get_kv_f(m, "mr_obj_scale", u.mr_obj_scale), 0.1f, 4.f);
    u.mr_obj_rot_deg = std::clamp(get_kv_f(m, "mr_obj_rot_deg", u.mr_obj_rot_deg), -180.f, 180.f);
    u.mr_ground_snap_on_click = get_kv_b(m, "mr_ground_snap_on_click", u.mr_ground_snap_on_click);
    u.mr_ground_snap_offset_m = std::clamp(get_kv_f(m, "mr_ground_snap_offset_m", u.mr_ground_snap_offset_m), -1.f, 1.f);
    const auto it = m.find("rvm_onnx");
    if (it != m.end() && rvm_onnx && rvm_cap > 0)
    {
#ifdef _MSC_VER
        strncpy_s(rvm_onnx, rvm_cap, it->second.c_str(), _TRUNCATE);
#else
        std::strncpy(rvm_onnx, it->second.c_str(), rvm_cap - 1);
        rvm_onnx[rvm_cap - 1] = '\0';
#endif
    }
}

std::string canonical_kv_string(const viewer_io::KVMap& m)
{
    std::ostringstream o;
    for (const auto& kv : m)
        o << kv.first << '=' << kv.second << '\n';
    return o.str();
}

void save_app_settings(const UiConfig& u, const char* rvm_onnx)
{
    viewer_io::KVMap m;
    ui_to_kv(u, rvm_onnx, m);
    viewer_io::save_ini(viewer_io::default_settings_path(), m);
}

void load_app_settings(UiConfig& u, char* rvm_onnx, size_t rvm_cap)
{
    viewer_io::KVMap m;
    if (!viewer_io::load_ini(viewer_io::default_settings_path(), m))
        return;
    kv_to_ui(m, u, rvm_onnx, rvm_cap);
}

k4a_device_configuration_t build_k4a_config(const UiConfig& u)
{
    k4a_device_configuration_t c = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    c.depth_mode = u.depth_on ? kDepthEnums[std::clamp(u.depth_mode, 0, 4)] : K4A_DEPTH_MODE_OFF;
    c.color_format = u.color_on ? kColorFmtEnums[std::clamp(u.color_format, 0, 3)] : K4A_IMAGE_FORMAT_COLOR_MJPG;
    c.color_resolution = u.color_on ? map_color_resolution(u) : K4A_COLOR_RESOLUTION_OFF;
    c.camera_fps = kFpsEnums[std::clamp(u.fps, 0, 2)];
    c.synchronized_images_only = u.depth_on && u.color_on;
    c.disable_streaming_indicator = u.disable_led;
    return c;
}

GLuint compile_shader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char buf[1024];
        glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint create_point_program()
{
    const char* vs = R"(#version 330 core
layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_col;
uniform mat4 u_mvp;
uniform float u_psize;
out vec3 v_col;
void main(){
  v_col = in_col;
  gl_Position = u_mvp * vec4(in_pos, 1.0);
  gl_PointSize = u_psize;
})";
    const char* fs = R"(#version 330 core
in vec3 v_col;
out vec4 o;
void main(){ o = vec4(v_col,1.0); }
)";
    GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
    GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs);
    if (!v || !f)
        return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

cv::Mat k4a_bgra_to_mat(const k4a::image& im)
{
    const int w = im.get_width_pixels();
    const int h = im.get_height_pixels();
    cv::Mat m(h, w, CV_8UC4, (void*)im.get_buffer(), im.get_stride_bytes());
    return m.clone();
}

cv::Mat k4a_color_to_bgr8(const k4a::image& im)
{
    const k4a_image_format_t fmt = im.get_format();
    const int w = im.get_width_pixels();
    const int h = im.get_height_pixels();
    const uint8_t* buf = im.get_buffer();
    const int stride = im.get_stride_bytes();

    if (fmt == K4A_IMAGE_FORMAT_COLOR_BGRA32)
    {
        cv::Mat bgra(h, w, CV_8UC4, (void*)buf, stride);
        cv::Mat bgr;
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    if (fmt == K4A_IMAGE_FORMAT_COLOR_MJPG)
    {
        std::vector<uint8_t> jpeg(buf, buf + im.get_size());
        return cv::imdecode(jpeg, cv::IMREAD_COLOR);
    }
    if (fmt == K4A_IMAGE_FORMAT_COLOR_NV12)
    {
        cv::Mat nv(h * 3 / 2, w, CV_8UC1, (void*)buf, stride);
        cv::Mat bgr;
        cv::cvtColor(nv, bgr, cv::COLOR_YUV2BGR_NV12);
        return bgr;
    }
    if (fmt == K4A_IMAGE_FORMAT_COLOR_YUY2)
    {
        cv::Mat yuy(h, w, CV_8UC2, (void*)buf, stride);
        cv::Mat bgr;
        cv::cvtColor(yuy, bgr, cv::COLOR_YUV2BGR_YUY2);
        return bgr;
    }
    return {};
}

cv::Mat depth_to_colormap(const k4a::image& depth)
{
    const int w = depth.get_width_pixels();
    const int h = depth.get_height_pixels();
    cv::Mat d16(h, w, CV_16UC1, (void*)depth.get_buffer(), depth.get_stride_bytes());
    cv::Mat mask = d16 > 0;
    double mn = 0, mx = 0;
    cv::minMaxLoc(d16, &mn, &mx, nullptr, nullptr, mask);
    if (mx <= mn)
        mx = mn + 1;
    cv::Mat f;
    d16.convertTo(f, CV_32F);
    f = (f - static_cast<float>(mn)) * (255.0f / static_cast<float>(mx - mn));
    cv::Mat u8;
    f.convertTo(u8, CV_8U);
    cv::Mat color;
    cv::applyColorMap(u8, color, cv::COLORMAP_TURBO);
    return color;
}

cv::Mat depth_u16_to_colormap(const cv::Mat& d16)
{
    if (d16.empty() || d16.type() != CV_16UC1)
        return {};
    cv::Mat mask = d16 > 0;
    double mn = 0, mx = 0;
    cv::minMaxLoc(d16, &mn, &mx, nullptr, nullptr, mask);
    if (mx <= mn)
        mx = mn + 1;
    cv::Mat f;
    d16.convertTo(f, CV_32F);
    f = (f - static_cast<float>(mn)) * (255.0f / static_cast<float>(mx - mn));
    cv::Mat u8;
    f.convertTo(u8, CV_8U);
    cv::Mat color;
    cv::applyColorMap(u8, color, cv::COLORMAP_TURBO);
    return color;
}

cv::Mat fill_depth_holes_fast(const cv::Mat& d16_in, int iters, int kernel_size)
{
    if (d16_in.empty() || d16_in.type() != CV_16UC1)
        return d16_in;
    cv::Mat d16 = d16_in.clone();
    const int k = std::clamp(kernel_size | 1, 3, 11);
    cv::Mat elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
    for (int i = 0; i < std::clamp(iters, 0, 8); ++i)
    {
        cv::Mat hole = (d16 == 0);
        if (cv::countNonZero(hole) == 0)
            break;
        cv::Mat dil;
        cv::dilate(d16, dil, elem);
        dil.copyTo(d16, hole);
    }
    return d16;
}

bool depth_to_color_aligned_mats(const k4a::transformation& tr,
                                 const k4a::image& depth,
                                 const k4a::image& color,
                                 cv::Mat& out_color_bgr,
                                 cv::Mat& out_depth_u16)
{
    if (!depth || !color)
        return false;
    try
    {
        out_color_bgr = k4a_color_to_bgr8(color);
        if (out_color_bgr.empty())
            return false;
        k4a::image d2c = tr.depth_image_to_color_camera(depth);
        if (!d2c || d2c.get_format() != K4A_IMAGE_FORMAT_DEPTH16)
            return false;
        cv::Mat d16(d2c.get_height_pixels(), d2c.get_width_pixels(), CV_16UC1, (void*)d2c.get_buffer(),
                    d2c.get_stride_bytes());
        d16.copyTo(out_depth_u16);
        if (out_depth_u16.size() != out_color_bgr.size())
            cv::resize(out_depth_u16, out_depth_u16, out_color_bgr.size(), 0, 0, cv::INTER_NEAREST);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

cv::Mat ir16_to_vis(const k4a::image& ir)
{
    const int w = ir.get_width_pixels();
    const int h = ir.get_height_pixels();
    cv::Mat i16(h, w, CV_16UC1, (void*)ir.get_buffer(), ir.get_stride_bytes());
    cv::Mat mask = i16 > 0;
    double mn = 0, mx = 0;
    cv::minMaxLoc(i16, &mn, &mx, nullptr, nullptr, mask);
    if (mx <= mn)
        mx = mn + 1;
    cv::Mat f;
    i16.convertTo(f, CV_32F);
    f = (f - static_cast<float>(mn)) * (255.0f / static_cast<float>(mx - mn));
    cv::Mat u8;
    f.convertTo(u8, CV_8U);
    cv::Mat bgr;
    cv::cvtColor(u8, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

void upload_tex_rgba(GLuint tex, const cv::Mat& bgr)
{
    if (bgr.empty())
        return;
    cv::Mat rgba;
    if (bgr.channels() == 1)
        cv::cvtColor(bgr, rgba, cv::COLOR_GRAY2RGBA);
    else if (bgr.channels() == 3)
        cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    else if (bgr.channels() == 4)
        cv::cvtColor(bgr, rgba, cv::COLOR_BGRA2RGBA);
    else
        return;

    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (rgba.cols > 0 && rgba.rows > 0)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rgba.cols, rgba.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

struct GlTex
{
    GLuint id = 0;
    void ensure()
    {
        if (!id)
            glGenTextures(1, &id);
    }
    void destroy()
    {
        if (id)
        {
            glDeleteTextures(1, &id);
            id = 0;
        }
    }
};

struct PointCloudGl
{
    GLuint vao = 0, vbo = 0;
    GLsizei count = 0;
    void destroy()
    {
        if (vbo)
        {
            glDeleteBuffers(1, &vbo);
            vbo = 0;
        }
        if (vao)
        {
            glDeleteVertexArrays(1, &vao);
            vao = 0;
        }
        count = 0;
    }

    static uint16_t depth_mm_at(const k4a::image& depth, int x, int y)
    {
        const uint8_t* row = depth.get_buffer() + y * depth.get_stride_bytes();
        return *reinterpret_cast<const uint16_t*>(row + x * sizeof(uint16_t));
    }

    /** style: 0 Simple (flat gray by distance), 1 Shaded (depth normals + lambert), 2 Color (RGB in depth frame) */
    void update(const k4a::transformation& tr,
                const k4a::image& depth,
                const k4a::image* color_img,
                int style,
                int stride_step)
    {
        if (!depth || depth.get_format() != K4A_IMAGE_FORMAT_DEPTH16)
            return;
        const int w = depth.get_width_pixels();
        const int h = depth.get_height_pixels();

        k4a::image xyz = tr.depth_image_to_point_cloud(depth, K4A_CALIBRATION_TYPE_DEPTH);
        const int16_t* p = reinterpret_cast<const int16_t*>(xyz.get_buffer());

        std::optional<k4a::image> color_in_depth;
        if (style == 2 && color_img && *color_img)
        {
            try
            {
                color_in_depth = tr.color_image_to_depth_camera(depth, *color_img);
            }
            catch (const k4a::error&)
            {
                color_in_depth.reset();
            }
        }

        auto bgra_at = [&](int x, int y, float& rr, float& gg, float& bb) -> bool {
            if (!color_in_depth.has_value() || !color_in_depth.value())
                return false;
            const k4a::image& im = color_in_depth.value();
            if (im.get_format() != K4A_IMAGE_FORMAT_COLOR_BGRA32)
                return false;
            if (x < 0 || y < 0 || x >= im.get_width_pixels() || y >= im.get_height_pixels())
                return false;
            const uint8_t* row = im.get_buffer() + y * im.get_stride_bytes();
            const uint8_t* px = row + x * 4;
            bb = px[0] / 255.0f;
            gg = px[1] / 255.0f;
            rr = px[2] / 255.0f;
            return true;
        };

        std::vector<float> inter;
        inter.reserve(static_cast<size_t>((w / stride_step) * (h / stride_step) * 6));

        const int sx = stride_step;
        const int sy = stride_step;
        for (int y = 0; y < h; y += sy)
        {
            for (int x = 0; x < w; x += sx)
            {
                const uint16_t d0 = depth_mm_at(depth, x, y);
                if (d0 == 0)
                    continue;

                const int i = (y * w + x) * 3;
                const float xf = p[i] * 0.001f;
                const float yf = p[i + 1] * 0.001f;
                const float zf = p[i + 2] * 0.001f;
                if (zf <= 0.01f || zf > 8.0f)
                    continue;

                const float t = std::clamp((zf - 0.3f) / 4.0f, 0.0f, 1.0f);
                float r = 0.5f, g = 0.5f, b = 0.55f;

                if (style == 0)
                {
                    const float c = 0.28f + 0.55f * (1.0f - t);
                    r = g = b = c;
                }
                else if (style == 1)
                {
                    const int x1 = std::min(x + sx, w - 1);
                    const int y1 = std::min(y + sy, h - 1);
                    if (depth_mm_at(depth, x1, y) == 0 || depth_mm_at(depth, x, y1) == 0)
                    {
                        r = g = b = 0.35f + 0.45f * (1.0f - t);
                    }
                    else
                    {
                        const int i1 = (y * w + x1) * 3;
                        const int i2 = (y1 * w + x) * 3;
                        const glm::vec3 P0(p[i] * 0.001f, p[i + 1] * 0.001f, p[i + 2] * 0.001f);
                        const glm::vec3 P1(p[i1] * 0.001f, p[i1 + 1] * 0.001f, p[i1 + 2] * 0.001f);
                        const glm::vec3 P2(p[i2] * 0.001f, p[i2 + 1] * 0.001f, p[i2 + 2] * 0.001f);
                        glm::vec3 n = glm::cross(P1 - P0, P2 - P0);
                        const float len = glm::length(n);
                        if (len > 1e-6f)
                            n /= len;
                        else
                            n = glm::vec3(0.0f, 0.0f, 1.0f);
                        const glm::vec3 L = glm::normalize(glm::vec3(0.35f, -0.55f, 0.75f));
                        float shade = 0.18f + 0.82f * std::max(0.0f, glm::dot(n, L));
                        const float base = 0.42f + 0.48f * (1.0f - t);
                        r = g = b = base * shade;
                    }
                }
                else
                {
                    if (!bgra_at(x, y, r, g, b))
                    {
                        r = t;
                        g = std::max(0.0f, 1.0f - std::abs(t - 0.5f) * 2.0f);
                        b = 1.0f - t;
                    }
                }

                inter.push_back(xf);
                inter.push_back(-yf);
                inter.push_back(-zf);
                inter.push_back(r);
                inter.push_back(g);
                inter.push_back(b);
            }
        }
        if (inter.empty())
            return;

        if (!vao)
            glGenVertexArrays(1, &vao);
        if (!vbo)
            glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, inter.size() * sizeof(float), inter.data(), GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glBindVertexArray(0);
        count = static_cast<GLsizei>(inter.size() / 6);
    }
};

void draw_point_cloud(GLuint program,
                      const PointCloudGl& pc,
                      float yaw,
                      float pitch,
                      float orbit_dist,
                      float point_px,
                      int fb_w,
                      int fb_h)
{
    if (!program || pc.count <= 0 || !pc.vao)
        return;
    glEnable(GL_PROGRAM_POINT_SIZE);
    glUseProgram(program);
    const float aspect = (fb_h > 0) ? (static_cast<float>(fb_w) / static_cast<float>(fb_h)) : 1.0f;
    const glm::vec3 eye(orbit_dist * std::cos(yaw) * std::cos(pitch),
                        orbit_dist * std::sin(pitch),
                        orbit_dist * std::sin(yaw) * std::cos(pitch));
    const glm::vec3 center(0.0f, 0.0f, -1.2f);
    const glm::vec3 up(0.0f, -1.0f, 0.0f);
    const glm::mat4 V = glm::lookAt(eye, center, up);
    const glm::mat4 P = glm::perspective(glm::radians(55.0f), aspect, 0.05f, 80.0f);
    const glm::mat4 mvp = P * V;
    const GLint loc_mvp = glGetUniformLocation(program, "u_mvp");
    const GLint loc_ps = glGetUniformLocation(program, "u_psize");
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(loc_ps, point_px);
    glBindVertexArray(pc.vao);
    glDrawArrays(GL_POINTS, 0, pc.count);
    glBindVertexArray(0);
    glUseProgram(0);
}

} // namespace

int main()
{
    if (!glfwInit())
        return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Azure Kinect Viewer", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    {
        ImFontConfig fc;
        fc.OversampleH = 2;
        fc.OversampleV = 2;
        static const char* kKoFonts[] = { "C:/Windows/Fonts/malgun.ttf", "C:\\Windows\\Fonts\\malgun.ttf" };
        for (const char* path : kKoFonts)
        {
            std::ifstream tf(path, std::ios::binary);
            if (!tf.good())
                continue;
            tf.close();
            if (ImFont* f = io.Fonts->AddFontFromFileTTF(path, 17.0f, &fc, io.Fonts->GetGlyphRangesKorean()))
            {
                io.FontDefault = f;
                break;
            }
        }
    }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    std::unique_ptr<k4a::device> dev;
    std::string serial = "(not connected)";
    std::string last_err;
    bool streaming = false;
    bool imu_running = false;
    UiConfig ui{};
    std::optional<k4a::transformation> xform;

    GlTex tex_ir, tex_depth, tex_color, tex_floor;
    rvm_floor::Engine rvm_eng;
    bool rvm_autoload_attempted = false;
    static char rvm_onnx_buf[1024];
    static bool rvm_path_inited = false;
    if (!rvm_path_inited)
    {
        rvm_path_inited = true;
        rvm_onnx_buf[0] = '\0';
        if (const char* ev = std::getenv("K4A_RVM_ONNX"))
        {
#ifdef _MSC_VER
            strncpy_s(rvm_onnx_buf, sizeof(rvm_onnx_buf), ev, _TRUNCATE);
#else
            std::strncpy(rvm_onnx_buf, ev, sizeof(rvm_onnx_buf) - 1);
            rvm_onnx_buf[sizeof(rvm_onnx_buf) - 1] = '\0';
#endif
        }
        else
        {
            static const char* kOnnxCandidates[] = {
                "D:/2026/Dev/cursor/models/rvm_mobilenetv3_fp32.onnx",
                "../../models/rvm_mobilenetv3_fp32.onnx",
                "models/rvm_mobilenetv3_fp32.onnx",
            };
            for (const char* cand : kOnnxCandidates)
            {
                std::ifstream test(cand, std::ios::binary);
                if (test.good())
                {
#ifdef _MSC_VER
                    strncpy_s(rvm_onnx_buf, sizeof(rvm_onnx_buf), cand, _TRUNCATE);
#else
                    std::strncpy(rvm_onnx_buf, cand, sizeof(rvm_onnx_buf) - 1);
                    rvm_onnx_buf[sizeof(rvm_onnx_buf) - 1] = '\0';
#endif
                    break;
                }
            }
        }
    }
    load_app_settings(ui, rvm_onnx_buf, IM_ARRAYSIZE(rvm_onnx_buf));
    // If settings.ini didn't exist (or older presets set near-zero), apply safe-visible shadow defaults.
    if (ui.cast_shadow_alpha < 0.05f)
        ui.cast_shadow_alpha = 0.96f;
    if (ui.cast_shadow_blur < 0.5f)
        ui.cast_shadow_blur = 6.0f;
    if (ui.sun_altitude_deg < 8.f)
        ui.sun_altitude_deg = 28.f;
    // azimuth can be 0..360; only fix obviously invalid
    if (!(ui.sun_azimuth_deg >= 0.f && ui.sun_azimuth_deg <= 360.f))
        ui.sun_azimuth_deg = 252.f;
    static char preset_path[1024] = "floor_preset.ini";

    cv::Mat m_floor;
    cv::Mat m_color_fx;
    cv::Mat m_color_mr_rgb;
    cv::Mat m_floor_alpha;
    cv::Mat m_rgb_alpha;
    /** RVM 실패 시 reflection 전용 깊이 기반 알파(주기적 갱신). */
    cv::Mat m_rgb_alpha_refl_fb;
    cv::Mat m_depth_aligned_u16;
    cv::Mat m_depth_rgb_u16;
    cv::Mat m_depth_rgb_u16_filled;
    cv::Mat m_depth_rgb_vis;
    cv::Mat m_ir, m_depth, m_color;
    k4a::image k4a_dep, k4a_col;
    std::optional<k4a::capture> cap_last;
    GLuint fbo = 0, fbo_tex = 0, rbo = 0;
    int fbo_w = 0, fbo_h = 0;
    PointCloudGl pcgl;
    GLuint pc_program = create_point_program();
    float yaw = 0.35f, pitch = 0.25f;
    bool dragging_pc = false;
    double drag_sx = 0, drag_sy = 0;
    int frame_counter = 0;
    static int floor_stride_counter = 0;
    static int rgb_stride_counter = 0;
    std::string matte_mode_line = "Matte: unknown";
    bool have_depth_intrin = false;
    float depth_fx = 0.0f, depth_fy = 0.0f, depth_cx = 0.0f, depth_cy = 0.0f;
    bool have_color_intrin = false;
    float color_fx = 0.0f, color_fy = 0.0f, color_cx = 0.0f, color_cy = 0.0f;
    bool have_color_distortion = false;
    cv::Mat color_dist_coeff;
    cv::Mat undistort_map1;
    cv::Mat undistort_map2;
    cv::Size undistort_map_size(0, 0);
    bool mr_pick_valid = false;
    float mr_pick_x_m = 0.0f, mr_pick_y_m = 0.0f, mr_pick_z_m = 0.0f;
    int mr_pick_u = 0, mr_pick_v = 0;
    auto t_prev = std::chrono::steady_clock::now();
    float mr_fps_smoothed = 0.0f;
    float mr_latency_ms_smoothed = 0.0f;

    auto try_open_device = [&]() {
        last_err.clear();
        try
        {
            const uint32_t n = k4a::device::get_installed_count();
            if (n == 0)
            {
                serial = "(no device)";
                dev.reset();
                return;
            }
            dev = std::make_unique<k4a::device>(k4a::device::open(0));
            serial = dev->get_serialnum();
        }
        catch (const k4a::error& e)
        {
            last_err = e.what();
            dev.reset();
            serial = "(open failed)";
        }
    };

    try_open_device();

    while (!glfwWindowShouldClose(window))
    {
        const auto t_now = std::chrono::steady_clock::now();
        const float dt_ms = std::chrono::duration<float, std::milli>(t_now - t_prev).count();
        t_prev = t_now;
        if (dt_ms > 0.05f)
        {
            const float fps_now = 1000.0f / dt_ms;
            if (mr_fps_smoothed <= 1e-3f)
                mr_fps_smoothed = fps_now;
            else
                mr_fps_smoothed = mr_fps_smoothed * 0.90f + fps_now * 0.10f;
            if (mr_latency_ms_smoothed <= 1e-3f)
                mr_latency_ms_smoothed = dt_ms;
            else
                mr_latency_ms_smoothed = mr_latency_ms_smoothed * 0.90f + dt_ms * 0.10f;
        }
        glfwPollEvents();
        const int fb_w = static_cast<int>(io.DisplaySize.x);
        const int fb_h = static_cast<int>(io.DisplaySize.y);
        glViewport(0, 0, fb_w, fb_h);
        glClearColor(0.08f, 0.08f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::SetNextWindowViewport(vp->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGuiWindowFlags host_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                      ImGuiWindowFlags_NoNavFocus;
        ImGui::Begin("Host", nullptr, host_flags);
        ImGuiID dockspace_id = ImGui::GetID("K4ADockSpace");
        static bool dock_layout_built = false;
        if (!dock_layout_built)
        {
            dock_layout_built = true;
            ImGui::DockBuilderRemoveNode(dockspace_id);
            ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockspace_id, vp->WorkSize);
            ImGuiID dock_right = dockspace_id;
            const float left_ratio =
                std::clamp(440.f / std::max(320.f, vp->WorkSize.x), 0.18f, 0.42f);
            ImGuiID dock_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, left_ratio, nullptr, &dock_right);
            ImGui::DockBuilderDockWindow("Settings", dock_left);
            ImGui::DockBuilderDockWindow("Streams", dock_right);
            ImGui::DockBuilderFinish(dockspace_id);
        }
        ImGui::DockSpace(dockspace_id, ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);
        ImGui::End();
        ImGui::PopStyleVar(3);

        ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(440, vp->WorkSize.y), ImGuiCond_FirstUseEver);

        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoCollapse);
        ImGui::Text("Device S/N: %s", serial.c_str());
        if (!last_err.empty())
            ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%s", last_err.c_str());

        if (ImGui::Button("Close device", ImVec2(-1, 0)) && dev)
        {
            if (streaming)
            {
                if (imu_running)
                {
                    dev->stop_imu();
                    imu_running = false;
                }
                dev->stop_cameras();
                streaming = false;
            }
            xform.reset();
            have_depth_intrin = false;
            have_color_intrin = false;
            have_color_distortion = false;
            color_dist_coeff.release();
            undistort_map1.release();
            undistort_map2.release();
            undistort_map_size = cv::Size(0, 0);
            dev.reset();
            serial = "(not connected)";
        }
        if (ImGui::Button("Open device", ImVec2(-1, 0)) && !dev)
            try_open_device();

        ImGui::Separator();
        ImGui::Checkbox("Enable Depth Camera", &ui.depth_on);
        if (ImGui::CollapsingHeader("Depth Configuration", ImGuiTreeNodeFlags_DefaultOpen))
        {
            for (int i = 0; i < 5; ++i)
                ImGui::RadioButton(kDepthLabels[i], &ui.depth_mode, i);
        }
        ImGui::Checkbox("Enable Color Camera", &ui.color_on);
        if (ImGui::CollapsingHeader("Color Configuration", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Format");
            for (int i = 0; i < 4; ++i)
                ImGui::RadioButton(kColorFmtLabels[i], &ui.color_format, i);
            ImGui::TextDisabled(
                "NV12/YUY2 + 30 FPS: 720p만 SDK 지원. 1080p+는 BGRA 또는 MJPG, 또는 15 FPS.");
            ImGui::Text("Resolution 16:9");
            for (int i = 0; i < 4; ++i)
                ImGui::RadioButton(kRes169Labels[i], &ui.color_res, i);
            ImGui::Text("Resolution 4:3");
            for (int i = 0; i < 2; ++i)
                ImGui::RadioButton(kRes43Labels[i], &ui.color_res, i + 4);
        }
        ImGui::Text("Framerate");
        for (int i = 0; i < 3; ++i)
            ImGui::RadioButton(kFpsLabels[i], &ui.fps, i);
        ImGui::Checkbox("Disable streaming LED", &ui.disable_led);
        ImGui::Checkbox("Enable IMU", &ui.imu_on);
        ImGui::Checkbox("Enable Microphone", &ui.mic_ui);
        if (ui.mic_ui)
            ImGui::TextDisabled("(Kinect mic uses Windows audio; not opened here.)");

        if (ImGui::CollapsingHeader("Device Firmware Version Info") && dev)
        {
            try
            {
                k4a_hardware_version_t v = dev->get_version();
                ImGui::Text("RGB: %u.%u.%u", v.rgb.major, v.rgb.minor, v.rgb.iteration);
                ImGui::Text("Depth: %u.%u.%u", v.depth.major, v.depth.minor, v.depth.iteration);
                ImGui::Text("Audio: %u.%u.%u", v.audio.major, v.audio.minor, v.audio.iteration);
            }
            catch (...)
            {
                ImGui::Text("(unavailable)");
            }
        }

        if (ImGui::Button("Restore", ImVec2(120, 0)))
        {
            ui = {};
        }
        ImGui::SameLine();
        if (ImGui::Button("Save", ImVec2(120, 0)))
        {
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset", ImVec2(120, 0)))
        {
            ui = {};
        }

        const bool can_stream = dev != nullptr;
        if (streaming)
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.1f, 0.1f, 1));
            if (ImGui::Button("Stop", ImVec2(-1, 36)) && dev)
            {
                if (imu_running)
                {
                    dev->stop_imu();
                    imu_running = false;
                }
                dev->stop_cameras();
                streaming = false;
                xform.reset();
                have_depth_intrin = false;
                have_color_intrin = false;
                have_color_distortion = false;
                color_dist_coeff.release();
                undistort_map1.release();
                undistort_map2.release();
                undistort_map_size = cv::Size(0, 0);
            }
            ImGui::PopStyleColor();
        }
        else
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.55f, 0.2f, 1));
            if (ImGui::Button("Start", ImVec2(-1, 36)) && can_stream)
            {
                try
                {
                    std::string color_fix_note;
                    fix_unsupported_color_combo(ui, &color_fix_note);
                    k4a_device_configuration_t cfg = build_k4a_config(ui);
                    dev->start_cameras(&cfg);
                    if (!color_fix_note.empty())
                        last_err = color_fix_note;
                    xform.reset();
                    k4a_calibration_t cal{};
                    if (K4A_RESULT_SUCCEEDED ==
                        k4a_device_get_calibration(dev->handle(), cfg.depth_mode, cfg.color_resolution, &cal))
                    {
                        xform.emplace(cal);
                        const auto& p = cal.depth_camera_calibration.intrinsics.parameters.param;
                        depth_fx = p.fx;
                        depth_fy = p.fy;
                        depth_cx = p.cx;
                        depth_cy = p.cy;
                        have_depth_intrin = (depth_fx > 1e-3f && depth_fy > 1e-3f);
                        const auto& pc = cal.color_camera_calibration.intrinsics.parameters.param;
                        color_fx = pc.fx;
                        color_fy = pc.fy;
                        color_cx = pc.cx;
                        color_cy = pc.cy;
                        have_color_intrin = (color_fx > 1e-3f && color_fy > 1e-3f);
                        color_dist_coeff =
                            (cv::Mat_<double>(1, 8) << pc.k1, pc.k2, pc.p1, pc.p2, pc.k3, pc.k4, pc.k5, pc.k6);
                        have_color_distortion = true;
                        undistort_map1.release();
                        undistort_map2.release();
                        undistort_map_size = cv::Size(0, 0);
                    }
                    if (ui.imu_on)
                    {
                        try
                        {
                            dev->start_imu();
                            imu_running = true;
                        }
                        catch (const k4a::error& e)
                        {
                            last_err = std::string("IMU: ") + e.what();
                            imu_running = false;
                        }
                    }
                    streaming = true;
                }
                catch (const k4a::error& e)
                {
                    last_err = e.what();
                }
            }
            ImGui::PopStyleColor();
        }

        ImGui::Text("View Mode");
        ImGui::RadioButton("2D", &ui.view_mode, kView2D);
        ImGui::SameLine();
        ImGui::RadioButton("3D", &ui.view_mode, kView3D);
        if (ImGui::Button("Pause", ImVec2(-1, 36)))
        {
        }
        ImGui::Separator();
        ImGui::TextUnformatted("2D Streams — 표시");
        ImGui::Checkbox("IR", &ui.show_ir);
        ImGui::SameLine();
        ImGui::Checkbox("Depth", &ui.show_depth);
        ImGui::SameLine();
        ImGui::Checkbox("Color", &ui.show_color);
        ImGui::Checkbox("IMU 패널", &ui.show_imu_panel);
        ImGui::SameLine();
        ImGui::Checkbox("Floor 합성", &ui.show_floor_panel);
        int stride = std::clamp(ui.floor_update_stride, 1, 8);
        if (ImGui::SliderInt("RVM·매트 추론 간격 (프레임)", &stride, 1, 8))
            ui.floor_update_stride = stride;
        ImGui::Checkbox("Depth-only person segment (fast)", &ui.person_segment_depth_only);
        if (ui.person_segment_depth_only)
        {
            ImGui::SameLine();
            if (ImGui::SmallButton("Switch to RVM"))
                ui.person_segment_depth_only = false;
            ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.25f, 1.0f),
                               "현재 Depth-only ON: RVM 화면이 아니라 Depth 분할 화면입니다.");
        }
        else
        {
            ImGui::TextColored(ImVec4(0.45f, 1.0f, 0.55f, 1.0f), "현재 RVM 모드 ON");
        }
        ImGui::TextDisabled("합성(그림자·바닥)은 매 프레임 최신 컬러·깊이로 갱신. 무거운 추론만 N프레임.");
        ImGui::Separator();
        ImGui::TextUnformatted("RVM + Floor (depth-matched matte)");
        ImGui::Checkbox("Enable floor / shadow / reflection", &ui.floor_fx_enabled);
        ImGui::SliderFloat("Floor contact shadow", &ui.floor_shadow, 0.0f, 1.0f);
        ImGui::SliderFloat("Reflection strength", &ui.floor_refl, 0.0f, 1.0f);
        ImGui::SliderFloat("RVM downsample_ratio", &ui.floor_ds_ratio, 0.1f, 1.0f);
        ImGui::Separator();
        ImGui::TextUnformatted("Floor — grid / sun / reflection");
        ImGui::Checkbox("Green floor grid", &ui.floor_grid_show);
        ImGui::SliderInt("Grid spacing (px)", &ui.floor_grid_spacing, 8, 80);
        ImGui::SliderFloat("Sun altitude (deg)", &ui.sun_altitude_deg, 8.f, 88.f);
        ImGui::SliderFloat("Sun azimuth (deg)", &ui.sun_azimuth_deg, 0.f, 360.f);
        ImGui::SliderFloat("Cast shadow alpha", &ui.cast_shadow_alpha, 0.f, 1.f);
        ImGui::SliderFloat("Cast shadow blur", &ui.cast_shadow_blur, 0.f, 40.f);
        ImGui::Checkbox("Reflection on floor", &ui.reflection_on);
        ImGui::TextDisabled("체크 해제해도 Reflection strength > 0.06이면 바닥 반사 적용.");
        ImGui::SliderFloat("Reflection plane Y (0-1)", &ui.reflection_plane_y, 0.35f, 0.92f);
        ImGui::Checkbox("Single reflection layer (avoid double blend)", &ui.reflection_single_pass_only);
        ImGui::Checkbox("Reflection lens correction (color undistort)", &ui.reflection_undistort_input);
        ImGui::Checkbox("Plane fit: foot ROI", &ui.plane_use_foot_roi);
        ImGui::SliderFloat("Plane foot ROI radius", &ui.plane_foot_roi_radius, 0.08f, 0.50f, "%.2f");
        ImGui::Checkbox("Plane temporal smoothing", &ui.plane_temporal_smooth_on);
        ImGui::SliderFloat("Plane temporal alpha", &ui.plane_temporal_alpha, 0.02f, 1.0f, "%.2f");
        static const char* kRvmBgModes[] = { "Black (no composite)", "Color background", "Shadow/Reflection FX" };
        ImGui::Combo("RVM background mode", &ui.rvm_bg_mode, kRvmBgModes, IM_ARRAYSIZE(kRvmBgModes));
        ImGui::Checkbox("RVM floor overlay", &ui.rvm_floor_overlay_on);
        ImGui::Checkbox("RVM shadow overlay", &ui.rvm_shadow_overlay_on);
        ImGui::Checkbox("RVM reflection overlay", &ui.rvm_reflection_overlay_on);
        static const char* kColorViewModes[] = { "Raw color", "RVM human matte", "RVM matte + depth shadow" };
        ImGui::Combo("Color panel mode", &ui.color_view_mode, kColorViewModes, IM_ARRAYSIZE(kColorViewModes));
        static const char* kColorDepthModes[] = { "Depth view: Off", "Depth view: RGB overlay", "Depth view: Depth only" };
        ImGui::Combo("RGB depth view", &ui.color_depth_view_mode, kColorDepthModes, IM_ARRAYSIZE(kColorDepthModes));
        ImGui::SliderFloat("RGB depth overlay alpha", &ui.color_depth_overlay_alpha, 0.0f, 1.0f);
        ImGui::Checkbox("RGB depth hole fill", &ui.rgb_depth_hole_fill);
        ImGui::SliderInt("RGB depth fill iterations", &ui.rgb_depth_fill_iters, 0, 8);
        int kfill = ui.rgb_depth_fill_kernel | 1;
        if (ImGui::SliderInt("RGB depth fill kernel", &kfill, 3, 11))
            ui.rgb_depth_fill_kernel = (kfill | 1);
        if (ImGui::Button("Clear floor click anchor"))
            ui.floor_anchor_valid = false;
        ImGui::TextDisabled("Streams: Floor 패널에서 클릭 = 발/접지 기준(그림자 피벗).");
        ImGui::Separator();
        ImGui::Checkbox("MR menu (2nd column)", &ui.mr_menu_enabled);
        if (ui.mr_menu_enabled && ImGui::BeginTable("mr_menu_2col", 2, ImGuiTableFlags_BordersInnerV))
        {
            ImGui::TableSetupColumn("Current viewer bridge", ImGuiTableColumnFlags_WidthStretch, 0.58f);
            ImGui::TableSetupColumn("MR roadmap control", ImGuiTableColumnFlags_WidthStretch, 0.42f);
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted("Depth/RGB + PointCloud + RVM preview");
            ImGui::BulletText("Stage 1: Depth->PointCloud->GL widget (implemented baseline)");
            ImGui::BulletText("Stage 4 focus: Real depth as GPU depth for occlusion");
            ImGui::BulletText("Color panel mode=2: RVM human matte + depth shadow preview");
            ImGui::BulletText("Floor panel: reflection/shadow parameter validation");
            ImGui::Separator();
            ImGui::TextDisabled("Target env: Python + PySide6 + OpenGL + OpenCV + NumPy + Open3D");
            ImGui::TextDisabled("Design intent: Modern OpenGL(VBO/VAO/FBO/GLSL), GPU-first");

            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted("Mixed Reality Pipeline");
            ImGui::SliderInt("MR Stage", &ui.mr_stage, 1, 8);
            ImGui::Checkbox("Depth Occlusion", &ui.mr_occlusion_enabled);
            ImGui::Checkbox("Occlusion Debug Overlay", &ui.mr_occlusion_debug);
            ImGui::Checkbox("Occlusion debug: local ROI only", &ui.mr_occlusion_debug_local_only);
            ImGui::SliderFloat("Occlusion debug ROI x radius", &ui.mr_occlusion_debug_roi_mul, 1.0f, 12.0f, "%.1f");
            ImGui::Checkbox("Shadow Pass", &ui.mr_shadow_enabled);
            ImGui::Checkbox("Reflection Pass", &ui.mr_reflection_enabled);
            ImGui::SliderFloat("Depth edge smooth", &ui.mr_depth_edge_smooth, 0.0f, 6.0f, "%.2f");
            ImGui::SliderFloat("Target FPS", &ui.mr_target_fps, 5.0f, 120.0f, "%.0f");
            ImGui::SliderFloat("Virtual depth (mm)", &ui.mr_virtual_depth_mm, 500.0f, 8000.0f, "%.0f");
            ImGui::SliderFloat("Occlusion overlay alpha", &ui.mr_occlusion_overlay_alpha, 0.0f, 1.0f, "%.2f");
            ImGui::Separator();
            ImGui::Checkbox("Virtual object (debug)", &ui.mr_virtual_object_enabled);
            static const char* kObjShapes[] = { "Disc", "Box" };
            ImGui::Combo("Object shape", &ui.mr_object_shape, kObjShapes, IM_ARRAYSIZE(kObjShapes));
            ImGui::SliderFloat("Object U", &ui.mr_object_u, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Object V", &ui.mr_object_v, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Object radius (px)", &ui.mr_object_radius_px, 8.0f, 420.0f, "%.1f");
            ImGui::SliderFloat("Object depth (mm)", &ui.mr_object_depth_mm, 500.0f, 8000.0f, "%.0f");
            ImGui::SliderFloat("Object shadow alpha", &ui.mr_object_shadow_alpha, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Object reflection alpha", &ui.mr_object_reflection_alpha, 0.0f, 1.0f, "%.2f");
            ImGui::Checkbox("Show picked XYZ", &ui.mr_show_pick_xyz);
            ImGui::Checkbox("Use world transform", &ui.mr_use_world_transform);
            ImGui::SliderFloat("Obj Tx (m)", &ui.mr_obj_tx_m, -6.0f, 6.0f, "%.3f");
            ImGui::SliderFloat("Obj Ty (m)", &ui.mr_obj_ty_m, -4.0f, 4.0f, "%.3f");
            ImGui::SliderFloat("Obj Tz (m)", &ui.mr_obj_tz_m, 0.3f, 10.0f, "%.3f");
            ImGui::SliderFloat("Obj Scale", &ui.mr_obj_scale, 0.1f, 4.0f, "%.3f");
            ImGui::SliderFloat("Obj Rot Z (deg)", &ui.mr_obj_rot_deg, -180.0f, 180.0f, "%.1f");
            ImGui::Checkbox("Ground snap on click", &ui.mr_ground_snap_on_click);
            ImGui::SliderFloat("Ground snap offset (m)", &ui.mr_ground_snap_offset_m, -1.0f, 1.0f, "%.3f");
            ImGui::TextDisabled("Color 패널 클릭으로 object 위치 배치");
            ImGui::Text("MR FPS: %.1f / target %.1f", mr_fps_smoothed, ui.mr_target_fps);
            ImGui::Text("MR latency: %.2f ms", mr_latency_ms_smoothed);
            const bool pass_obj = ui.mr_virtual_object_enabled;
            const bool pass_occ = ui.mr_occlusion_enabled;
            const bool pass_sh = ui.mr_shadow_enabled;
            const bool pass_rf = ui.mr_reflection_enabled;
            ImGui::TextDisabled("Passes: Obj[%s] Occ[%s] Sh[%s] Refl[%s]", pass_obj ? "on" : "off",
                                pass_occ ? "on" : "off", pass_sh ? "on" : "off", pass_rf ? "on" : "off");
            if (ui.mr_show_pick_xyz)
            {
                if (mr_pick_valid)
                    ImGui::Text("Pick XYZ(m): X=%.3f  Y=%.3f  Z=%.3f  (u=%d v=%d)", mr_pick_x_m, mr_pick_y_m,
                                mr_pick_z_m, mr_pick_u, mr_pick_v);
                else
                    ImGui::TextDisabled("Pick XYZ: no valid depth/intrinsic");
            }
            ImGui::Separator();
            if (ui.mr_stage == 4)
                ImGui::TextColored(ImVec4(0.45f, 1.0f, 0.55f, 1.0f), "Priority: Real-depth GPU occlusion");
            else
                ImGui::TextDisabled("Priority: stage 4 (real depth occlusion)");
            ImGui::TextWrapped("Render order plan: RGB -> RealDepth -> VirtualObj -> Occlusion -> Shadow -> Reflection -> PostFX");
            ImGui::TextWrapped("Required classes: DepthTextureManager / DepthOcclusionRenderer / RGBDAligner");
            ImGui::EndTable();
        }
        ImGui::Separator();
        ImGui::TextUnformatted("Preset template (INI)");
        ImGui::InputText("Path##preset", preset_path, IM_ARRAYSIZE(preset_path));
        if (ImGui::Button("Save preset##p"))
        {
            viewer_io::KVMap m;
            ui_to_kv(ui, rvm_onnx_buf, m);
            if (!viewer_io::save_ini(std::string(preset_path), m))
                last_err = "Preset save failed.";
            else
                last_err.clear();
        }
        ImGui::SameLine();
        if (ImGui::Button("Load preset##p"))
        {
            viewer_io::KVMap m;
            if (!viewer_io::load_ini(std::string(preset_path), m))
                last_err = "Preset load failed.";
            else
            {
                kv_to_ui(m, ui, rvm_onnx_buf, IM_ARRAYSIZE(rvm_onnx_buf));
                last_err.clear();
                rvm_autoload_attempted = true;
                if (rvm_floor::try_load_onnx(rvm_eng, std::string(rvm_onnx_buf)))
                {
                }
            }
        }
        if (!rvm_eng.net_loaded && !rvm_autoload_attempted && rvm_onnx_buf[0] != '\0')
        {
            rvm_autoload_attempted = true;
            if (rvm_floor::try_load_onnx(rvm_eng, std::string(rvm_onnx_buf)))
                last_err.clear();
            else
                last_err = rvm_eng.last_error.empty() ? "RVM ONNX auto load failed." : ("RVM ONNX auto load failed: " + rvm_eng.last_error);
        }
        ImGui::InputText("RVM ONNX path", rvm_onnx_buf, IM_ARRAYSIZE(rvm_onnx_buf));
        const bool need_rvm_load = !rvm_eng.net_loaded;
        if (need_rvm_load)
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.70f, 0.15f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.82f, 0.22f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.58f, 0.10f, 0.10f, 1.0f));
        }
        if (ImGui::Button("Load RVM ONNX", ImVec2(-1, 0)))
        {
            rvm_autoload_attempted = true;
            if (rvm_floor::try_load_onnx(rvm_eng, std::string(rvm_onnx_buf)))
                last_err.clear();
            else
                last_err = rvm_eng.last_error.empty()
                               ? "RVM ONNX load failed (unknown reason)."
                               : ("RVM ONNX load failed: " + rvm_eng.last_error);
        }
        if (need_rvm_load)
            ImGui::PopStyleColor(3);
        if (rvm_eng.net_loaded)
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, 1), "RVM network: loaded (%s)",
                               rvm_eng.backend_name.empty() ? "unknown backend" : rvm_eng.backend_name.c_str());
        else
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.35f, 1.0f),
                               "RVM network: not loaded (RVM 모드를 보려면 ONNX 로드 필요)");
        if (matte_mode_line.find("RVM") != std::string::npos)
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, 1), "%s", matte_mode_line.c_str());
        else if (matte_mode_line.find("Depth") != std::string::npos)
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.35f, 1), "%s", matte_mode_line.c_str());
        else
            ImGui::TextDisabled("%s", matte_mode_line.c_str());
        ImGui::End();

        std::string imu_line = imu_running ? "IMU: (no sample)" : "IMU: off";
        static int last_cloud_style = -1;
        if (!streaming || !dev)
        {
            cap_last.reset();
            m_ir.release();
            m_depth.release();
            m_color.release();
            k4a_dep = k4a::image();
            k4a_col = k4a::image();
        }
        else
        {
            k4a::capture cap;
            if (dev->get_capture(&cap, std::chrono::milliseconds(2)))
                cap_last = std::move(cap);
            if (cap_last.has_value())
            {
                k4a::image ir = cap_last->get_ir_image();
                if (ir)
                    m_ir = ir16_to_vis(ir);
                k4a_dep = cap_last->get_depth_image();
                if (k4a_dep)
                    m_depth = depth_to_colormap(k4a_dep);
                k4a_col = cap_last->get_color_image();
                if (k4a_col)
                    m_color = k4a_color_to_bgr8(k4a_col);
                if (imu_running)
                {
                    k4a_imu_sample_t s{};
                    if (dev->get_imu_sample(&s, std::chrono::milliseconds(0)))
                    {
                        std::ostringstream oss;
                        oss << "acc " << std::fixed << std::setprecision(2) << s.acc_sample.xyz.x << ", "
                            << s.acc_sample.xyz.y << ", " << s.acc_sample.xyz.z;
                        imu_line = oss.str();
                    }
                }
                if (ui.view_mode == kView3D && xform.has_value() && k4a_dep)
                {
                    const bool tick = (++frame_counter % 4) == 0;
                    const bool style_changed = (last_cloud_style != ui.cloud_style);
                    if (tick || style_changed)
                    {
                        try
                        {
                            const k4a::image* col_ptr = k4a_col ? &k4a_col : nullptr;
                            pcgl.update(*xform, k4a_dep, col_ptr, ui.cloud_style, 3);
                            last_cloud_style = ui.cloud_style;
                        }
                        catch (const k4a::error&)
                        {
                        }
                    }
                }
            }
        }

        // RGB 기준 깊이맵(색상 해상도/화면비) 준비: Color 창 depth 토글/오버레이/MR용
        // NOTE: 매 프레임 depth_to_color 변환은 무거워 capturesync_drop(큐 full)를 유발할 수 있어 N프레임 스로틀.
        const bool want_rgb_depth =
            ui.show_color && (ui.color_depth_view_mode != 0 || ui.mr_virtual_object_enabled || ui.floor_fx_enabled ||
                              ui.rvm_reflection_overlay_on);
        static int rgb_align_counter = 0;
        const int rgb_align_stride = std::max(1, ui.floor_update_stride);
        const bool do_rgb_align =
            want_rgb_depth && (++rgb_align_counter % rgb_align_stride == 0 || m_depth_rgb_u16.empty() || m_color_mr_rgb.empty());

        if (do_rgb_align && streaming && xform.has_value() && k4a_dep && k4a_col)
        {
            cv::Mat color_rgb, depth_rgb;
            if (depth_to_color_aligned_mats(*xform, k4a_dep, k4a_col, color_rgb, depth_rgb))
            {
                if (ui.reflection_undistort_input && have_color_intrin && have_color_distortion && !color_dist_coeff.empty())
                {
                    const cv::Size sz = color_rgb.size();
                    if (undistort_map_size != sz || undistort_map1.empty() || undistort_map2.empty())
                    {
                        const cv::Mat K = (cv::Mat_<double>(3, 3) << color_fx, 0.0, color_cx, 0.0, color_fy, color_cy, 0.0, 0.0, 1.0);
                        cv::initUndistortRectifyMap(K,
                                                    color_dist_coeff,
                                                    cv::Mat(),
                                                    K,
                                                    sz,
                                                    CV_32FC1,
                                                    undistort_map1,
                                                    undistort_map2);
                        undistort_map_size = sz;
                    }
                    cv::Mat color_ud, depth_ud;
                    cv::remap(color_rgb, color_ud, undistort_map1, undistort_map2, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
                    cv::remap(depth_rgb, depth_ud, undistort_map1, undistort_map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
                    color_rgb = color_ud;
                    depth_rgb = depth_ud;
                }
                // 깊이 원본(0 포함)은 판정/그림자용으로 유지하고,
                // 채운(depth hole fill) 버전은 "표시용 overlay"에만 사용한다.
                m_depth_rgb_u16 = depth_rgb;
                m_depth_rgb_u16_filled = depth_rgb;
                if (ui.rgb_depth_hole_fill)
                    m_depth_rgb_u16_filled =
                        fill_depth_holes_fast(m_depth_rgb_u16_filled, ui.rgb_depth_fill_iters, ui.rgb_depth_fill_kernel);
                m_depth_rgb_vis = depth_u16_to_colormap(m_depth_rgb_u16_filled);
                // Raw color에서도 MR/사람그림자가 보이도록 RGB 기준 합성 프레임 구성
                m_color_mr_rgb = color_rgb.clone();
                if (ui.show_color && (ui.floor_fx_enabled || ui.rvm_reflection_overlay_on))
                {
                    ++rgb_stride_counter;
                    const int st = std::max(1, ui.floor_update_stride);
                    const bool refresh_alpha_rgb = ((rgb_stride_counter - 1) % st) == 0;
                    if (refresh_alpha_rgb)
                    {
                        cv::Mat alpha;
                        if (!ui.person_segment_depth_only)
                        {
                            // RVM 검증 모드: depth fallback을 섞지 않고 RVM만 사용
                            if (rvm_floor::infer_alpha(rvm_eng, color_rgb, alpha, ui.floor_ds_ratio))
                            {
                                m_rgb_alpha = alpha;
                                m_rgb_alpha_refl_fb.release();
                            }
                            else
                            {
                                m_rgb_alpha.release();
                                if (ui.rvm_reflection_overlay_on)
                                {
                                    rvm_floor::infer_alpha_depth_fallback(color_rgb, m_depth_rgb_u16, alpha);
                                    m_rgb_alpha_refl_fb = alpha;
                                }
                                else
                                    m_rgb_alpha_refl_fb.release();
                            }
                        }
                        else
                        {
                            rvm_floor::infer_alpha_depth_fallback(color_rgb, m_depth_rgb_u16, alpha);
                            m_rgb_alpha = alpha;
                            m_rgb_alpha_refl_fb.release();
                        }
                    }
                    cv::Mat alpha_use;
                    if (!m_rgb_alpha.empty())
                        alpha_use = m_rgb_alpha;
                    else if (ui.rvm_reflection_overlay_on && !m_rgb_alpha_refl_fb.empty())
                        m_rgb_alpha_refl_fb.copyTo(alpha_use);
                    if (!alpha_use.empty())
                    {
                        if (alpha_use.size() != color_rgb.size())
                            cv::resize(alpha_use, alpha_use, color_rgb.size(), 0, 0, cv::INTER_LINEAR);
                        // 동일한 경계 안정화(halo 억제)
                        cv::GaussianBlur(alpha_use, alpha_use, cv::Size(0, 0), 1.25);
                        cv::morphologyEx(alpha_use,
                                         alpha_use,
                                         cv::MORPH_CLOSE,
                                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
                        cv::Mat a8, a8_er, a_er_f;
                        alpha_use.convertTo(a8, CV_8U, 255.0);
                        cv::erode(a8, a8_er, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
                        a8_er.convertTo(a_er_f, CV_32F, 1.0 / 255.0);
                        alpha_use = alpha_use.mul(a_er_f * 0.75f + 0.25f);
                        // depth hole은 알파를 약화
                        for (int y = 0; y < alpha_use.rows; ++y)
                        {
                            const uint16_t* dr = m_depth_rgb_u16.ptr<uint16_t>(y);
                            float* ar = alpha_use.ptr<float>(y);
                            for (int x = 0; x < alpha_use.cols; ++x)
                                if (dr[x] == 0)
                                    ar[x] *= 0.30f;
                        }

                        rvm_floor::Params pp;
                        pp.enabled = true;
                        pp.shadow_strength = ui.floor_shadow;
                        pp.reflection_strength = ui.floor_refl;
                        pp.downsample_ratio = ui.floor_ds_ratio;
                        pp.floor_grid_show = false;
                        pp.floor_grid_spacing = ui.floor_grid_spacing;
                        pp.sun_altitude_deg = ui.sun_altitude_deg;
                        pp.sun_azimuth_deg = ui.sun_azimuth_deg;
                        pp.floor_anchor_valid = ui.floor_anchor_valid;
                        pp.floor_anchor_u = ui.floor_anchor_u;
                        pp.floor_anchor_v = ui.floor_anchor_v;
                        pp.cast_shadow_alpha = ui.cast_shadow_alpha;
                        pp.cast_shadow_blur = ui.cast_shadow_blur;
                        pp.cam_intrin_valid = have_color_intrin;
                        pp.cam_fx = color_fx;
                        pp.cam_fy = color_fy;
                        pp.cam_cx = color_cx;
                        pp.cam_cy = color_cy;
                        pp.shadow_ray_step = ui.person_segment_depth_only ? 4 : 2;
                        pp.reflection_on =
                            (ui.reflection_on || (ui.floor_refl > 0.06f)) &&
                            !(ui.reflection_single_pass_only && ui.rvm_reflection_overlay_on);
                        pp.reflection_plane_y = ui.reflection_plane_y;
                        pp.plane_use_foot_roi = ui.plane_use_foot_roi;
                        pp.plane_foot_roi_radius = ui.plane_foot_roi_radius;
                        pp.plane_temporal_smooth_on = ui.plane_temporal_smooth_on;
                        pp.plane_temporal_alpha = ui.plane_temporal_alpha;

                        bool mr_rgb_is_full_compose_output = false;
                        if (ui.floor_fx_enabled)
                        {
                            cv::Mat shadowed = rvm_floor::compose_floor_with_fx(color_rgb, m_depth_rgb_u16, alpha_use, pp);
                            if (!shadowed.empty())
                            {
                                if (ui.rvm_floor_overlay_on && ui.rvm_shadow_overlay_on)
                                {
                                    m_color_mr_rgb = shadowed;
                                    mr_rgb_is_full_compose_output = true;
                                }
                                else
                                {
                                cv::Mat dark_raw(color_rgb.size(), CV_32F, cv::Scalar(0));
                                for (int y = 0; y < dark_raw.rows; ++y)
                                {
                                    const cv::Vec3b* sp = color_rgb.ptr<cv::Vec3b>(y);
                                    const cv::Vec3b* hp = shadowed.ptr<cv::Vec3b>(y);
                                    float* dp = dark_raw.ptr<float>(y);
                                    for (int x = 0; x < dark_raw.cols; ++x)
                                    {
                                        const float l0 = (0.114f * sp[x][0] + 0.587f * sp[x][1] + 0.299f * sp[x][2]) + 1.0f;
                                        const float l1 = (0.114f * hp[x][0] + 0.587f * hp[x][1] + 0.299f * hp[x][2]) + 1.0f;
                                        dp[x] = std::clamp((l0 - l1) / std::max(8.0f, l0), 0.0f, 1.0f);
                                    }
                                }
                                cv::Mat dark_floor, dark_soft, dark_bin;
                                cv::GaussianBlur(dark_raw, dark_floor, cv::Size(0, 0), std::max(10.0, (double)ui.cast_shadow_blur * 2.0));
                                cv::GaussianBlur(dark_raw, dark_soft, cv::Size(0, 0), std::max(1.2, (double)ui.cast_shadow_blur * 0.85));
                                cv::threshold(dark_soft, dark_bin, 0.045, 1.0, cv::THRESH_BINARY);
                                cv::dilate(dark_bin, dark_bin,
                                           cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
                                cv::GaussianBlur(dark_bin, dark_bin, cv::Size(0, 0), std::max(0.8, (double)ui.cast_shadow_blur * 0.7));

                                cv::Mat cast_w = rvm_floor::compute_cast_shadow_weight(color_rgb, m_depth_rgb_u16, alpha_use, pp);
                                const bool use_3d_person_shadow = !cast_w.empty();
                                cv::Mat sh_roi_f;
                                if (!use_3d_person_shadow)
                                {
                                    cv::Mat person_u8;
                                    cv::threshold(alpha_use, person_u8, 0.20, 255.0, cv::THRESH_BINARY);
                                    person_u8.convertTo(person_u8, CV_8U);
                                    const float alt_rad =
                                        std::clamp(ui.sun_altitude_deg, 6.0f, 88.0f) * 3.14159265f / 180.0f;
                                    const float azi_rad = ui.sun_azimuth_deg * 3.14159265f / 180.0f;
                                    float stretch = 52.f / std::tan(std::max(0.12f, alt_rad));
                                    stretch = std::clamp(stretch, 6.f, 240.f);
                                    const float tx = -std::cos(azi_rad) * stretch;
                                    const float ty = -std::sin(azi_rad) * stretch * 0.42f;
                                    const cv::Mat M = (cv::Mat_<double>(2, 3) << 1., 0., static_cast<double>(tx), 0., 1.,
                                                       static_cast<double>(ty));
                                    cv::Mat sh_roi_u8;
                                    cv::warpAffine(person_u8, sh_roi_u8, M, person_u8.size(), cv::INTER_LINEAR,
                                                   cv::BORDER_CONSTANT, cv::Scalar(0));
                                    cv::dilate(sh_roi_u8, sh_roi_u8,
                                               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));
                                    sh_roi_u8.convertTo(sh_roi_f, CV_32F, 1.0 / 255.0);
                                    cv::GaussianBlur(sh_roi_f, sh_roi_f, cv::Size(0, 0),
                                                     std::max(1.0, (double)ui.cast_shadow_blur));
                                }

                                m_color_mr_rgb = color_rgb.clone();
                                for (int y = 0; y < m_color_mr_rgb.rows; ++y)
                                {
                                    const cv::Vec3b* sp = color_rgb.ptr<cv::Vec3b>(y);
                                    const float* fp = dark_floor.ptr<float>(y);
                                    const float* ds = dark_soft.ptr<float>(y);
                                    const float* db = dark_bin.ptr<float>(y);
                                    const float* cwp =
                                        use_3d_person_shadow ? cast_w.ptr<float>(y) : sh_roi_f.ptr<float>(y);
                                    cv::Vec3b* op = m_color_mr_rgb.ptr<cv::Vec3b>(y);
                                    for (int x = 0; x < m_color_mr_rgb.cols; ++x)
                                    {
                                        const float rp = cwp[x];
                                        const float floor_w =
                                            ui.rvm_floor_overlay_on ? std::clamp(fp[x] * 0.45f, 0.0f, 1.0f) : 0.0f;
                                        const float sh_unified =
                                            std::clamp(0.35f * ds[x] + 0.95f * db[x], 0.0f, 1.0f);
                                        const float shadow_w =
                                            ui.rvm_shadow_overlay_on
                                                ? (use_3d_person_shadow
                                                       ? std::clamp(rp * ui.cast_shadow_alpha * 1.25f, 0.0f, 1.0f)
                                                       : std::clamp(sh_unified * rp * ui.cast_shadow_alpha * 1.25f,
                                                                    0.0f, 1.0f))
                                                : 0.0f;
                                        const float dark_w = 1.0f - (1.0f - floor_w) * (1.0f - shadow_w);
                                        op[x][0] = static_cast<uint8_t>(std::clamp(sp[x][0] * (1.0f - dark_w), 0.0f, 255.0f));
                                        op[x][1] = static_cast<uint8_t>(std::clamp(sp[x][1] * (1.0f - dark_w), 0.0f, 255.0f));
                                        op[x][2] = static_cast<uint8_t>(std::clamp(sp[x][2] * (1.0f - dark_w), 0.0f, 255.0f));
                                    }
                                }
                                }
                            }
                        }
                        const bool want_3d_refl = ui.rvm_reflection_overlay_on && have_color_intrin;
                        const bool want_compose_refl =
                            (ui.reflection_on || (ui.floor_refl > 0.06f)) &&
                            !(ui.reflection_single_pass_only && want_3d_refl);
                        if (want_compose_refl && !mr_rgb_is_full_compose_output && !m_color_mr_rgb.empty())
                            rvm_floor::apply_compose_style_floor_reflection(m_color_mr_rgb,
                                                                            color_rgb,
                                                                            m_depth_rgb_u16,
                                                                            alpha_use,
                                                                            pp);
                        if (want_3d_refl && !m_color_mr_rgb.empty())
                            rvm_floor::apply_depth_mirror_person_reflection(m_color_mr_rgb,
                                                                          color_rgb,
                                                                          m_depth_rgb_u16,
                                                                          alpha_use,
                                                                          pp,
                                                                          ui.floor_refl);
                    }
                }

                if (!m_color_mr_rgb.empty() && ui.mr_virtual_object_enabled)
                {
                    const int W = m_color_mr_rgb.cols;
                    const int H = m_color_mr_rgb.rows;
                    float cx_obj = std::clamp(ui.mr_object_u, 0.f, 1.f) * static_cast<float>(W - 1);
                    float cy_obj = std::clamp(ui.mr_object_v, 0.f, 1.f) * static_cast<float>(H - 1);
                    float rad = std::max(4.f, ui.mr_object_radius_px);
                    float obj_depth = std::clamp(ui.mr_object_depth_mm, 500.f, 8000.f);

                    // RGB 정합 depth를 쓰는 경우엔 color intrinsics로 world transform 투영
                    if (ui.mr_use_world_transform && have_color_intrin)
                    {
                        const float tz = std::max(0.3f, ui.mr_obj_tz_m);
                        const float uproj = color_fx * (ui.mr_obj_tx_m / tz) + color_cx;
                        const float vproj = color_fy * (ui.mr_obj_ty_m / tz) + color_cy;
                        cx_obj = std::clamp(uproj, 0.0f, static_cast<float>(W - 1));
                        cy_obj = std::clamp(vproj, 0.0f, static_cast<float>(H - 1));
                        obj_depth = std::clamp(tz * 1000.0f, 500.0f, 8000.0f);
                        const float depth_scale = std::clamp(2.3f / tz, 0.35f, 4.0f);
                        rad = std::max(4.0f, ui.mr_object_radius_px * ui.mr_obj_scale * depth_scale);
                    }

                    const float inv_r2 = 1.0f / std::max(1.0f, rad * rad);
                    const cv::Vec3f obj_col(80.f, 220.f, 255.f); // BGR
                    const int x0 = std::max(0, static_cast<int>(std::floor(cx_obj - rad - 1.f)));
                    const int x1 = std::min(W - 1, static_cast<int>(std::ceil(cx_obj + rad + 1.f)));
                    const int y0 = std::max(0, static_cast<int>(std::floor(cy_obj - rad - 1.f)));
                    const int y1 = std::min(H - 1, static_cast<int>(std::ceil(cy_obj + rad + 1.f)));

                    cv::Mat obj_mask(H, W, CV_8U, cv::Scalar(0));
                    cv::Mat obj_vis_mask(H, W, CV_8U, cv::Scalar(0));
                    for (int y = y0; y <= y1; ++y)
                    {
                        cv::Vec3b* op = m_color_mr_rgb.ptr<cv::Vec3b>(y);
                        uint8_t* om = obj_mask.ptr<uint8_t>(y);
                        uint8_t* vm = obj_vis_mask.ptr<uint8_t>(y);
                        const uint16_t* dr = m_depth_rgb_u16.ptr<uint16_t>(y);
                        for (int x = x0; x <= x1; ++x)
                        {
                            const float dx = static_cast<float>(x) - cx_obj;
                            const float dy = static_cast<float>(y) - cy_obj;
                            float shape_w = 0.0f;
                            const float th = ui.mr_obj_rot_deg * 3.14159265f / 180.0f;
                            const float cs = std::cos(th);
                            const float sn = std::sin(th);
                            const float rx = dx * cs + dy * sn;
                            const float ry = -dx * sn + dy * cs;
                            if (ui.mr_object_shape == 0)
                            {
                                const float r2n = (dx * dx + dy * dy) * inv_r2;
                                if (r2n > 1.0f)
                                    continue;
                                shape_w = std::clamp(1.0f - r2n, 0.0f, 1.0f);
                            }
                            else
                            {
                                const float boxd = std::max(std::abs(rx), std::abs(ry)) / std::max(1.0f, rad);
                                if (boxd > 1.0f)
                                    continue;
                                shape_w = std::clamp(1.0f - boxd, 0.0f, 1.0f);
                            }

                            om[x] = 255;
                            const float local_depth = obj_depth - shape_w * 130.0f;
                            const uint16_t rz = dr[x];
                            const bool occluded = ui.mr_occlusion_enabled && (rz > 250) && (static_cast<float>(rz) < local_depth);
                            if (occluded)
                                continue;
                            vm[x] = 255;
                            const float w = 0.20f + 0.65f * std::sqrt(shape_w);
                            op[x][0] = static_cast<uint8_t>(std::clamp(op[x][0] * (1.f - w) + obj_col[0] * w, 0.f, 255.f));
                            op[x][1] = static_cast<uint8_t>(std::clamp(op[x][1] * (1.f - w) + obj_col[1] * w, 0.f, 255.f));
                            op[x][2] = static_cast<uint8_t>(std::clamp(op[x][2] * (1.f - w) + obj_col[2] * w, 0.f, 255.f));
                        }
                    }

                    if (ui.mr_shadow_enabled)
                    {
                        float alt_rad = std::clamp(ui.sun_altitude_deg, 6.0f, 88.0f) * 3.14159265f / 180.0f;
                        const float azi_rad = ui.sun_azimuth_deg * 3.14159265f / 180.0f;
                        float stretch = 48.f / std::tan(std::max(0.12f, alt_rad));
                        stretch = std::clamp(stretch, 6.f, 220.f);
                        float tx = -std::cos(azi_rad) * stretch;
                        float ty = -std::sin(azi_rad) * stretch * 0.35f;
                        tx = std::clamp(tx, -0.40f * static_cast<float>(W), 0.40f * static_cast<float>(W));
                        ty = std::clamp(ty, -0.30f * static_cast<float>(H), 0.10f * static_cast<float>(H));
                        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1., 0., static_cast<double>(tx), 0., 1., static_cast<double>(ty));
                        cv::Mat sh;
                        cv::warpAffine(obj_mask, sh, M, obj_mask.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
                        cv::Mat shf;
                        sh.convertTo(shf, CV_32F, 1.0 / 255.0);
                        const double bs = std::max(0.8, static_cast<double>(ui.cast_shadow_blur));
                        cv::GaussianBlur(shf, shf, cv::Size(0, 0), bs, bs);
                        const float sa = std::clamp(ui.mr_object_shadow_alpha, 0.f, 1.f);
                        for (int y = 0; y < H; ++y)
                        {
                            cv::Vec3b* op = m_color_mr_rgb.ptr<cv::Vec3b>(y);
                            const float* sp = shf.ptr<float>(y);
                            const uint16_t* dr = m_depth_rgb_u16.ptr<uint16_t>(y);
                            for (int x = 0; x < W; ++x)
                            {
                                if (dr[x] < 250)
                                    continue;
                                const float v = std::clamp(sp[x], 0.f, 1.f);
                                if (v < 1e-3f)
                                    continue;
                                const float dark = 1.f - v * sa;
                                op[x][0] = static_cast<uint8_t>(std::clamp(op[x][0] * dark, 0.f, 255.f));
                                op[x][1] = static_cast<uint8_t>(std::clamp(op[x][1] * dark, 0.f, 255.f));
                                op[x][2] = static_cast<uint8_t>(std::clamp(op[x][2] * dark, 0.f, 255.f));
                            }
                        }
                    }

                    if (ui.mr_reflection_enabled)
                    {
                        const int y_mirror = std::clamp(static_cast<int>(ui.reflection_plane_y * static_cast<float>(H)), 0, H - 1);
                        const float ra = std::clamp(ui.mr_object_reflection_alpha, 0.f, 1.f);
                        for (int y = y_mirror; y < H; ++y)
                        {
                            cv::Vec3b* op = m_color_mr_rgb.ptr<cv::Vec3b>(y);
                            const int ys = std::clamp(2 * y_mirror - y, 0, H - 1);
                            const uint8_t* vm = obj_vis_mask.ptr<uint8_t>(ys);
                            const float yfade = 1.0f - std::clamp(static_cast<float>(y - y_mirror) / std::max(1.0f, rad * 1.8f), 0.0f, 1.0f);
                            for (int x = 0; x < W; ++x)
                            {
                                if (!vm[x])
                                    continue;
                                const float w = ra * yfade;
                                op[x][0] = static_cast<uint8_t>(std::clamp(op[x][0] * (1.f - w) + obj_col[0] * w, 0.f, 255.f));
                                op[x][1] = static_cast<uint8_t>(std::clamp(op[x][1] * (1.f - w) + obj_col[1] * w, 0.f, 255.f));
                                op[x][2] = static_cast<uint8_t>(std::clamp(op[x][2] * (1.f - w) + obj_col[2] * w, 0.f, 255.f));
                            }
                        }
                    }

                    const cv::Point cpt(static_cast<int>(std::round(cx_obj)), static_cast<int>(std::round(cy_obj)));
                    cv::line(m_color_mr_rgb, cpt + cv::Point(-16, 0), cpt + cv::Point(16, 0), cv::Scalar(60, 255, 60), 2, cv::LINE_AA);
                    cv::line(m_color_mr_rgb, cpt + cv::Point(0, -16), cpt + cv::Point(0, 16), cv::Scalar(60, 60, 255), 2, cv::LINE_AA);
                }
            }
            else
            {
                m_depth_rgb_u16.release();
                m_depth_rgb_vis.release();
                m_color_mr_rgb.release();
            }
        }
        else if (!want_rgb_depth)
        {
            m_depth_rgb_u16.release();
            m_depth_rgb_vis.release();
            m_color_mr_rgb.release();
        }

        // Color 창은 RGB 기준으로 동작: raw 모드에서도 사람 바닥 그림자/MR 오브젝트를 허용
        const bool want_color_fx = ui.floor_fx_enabled && ui.show_color;
        const bool want_floor_compute = ui.floor_fx_enabled && (ui.show_floor_panel || want_color_fx) && streaming &&
                                        xform.has_value() && k4a_dep && k4a_col;
        if (!want_floor_compute)
        {
            m_floor.release();
            m_color_fx.release();
            m_color_mr_rgb.release();
            m_floor_alpha.release();
            m_depth_aligned_u16.release();
            m_depth_rgb_u16.release();
            m_depth_rgb_u16_filled.release();
            m_depth_rgb_vis.release();
            floor_stride_counter = 0;
        }
        else
        {
            cv::Mat bgr_align, d16_align;
            if (!rvm_floor::color_depth_aligned_mats(*xform, k4a_dep, k4a_col, bgr_align, d16_align))
            {
                /* 일시적 정합 실패: 이전 Floor 유지 (검은 화면/이중 화면 번쩍임 방지) */
            }
            else
            {
                m_depth_aligned_u16 = d16_align;
                ++floor_stride_counter;
                const int st = std::max(1, ui.floor_update_stride);
                const bool refresh_alpha = ((floor_stride_counter - 1) % st) == 0;

                rvm_floor::Params fp;
                fp.enabled = true;
                fp.shadow_strength = ui.floor_shadow;
                fp.reflection_strength = ui.floor_refl;
                fp.downsample_ratio = ui.floor_ds_ratio;
                fp.floor_grid_show = ui.floor_grid_show;
                fp.floor_grid_spacing = ui.floor_grid_spacing;
                fp.sun_altitude_deg = ui.sun_altitude_deg;
                fp.sun_azimuth_deg = ui.sun_azimuth_deg;
                fp.floor_anchor_valid = ui.floor_anchor_valid;
                fp.floor_anchor_u = ui.floor_anchor_u;
                fp.floor_anchor_v = ui.floor_anchor_v;
                fp.cast_shadow_alpha = ui.cast_shadow_alpha;
                fp.cast_shadow_blur = ui.cast_shadow_blur;
                fp.cam_intrin_valid = have_depth_intrin;
                fp.cam_fx = depth_fx;
                fp.cam_fy = depth_fy;
                fp.cam_cx = depth_cx;
                fp.cam_cy = depth_cy;
                fp.shadow_ray_step = ui.person_segment_depth_only ? 4 : 2;
                // Reflection strength만 올려두고 체크를 끈 경우(또는 예전 INI)에도 반사가 동작하도록
                fp.reflection_on =
                    (ui.reflection_on || (ui.floor_refl > 0.06f)) &&
                    !(ui.reflection_single_pass_only && ui.rvm_reflection_overlay_on);
                fp.reflection_plane_y = ui.reflection_plane_y;
                fp.plane_use_foot_roi = ui.plane_use_foot_roi;
                fp.plane_foot_roi_radius = ui.plane_foot_roi_radius;
                fp.plane_temporal_smooth_on = ui.plane_temporal_smooth_on;
                fp.plane_temporal_alpha = ui.plane_temporal_alpha;

                if (refresh_alpha)
                {
                    cv::Mat alpha;
                    bool used_rvm = false;
                    const bool rvm_preferred = rvm_eng.net_loaded;
                    if (rvm_preferred || !ui.person_segment_depth_only)
                    {
                        // RVM 우선: 사람이미지 분리는 RVM으로 확정하고, 그림자는 해당 영역의 depth로 계산
                        if (rvm_floor::infer_alpha(rvm_eng, bgr_align, alpha, fp.downsample_ratio))
                            used_rvm = true;
                    }
                    if (!used_rvm && ui.person_segment_depth_only)
                    {
                        // RVM이 실제로 없거나 실패한 경우에만 depth-only 폴백
                        rvm_floor::infer_alpha_depth_fallback(bgr_align, d16_align, alpha);
                    }
                    if (!alpha.empty())
                    {
                        m_floor_alpha = alpha;
                        if (used_rvm)
                            matte_mode_line = ui.person_segment_depth_only ? "Matte: RVM mask + depth shadow"
                                                                           : "Matte: RVM ONNX";
                        else
                            matte_mode_line = "Matte: Depth-only fallback";
                    }
                    else
                    {
                        m_floor_alpha.release();
                        matte_mode_line = ui.person_segment_depth_only ? "Matte: unavailable"
                                                                       : "Matte: RVM required (load ONNX)";
                    }
                }

                if (!m_floor_alpha.empty())
                {
                    cv::Mat alpha_use;
                    if (m_floor_alpha.size() != bgr_align.size())
                        cv::resize(m_floor_alpha, alpha_use, bgr_align.size(), 0, 0, cv::INTER_LINEAR);
                    else
                        m_floor_alpha.copyTo(alpha_use);
                    if (!alpha_use.empty())
                    {
                        cv::GaussianBlur(alpha_use, alpha_use, cv::Size(0, 0), 1.25);
                        cv::morphologyEx(alpha_use,
                                         alpha_use,
                                         cv::MORPH_CLOSE,
                                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
                        // 경계 halo 억제: 아주 얇은 알파 테두리를 살짝 안쪽으로 정리
                        cv::Mat a8, a8_er, a_er_f;
                        alpha_use.convertTo(a8, CV_8U, 255.0);
                        cv::erode(a8, a8_er, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
                        a8_er.convertTo(a_er_f, CV_32F, 1.0 / 255.0);
                        alpha_use = alpha_use.mul(a_er_f * 0.75f + 0.25f);
                        for (int y = 0; y < alpha_use.rows; ++y)
                        {
                            const uint16_t* dr = d16_align.ptr<uint16_t>(y);
                            float* ar = alpha_use.ptr<float>(y);
                            for (int x = 0; x < alpha_use.cols; ++x)
                                if (dr[x] == 0)
                                    ar[x] *= 0.30f;
                        }
                        cv::Mat fx = rvm_floor::compose_floor_with_fx(bgr_align, d16_align, alpha_use, fp);
                        if (want_color_fx)
                        {
                            if (ui.color_view_mode == 1)
                            {
                                // 사람 매트 품질 확인용: 색상 기준 사람만 추출(배경은 흑색)
                                m_color_fx.create(bgr_align.size(), CV_8UC3);
                                for (int y = 0; y < m_color_fx.rows; ++y)
                                {
                                    const cv::Vec3b* cp = bgr_align.ptr<cv::Vec3b>(y);
                                    const float* ap = alpha_use.ptr<float>(y);
                                    cv::Vec3b* op = m_color_fx.ptr<cv::Vec3b>(y);
                                    for (int x = 0; x < m_color_fx.cols; ++x)
                                    {
                                        const float al = std::clamp(ap[x], 0.f, 1.f);
                                        op[x][0] = static_cast<uint8_t>(cp[x][0] * al);
                                        op[x][1] = static_cast<uint8_t>(cp[x][1] * al);
                                        op[x][2] = static_cast<uint8_t>(cp[x][2] * al);
                                    }
                                }
                            }
                            else
                            {
                                rvm_floor::Params cp = fp;
                                cp.reflection_on = false;
                                cp.reflection_strength = 0.0f;
                                cp.floor_grid_show = false;
                                m_color_fx = rvm_floor::compose_floor_with_fx(bgr_align, d16_align, alpha_use, cp);
                                if (!m_color_fx.empty() && (!ui.rvm_floor_overlay_on || !ui.rvm_shadow_overlay_on))
                                {
                                    cv::Mat dark_raw(bgr_align.size(), CV_32F, cv::Scalar(0));
                                    for (int y = 0; y < dark_raw.rows; ++y)
                                    {
                                        const cv::Vec3b* sp = bgr_align.ptr<cv::Vec3b>(y);
                                        const cv::Vec3b* hp = m_color_fx.ptr<cv::Vec3b>(y);
                                        float* dp = dark_raw.ptr<float>(y);
                                        for (int x = 0; x < dark_raw.cols; ++x)
                                        {
                                            const float l0 = (0.114f * sp[x][0] + 0.587f * sp[x][1] + 0.299f * sp[x][2]) + 1.0f;
                                            const float l1 = (0.114f * hp[x][0] + 0.587f * hp[x][1] + 0.299f * hp[x][2]) + 1.0f;
                                            dp[x] = std::clamp((l0 - l1) / std::max(8.0f, l0), 0.0f, 1.0f);
                                        }
                                    }
                                    cv::Mat dark_floor, dark_soft, dark_bin;
                                    cv::GaussianBlur(dark_raw, dark_floor, cv::Size(0, 0), std::max(10.0, (double)ui.cast_shadow_blur * 2.0));
                                    cv::GaussianBlur(dark_raw, dark_soft, cv::Size(0, 0), std::max(1.2, (double)ui.cast_shadow_blur * 0.85));
                                    cv::threshold(dark_soft, dark_bin, 0.045, 1.0, cv::THRESH_BINARY);
                                    cv::dilate(dark_bin, dark_bin,
                                               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
                                    cv::GaussianBlur(dark_bin, dark_bin, cv::Size(0, 0),
                                                     std::max(0.8, (double)ui.cast_shadow_blur * 0.7));

                                    cv::Mat cast_w =
                                        rvm_floor::compute_cast_shadow_weight(bgr_align, d16_align, alpha_use, cp);
                                    const bool use_3d_person_shadow = !cast_w.empty();
                                    cv::Mat sh_roi_f;
                                    if (!use_3d_person_shadow)
                                    {
                                        cv::Mat person_u8;
                                        cv::threshold(alpha_use, person_u8, 0.20, 255.0, cv::THRESH_BINARY);
                                        person_u8.convertTo(person_u8, CV_8U);
                                        const float alt_rad =
                                            std::clamp(ui.sun_altitude_deg, 6.0f, 88.0f) * 3.14159265f / 180.0f;
                                        const float azi_rad = ui.sun_azimuth_deg * 3.14159265f / 180.0f;
                                        float stretch = 52.f / std::tan(std::max(0.12f, alt_rad));
                                        stretch = std::clamp(stretch, 6.f, 240.f);
                                        const float tx = -std::cos(azi_rad) * stretch;
                                        const float ty = -std::sin(azi_rad) * stretch * 0.42f;
                                        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1., 0., static_cast<double>(tx), 0.,
                                                           1., static_cast<double>(ty));
                                        cv::Mat sh_roi_u8;
                                        cv::warpAffine(person_u8, sh_roi_u8, M, person_u8.size(), cv::INTER_LINEAR,
                                                       cv::BORDER_CONSTANT, cv::Scalar(0));
                                        cv::dilate(sh_roi_u8, sh_roi_u8,
                                                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));
                                        sh_roi_u8.convertTo(sh_roi_f, CV_32F, 1.0 / 255.0);
                                        cv::GaussianBlur(sh_roi_f, sh_roi_f, cv::Size(0, 0),
                                                         std::max(1.0, (double)ui.cast_shadow_blur));
                                    }

                                    for (int y = 0; y < m_color_fx.rows; ++y)
                                    {
                                        const cv::Vec3b* sp = bgr_align.ptr<cv::Vec3b>(y);
                                        const float* fp = dark_floor.ptr<float>(y);
                                        const float* ds = dark_soft.ptr<float>(y);
                                        const float* db = dark_bin.ptr<float>(y);
                                        const float* cwp =
                                            use_3d_person_shadow ? cast_w.ptr<float>(y) : sh_roi_f.ptr<float>(y);
                                        cv::Vec3b* op = m_color_fx.ptr<cv::Vec3b>(y);
                                        for (int x = 0; x < m_color_fx.cols; ++x)
                                        {
                                            const float rp = cwp[x];
                                            const float floor_w =
                                                ui.rvm_floor_overlay_on ? std::clamp(fp[x] * 0.45f, 0.0f, 1.0f) : 0.0f;
                                            const float sh_unified =
                                                std::clamp(0.35f * ds[x] + 0.95f * db[x], 0.0f, 1.0f);
                                            const float shadow_w =
                                                ui.rvm_shadow_overlay_on
                                                    ? (use_3d_person_shadow
                                                           ? std::clamp(rp * ui.cast_shadow_alpha * 1.25f, 0.0f, 1.0f)
                                                           : std::clamp(sh_unified * rp * ui.cast_shadow_alpha * 1.25f,
                                                                        0.0f, 1.0f))
                                                    : 0.0f;
                                            const float dark_w = 1.0f - (1.0f - floor_w) * (1.0f - shadow_w);
                                            op[x][0] = static_cast<uint8_t>(std::clamp(sp[x][0] * (1.0f - dark_w), 0.0f, 255.0f));
                                            op[x][1] = static_cast<uint8_t>(std::clamp(sp[x][1] * (1.0f - dark_w), 0.0f, 255.0f));
                                            op[x][2] = static_cast<uint8_t>(std::clamp(sp[x][2] * (1.0f - dark_w), 0.0f, 255.0f));
                                        }
                                    }
                                }
                                if (ui.rvm_reflection_overlay_on && have_depth_intrin && !m_color_fx.empty())
                                    rvm_floor::apply_depth_mirror_person_reflection(m_color_fx,
                                                                                  bgr_align,
                                                                                  d16_align,
                                                                                  alpha_use,
                                                                                  cp,
                                                                                  ui.floor_refl);
                                if (ui.mr_virtual_object_enabled && !m_color_fx.empty())
                                {
                                    // Stage 3/4/6/7 bridge: object + occlusion + object shadow + reflection + gizmo
                                    float cx_obj =
                                        std::clamp(ui.mr_object_u, 0.f, 1.f) * static_cast<float>(m_color_fx.cols - 1);
                                    float cy_obj =
                                        std::clamp(ui.mr_object_v, 0.f, 1.f) * static_cast<float>(m_color_fx.rows - 1);
                                    float rad = std::max(4.f, ui.mr_object_radius_px);
                                    const float inv_r2 = 1.0f / std::max(1.0f, rad * rad);
                                    float obj_depth = std::clamp(ui.mr_object_depth_mm, 500.f, 8000.f);
                                    if (ui.mr_use_world_transform && have_depth_intrin)
                                    {
                                        const float tz = std::max(0.3f, ui.mr_obj_tz_m);
                                        const float uproj = depth_fx * (ui.mr_obj_tx_m / tz) + depth_cx;
                                        const float vproj = depth_fy * (ui.mr_obj_ty_m / tz) + depth_cy;
                                        cx_obj = std::clamp(uproj, 0.0f, static_cast<float>(m_color_fx.cols - 1));
                                        cy_obj = std::clamp(vproj, 0.0f, static_cast<float>(m_color_fx.rows - 1));
                                        obj_depth = std::clamp(tz * 1000.0f, 500.0f, 8000.0f);
                                        const float depth_scale = std::clamp(2.3f / tz, 0.35f, 4.0f);
                                        rad = std::max(4.0f, ui.mr_object_radius_px * ui.mr_obj_scale * depth_scale);
                                    }
                                    const float inv_r2_use = 1.0f / std::max(1.0f, rad * rad);
                                    const cv::Vec3f obj_col(80.f, 220.f, 255.f); // BGR
                                    const int x0 = std::max(0, static_cast<int>(std::floor(cx_obj - rad - 1.f)));
                                    const int x1 =
                                        std::min(m_color_fx.cols - 1, static_cast<int>(std::ceil(cx_obj + rad + 1.f)));
                                    const int y0 = std::max(0, static_cast<int>(std::floor(cy_obj - rad - 1.f)));
                                    const int y1 =
                                        std::min(m_color_fx.rows - 1, static_cast<int>(std::ceil(cy_obj + rad + 1.f)));

                                    cv::Mat obj_mask(m_color_fx.rows, m_color_fx.cols, CV_8U, cv::Scalar(0));
                                    cv::Mat obj_vis_mask(m_color_fx.rows, m_color_fx.cols, CV_8U, cv::Scalar(0));

                                    for (int y = y0; y <= y1; ++y)
                                    {
                                        cv::Vec3b* op = m_color_fx.ptr<cv::Vec3b>(y);
                                        uint8_t* om = obj_mask.ptr<uint8_t>(y);
                                        uint8_t* vm = obj_vis_mask.ptr<uint8_t>(y);
                                        const uint16_t* dr = d16_align.ptr<uint16_t>(y);
                                        for (int x = x0; x <= x1; ++x)
                                        {
                                            const float dx = static_cast<float>(x) - cx_obj;
                                            const float dy = static_cast<float>(y) - cy_obj;
                                            const float ax = std::abs(dx);
                                            const float ay = std::abs(dy);
                                            float shape_w = 0.0f;
                                            const float th = ui.mr_obj_rot_deg * 3.14159265f / 180.0f;
                                            const float cs = std::cos(th);
                                            const float sn = std::sin(th);
                                            const float rx = dx * cs + dy * sn;
                                            const float ry = -dx * sn + dy * cs;
                                            const float axr = std::abs(rx);
                                            const float ayr = std::abs(ry);
                                            if (ui.mr_object_shape == 0)
                                            {
                                                const float r2n = (dx * dx + dy * dy) * inv_r2_use;
                                                if (r2n > 1.0f)
                                                    continue;
                                                shape_w = std::clamp(1.0f - r2n, 0.0f, 1.0f);
                                            }
                                            else
                                            {
                                                const float boxd = std::max(axr, ayr) / std::max(1.0f, rad);
                                                if (boxd > 1.0f)
                                                    continue;
                                                shape_w = std::clamp(1.0f - boxd, 0.0f, 1.0f);
                                            }

                                            om[x] = 255;
                                            const float local_depth = obj_depth - shape_w * 130.0f;
                                            const uint16_t rz = dr[x];
                                            const bool occluded =
                                                ui.mr_occlusion_enabled && (rz > 250) && (static_cast<float>(rz) < local_depth);
                                            if (occluded)
                                                continue;
                                            vm[x] = 255;
                                            const float w = 0.20f + 0.65f * std::sqrt(shape_w);
                                            op[x][0] =
                                                static_cast<uint8_t>(std::clamp(op[x][0] * (1.f - w) + obj_col[0] * w, 0.f, 255.f));
                                            op[x][1] =
                                                static_cast<uint8_t>(std::clamp(op[x][1] * (1.f - w) + obj_col[1] * w, 0.f, 255.f));
                                            op[x][2] =
                                                static_cast<uint8_t>(std::clamp(op[x][2] * (1.f - w) + obj_col[2] * w, 0.f, 255.f));
                                        }
                                    }

                                    if (ui.mr_shadow_enabled)
                                    {
                                        float alt_rad = std::clamp(ui.sun_altitude_deg, 6.0f, 88.0f) * 3.14159265f / 180.0f;
                                        const float azi_rad = ui.sun_azimuth_deg * 3.14159265f / 180.0f;
                                        float stretch = 48.f / std::tan(std::max(0.12f, alt_rad));
                                        stretch = std::clamp(stretch, 6.f, 220.f);
                                        float tx = -std::cos(azi_rad) * stretch;
                                        float ty = -std::sin(azi_rad) * stretch * 0.35f;
                                        tx = std::clamp(tx, -0.40f * static_cast<float>(m_color_fx.cols),
                                                        0.40f * static_cast<float>(m_color_fx.cols));
                                        ty = std::clamp(ty, -0.30f * static_cast<float>(m_color_fx.rows),
                                                        0.10f * static_cast<float>(m_color_fx.rows));
                                        const cv::Mat M = (cv::Mat_<double>(2, 3) << 1., 0., static_cast<double>(tx), 0., 1.,
                                                           static_cast<double>(ty));
                                        cv::Mat sh;
                                        cv::warpAffine(obj_mask, sh, M, obj_mask.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                                       cv::Scalar(0));
                                        cv::Mat shf;
                                        sh.convertTo(shf, CV_32F, 1.0 / 255.0);
                                        const double bs = std::max(0.8, static_cast<double>(ui.cast_shadow_blur));
                                        cv::GaussianBlur(shf, shf, cv::Size(0, 0), bs, bs);
                                        const float sa = std::clamp(ui.mr_object_shadow_alpha, 0.f, 1.f);
                                        for (int y = 0; y < m_color_fx.rows; ++y)
                                        {
                                            cv::Vec3b* op = m_color_fx.ptr<cv::Vec3b>(y);
                                            const float* sp = shf.ptr<float>(y);
                                            const uint16_t* dr = d16_align.ptr<uint16_t>(y);
                                            for (int x = 0; x < m_color_fx.cols; ++x)
                                            {
                                                if (dr[x] < 250)
                                                    continue;
                                                const float v = std::clamp(sp[x], 0.f, 1.f);
                                                if (v < 1e-3f)
                                                    continue;
                                                const float dark = 1.f - v * sa;
                                                op[x][0] = static_cast<uint8_t>(std::clamp(op[x][0] * dark, 0.f, 255.f));
                                                op[x][1] = static_cast<uint8_t>(std::clamp(op[x][1] * dark, 0.f, 255.f));
                                                op[x][2] = static_cast<uint8_t>(std::clamp(op[x][2] * dark, 0.f, 255.f));
                                            }
                                        }
                                    }

                                    if (ui.mr_reflection_enabled)
                                    {
                                        const int y_mirror = std::clamp(static_cast<int>(
                                                                            ui.reflection_plane_y *
                                                                            static_cast<float>(m_color_fx.rows)),
                                                                        0, m_color_fx.rows - 1);
                                        const float ra = std::clamp(ui.mr_object_reflection_alpha, 0.f, 1.f);
                                        for (int y = y_mirror; y < m_color_fx.rows; ++y)
                                        {
                                            cv::Vec3b* op = m_color_fx.ptr<cv::Vec3b>(y);
                                            const int ys = std::clamp(2 * y_mirror - y, 0, m_color_fx.rows - 1);
                                            const uint8_t* vm = obj_vis_mask.ptr<uint8_t>(ys);
                                            const float yfade = 1.0f - std::clamp(
                                                                          static_cast<float>(y - y_mirror) /
                                                                              std::max(1.0f, rad * 1.8f),
                                                                          0.0f, 1.0f);
                                            for (int x = 0; x < m_color_fx.cols; ++x)
                                            {
                                                if (!vm[x])
                                                    continue;
                                                const float w = ra * yfade;
                                                op[x][0] = static_cast<uint8_t>(
                                                    std::clamp(op[x][0] * (1.f - w) + obj_col[0] * w, 0.f, 255.f));
                                                op[x][1] = static_cast<uint8_t>(
                                                    std::clamp(op[x][1] * (1.f - w) + obj_col[1] * w, 0.f, 255.f));
                                                op[x][2] = static_cast<uint8_t>(
                                                    std::clamp(op[x][2] * (1.f - w) + obj_col[2] * w, 0.f, 255.f));
                                            }
                                        }
                                    }

                                    // 간단 gizmo 표시 (축 십자선)
                                    const cv::Point cpt(static_cast<int>(std::round(cx_obj)), static_cast<int>(std::round(cy_obj)));
                                    cv::line(m_color_fx, cpt + cv::Point(-16, 0), cpt + cv::Point(16, 0), cv::Scalar(60, 255, 60),
                                             2, cv::LINE_AA);
                                    cv::line(m_color_fx, cpt + cv::Point(0, -16), cpt + cv::Point(0, 16), cv::Scalar(60, 60, 255),
                                             2, cv::LINE_AA);
                                }
                                if (ui.mr_occlusion_enabled && ui.mr_occlusion_debug && !m_color_fx.empty())
                                {
                                    // Stage 4 디버그: real_depth < virtual_depth 조건을 Color 패널에 오버레이.
                                    // 전체 프레임에 적용하면 + 큰 Gaussian이 화면 대부분을 붉게 덮음 → ROI 옵션 추가.
                                    float occ_cx =
                                        std::clamp(ui.mr_object_u, 0.f, 1.f) * static_cast<float>(m_color_fx.cols - 1);
                                    float occ_cy =
                                        std::clamp(ui.mr_object_v, 0.f, 1.f) * static_cast<float>(m_color_fx.rows - 1);
                                    float occ_rad = std::max(4.f, ui.mr_object_radius_px);
                                    if (ui.mr_use_world_transform && have_depth_intrin)
                                    {
                                        const float tz = std::max(0.3f, ui.mr_obj_tz_m);
                                        const float uproj = depth_fx * (ui.mr_obj_tx_m / tz) + depth_cx;
                                        const float vproj = depth_fy * (ui.mr_obj_ty_m / tz) + depth_cy;
                                        occ_cx = std::clamp(uproj, 0.0f, static_cast<float>(m_color_fx.cols - 1));
                                        occ_cy = std::clamp(vproj, 0.0f, static_cast<float>(m_color_fx.rows - 1));
                                        const float depth_scale = std::clamp(2.3f / tz, 0.35f, 4.0f);
                                        occ_rad = std::max(4.0f, ui.mr_object_radius_px * ui.mr_obj_scale * depth_scale);
                                    }
                                    const float occ_roi_r = occ_rad * std::max(1.0f, ui.mr_occlusion_debug_roi_mul);
                                    const float occ_roi_r2 = occ_roi_r * occ_roi_r;

                                    cv::Mat occ_mask(m_color_fx.rows, m_color_fx.cols, CV_32F, cv::Scalar(0));
                                    const float vdepth = std::clamp(ui.mr_virtual_depth_mm, 500.f, 8000.f);
                                    for (int y = 0; y < occ_mask.rows; ++y)
                                    {
                                        const uint16_t* dr = d16_align.ptr<uint16_t>(y);
                                        float* mp = occ_mask.ptr<float>(y);
                                        for (int x = 0; x < occ_mask.cols; ++x)
                                        {
                                            if (ui.mr_occlusion_debug_local_only)
                                            {
                                                const float dx = static_cast<float>(x) - occ_cx;
                                                const float dy = static_cast<float>(y) - occ_cy;
                                                if (dx * dx + dy * dy > occ_roi_r2)
                                                    continue;
                                            }
                                            const uint16_t z = dr[x];
                                            if (z > 250 && z < static_cast<uint16_t>(vdepth))
                                                mp[x] = 1.0f;
                                        }
                                    }
                                    double sig = std::max(0.0, static_cast<double>(ui.mr_depth_edge_smooth));
                                    if (ui.mr_occlusion_debug_local_only && sig > 1e-6)
                                        sig = std::min(sig, static_cast<double>(occ_roi_r) * 0.22);
                                    if (sig > 1e-6)
                                        cv::GaussianBlur(occ_mask, occ_mask, cv::Size(0, 0), sig, sig);
                                    const float oa = std::clamp(ui.mr_occlusion_overlay_alpha, 0.f, 1.f);
                                    for (int y = 0; y < m_color_fx.rows; ++y)
                                    {
                                        cv::Vec3b* op = m_color_fx.ptr<cv::Vec3b>(y);
                                        const float* mp = occ_mask.ptr<float>(y);
                                        for (int x = 0; x < m_color_fx.cols; ++x)
                                        {
                                            const float w = std::clamp(mp[x] * oa, 0.f, 1.f);
                                            if (w <= 1e-4f)
                                                continue;
                                            // BGR = (40, 80, 255): 앞쪽 실깊이(occluder)를 따뜻한 색으로 표시
                                            op[x][0] = static_cast<uint8_t>(std::clamp(op[x][0] * (1.f - w) + 40.f * w, 0.f, 255.f));
                                            op[x][1] = static_cast<uint8_t>(std::clamp(op[x][1] * (1.f - w) + 80.f * w, 0.f, 255.f));
                                            op[x][2] = static_cast<uint8_t>(std::clamp(op[x][2] * (1.f - w) + 255.f * w, 0.f, 255.f));
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            m_color_fx.release();
                        }
                        if (ui.person_segment_depth_only)
                        {
                            // Depth-only 모드: Floor 합성은 "녹색 배경 + 사람 + 사람 그림자"만 표시
                            const cv::Vec3b kGreenBg(30, 150, 30); // BGR
                            // 1) 사람 마스크 정제: 연결 성분 중 인체 하나만 선택
                            cv::Mat person_bin_u8;
                            cv::threshold(alpha_use, person_bin_u8, 0.18, 255.0, cv::THRESH_BINARY);
                            person_bin_u8.convertTo(person_bin_u8, CV_8U);
                            cv::morphologyEx(person_bin_u8,
                                             person_bin_u8,
                                             cv::MORPH_CLOSE,
                                             cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
                            cv::Mat labels, stats, cents;
                            const int nlabels = cv::connectedComponentsWithStats(person_bin_u8, labels, stats, cents, 8, CV_32S);
                            cv::Mat keep(person_bin_u8.size(), CV_8U, cv::Scalar(0));
                            if (nlabels > 1)
                            {
                                int best = 0;
                                float best_score = -1e9f;
                                const int Hm = person_bin_u8.rows;
                                const int Wm = person_bin_u8.cols;
                                const int y_seed0 = static_cast<int>(Hm * 0.60f);
                                const int y_seed1 = static_cast<int>(Hm * 0.95f);
                                const int min_area = std::max(180, (Hm * Wm) / 2500);
                                const int max_area = std::max(min_area + 1, (Hm * Wm) / 5); // 벽/대면적 누설 배제
                                for (int id = 1; id < nlabels; ++id)
                                {
                                    const int area = stats.at<int>(id, cv::CC_STAT_AREA);
                                    if (area < min_area || area > max_area)
                                        continue;
                                    const int x = stats.at<int>(id, cv::CC_STAT_LEFT);
                                    const int y = stats.at<int>(id, cv::CC_STAT_TOP);
                                    const int w = stats.at<int>(id, cv::CC_STAT_WIDTH);
                                    const int h = stats.at<int>(id, cv::CC_STAT_HEIGHT);
                                    const float aspect = static_cast<float>(h) / std::max(1.0f, static_cast<float>(w));
                                    if (aspect < 0.85f)
                                        continue; // 사람/더미는 대체로 세로형

                                    int seed_hits = 0;
                                    for (int yy = y_seed0; yy < y_seed1; ++yy)
                                    {
                                        const int* lp = labels.ptr<int>(yy);
                                        for (int xx = 0; xx < Wm; ++xx)
                                            if (lp[xx] == id)
                                                ++seed_hits;
                                    }
                                    if (seed_hits < 8)
                                        continue;

                                    const float cxm = static_cast<float>(x + w * 0.5f);
                                    const float center_pen = std::abs(cxm - 0.5f * Wm) / static_cast<float>(Wm);
                                    const float area_norm = static_cast<float>(area) / static_cast<float>(Hm * Wm);
                                    const float score =
                                        seed_hits * 0.45f + aspect * 14.0f - center_pen * 12.0f - area_norm * 120.0f;
                                    if (score > best_score)
                                    {
                                        best_score = score;
                                        best = id;
                                    }
                                }
                                if (best <= 0)
                                {
                                    int best_area = 0;
                                    for (int id = 1; id < nlabels; ++id)
                                    {
                                        const int area = stats.at<int>(id, cv::CC_STAT_AREA);
                                        if (area > best_area)
                                        {
                                            best_area = area;
                                            best = id;
                                        }
                                    }
                                }
                                for (int y = 0; y < keep.rows; ++y)
                                {
                                    const int* lp = labels.ptr<int>(y);
                                    uint8_t* kp = keep.ptr<uint8_t>(y);
                                    for (int x = 0; x < keep.cols; ++x)
                                        if (lp[x] == best)
                                            kp[x] = 255;
                                }
                                // 상단 하드 컷(고정 16%)은 머리 절단을 유발할 수 있어 제거하고,
                                // "상단에 붙은 넓은 벽 성분"일 때만 완만 컷을 적용한다.
                                if (best > 0)
                                {
                                    const int top = stats.at<int>(best, cv::CC_STAT_TOP);
                                    const int bw = stats.at<int>(best, cv::CC_STAT_WIDTH);
                                    const int bh = stats.at<int>(best, cv::CC_STAT_HEIGHT);
                                    const float wall_like = static_cast<float>(bw) / std::max(1.0f, static_cast<float>(bh));
                                    if (top < static_cast<int>(Hm * 0.05f) && bw > static_cast<int>(Wm * 0.58f) &&
                                        wall_like > 1.05f)
                                    {
                                        const int cut = static_cast<int>(Hm * 0.10f);
                                        for (int y = 0; y < cut; ++y)
                                            keep.row(y).setTo(0);
                                    }
                                }
                                cv::dilate(keep, keep, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
                            }
                            else
                            {
                                keep = person_bin_u8;
                            }

                            // 2) 그림자 ROI: 인체 마스크를 태양 방향으로 이동한 영역에서만 적용
                            float alt_rad = std::clamp(ui.sun_altitude_deg, 6.0f, 88.0f) * 3.14159265f / 180.0f;
                            const float azi_rad = ui.sun_azimuth_deg * 3.14159265f / 180.0f;
                            float stretch = 52.f / std::tan(std::max(0.12f, alt_rad));
                            stretch = std::clamp(stretch, 6.f, 240.f);
                            float tx = -std::cos(azi_rad) * stretch;
                            float ty = -std::sin(azi_rad) * stretch * 0.42f;
                            const cv::Mat M = (cv::Mat_<double>(2, 3) << 1., 0., static_cast<double>(tx), 0., 1.,
                                               static_cast<double>(ty));
                            cv::Mat sh_roi_u8;
                            cv::warpAffine(keep, sh_roi_u8, M, keep.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                           cv::Scalar(0));
                            cv::dilate(sh_roi_u8, sh_roi_u8, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));

                            m_floor.create(bgr_align.size(), CV_8UC3);
                            for (int y = 0; y < m_floor.rows; ++y)
                            {
                                const cv::Vec3b* cp = bgr_align.ptr<cv::Vec3b>(y);
                                const cv::Vec3b* fpix = fx.ptr<cv::Vec3b>(y);
                                const float* ap = alpha_use.ptr<float>(y);
                                const uint8_t* kp = keep.ptr<uint8_t>(y);
                                const uint8_t* rp = sh_roi_u8.ptr<uint8_t>(y);
                                cv::Vec3b* op = m_floor.ptr<cv::Vec3b>(y);
                                for (int x = 0; x < m_floor.cols; ++x)
                                {
                                    const float al = kp[x] ? std::clamp(ap[x], 0.f, 1.f) : 0.0f;
                                    // 배경에서 그림자 강도 추정: fx가 원본보다 어두우면 shadow로 간주
                                    const float src_l =
                                        (0.114f * cp[x][0] + 0.587f * cp[x][1] + 0.299f * cp[x][2]) + 1.0f;
                                    const float fx_l =
                                        (0.114f * fpix[x][0] + 0.587f * fpix[x][1] + 0.299f * fpix[x][2]) + 1.0f;
                                    const float shadow_amount =
                                        std::clamp((src_l - fx_l) / std::max(8.0f, src_l), 0.0f, 1.0f);
                                    const float roi_w = (rp[x] > 0 && y > static_cast<int>(m_floor.rows * 0.42f)) ? 1.0f : 0.0f;
                                    // 검정 그림자를 명확히 보이게: 녹색 -> 검정으로 보간
                                    const float shw = std::clamp(shadow_amount * 2.2f * roi_w, 0.0f, 1.0f);
                                    cv::Vec3b bg_shadowed;
                                    bg_shadowed[0] = static_cast<uint8_t>(std::clamp(kGreenBg[0] * (1.0f - shw), 0.0f, 255.0f));
                                    bg_shadowed[1] = static_cast<uint8_t>(std::clamp(kGreenBg[1] * (1.0f - shw), 0.0f, 255.0f));
                                    bg_shadowed[2] = static_cast<uint8_t>(std::clamp(kGreenBg[2] * (1.0f - shw), 0.0f, 255.0f));
                                    // 사람은 원본 컬러 유지(알파 문턱 완화), 배경은 녹색+검정그림자
                                    const float person_w = std::clamp((al - 0.06f) / 0.32f, 0.0f, 1.0f);
                                    op[x][0] = static_cast<uint8_t>(cp[x][0] * person_w + bg_shadowed[0] * (1.f - person_w));
                                    op[x][1] = static_cast<uint8_t>(cp[x][1] * person_w + bg_shadowed[1] * (1.f - person_w));
                                    op[x][2] = static_cast<uint8_t>(cp[x][2] * person_w + bg_shadowed[2] * (1.f - person_w));
                                }
                            }
                        }
                        else if (ui.rvm_bg_mode == 2)
                        {
                            // RVM 모드에서도 벽 누설을 막기 위해: "사람 matte + 고정 배경 + 그림자"로 구성
                            const cv::Vec3b kGreenBg(30, 150, 30); // BGR
                            cv::Mat person_bin_u8;
                            cv::threshold(alpha_use, person_bin_u8, 0.22, 255.0, cv::THRESH_BINARY);
                            person_bin_u8.convertTo(person_bin_u8, CV_8U);
                            cv::morphologyEx(person_bin_u8,
                                             person_bin_u8,
                                             cv::MORPH_CLOSE,
                                             cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

                            // 단일 인체 컴포넌트만 선택(벽/대면적 누설 제거)
                            cv::Mat labels, stats, cents;
                            const int nlabels = cv::connectedComponentsWithStats(person_bin_u8, labels, stats, cents, 8, CV_32S);
                            cv::Mat keep(person_bin_u8.size(), CV_8U, cv::Scalar(0));
                            if (nlabels > 1)
                            {
                                int best = 0;
                                float best_score = -1e9f;
                                const int Hm = person_bin_u8.rows;
                                const int Wm = person_bin_u8.cols;
                                const int y_seed0 = static_cast<int>(Hm * 0.60f);
                                const int y_seed1 = static_cast<int>(Hm * 0.95f);
                                const int min_area = std::max(180, (Hm * Wm) / 2500);
                                const int max_area = std::max(min_area + 1, (Hm * Wm) / 5);
                                for (int id = 1; id < nlabels; ++id)
                                {
                                    const int area = stats.at<int>(id, cv::CC_STAT_AREA);
                                    if (area < min_area || area > max_area)
                                        continue;
                                    const int x = stats.at<int>(id, cv::CC_STAT_LEFT);
                                    const int y = stats.at<int>(id, cv::CC_STAT_TOP);
                                    const int w = stats.at<int>(id, cv::CC_STAT_WIDTH);
                                    const int h = stats.at<int>(id, cv::CC_STAT_HEIGHT);
                                    const float aspect = static_cast<float>(h) / std::max(1.0f, static_cast<float>(w));
                                    if (aspect < 0.85f)
                                        continue;
                                    int seed_hits = 0;
                                    for (int yy = y_seed0; yy < y_seed1; ++yy)
                                    {
                                        const int* lp = labels.ptr<int>(yy);
                                        for (int xx = 0; xx < Wm; ++xx)
                                            if (lp[xx] == id)
                                                ++seed_hits;
                                    }
                                    if (seed_hits < 8)
                                        continue;
                                    const float cxm = static_cast<float>(x + w * 0.5f);
                                    const float center_pen = std::abs(cxm - 0.5f * Wm) / static_cast<float>(Wm);
                                    const float area_norm = static_cast<float>(area) / static_cast<float>(Hm * Wm);
                                    const float score =
                                        seed_hits * 0.45f + aspect * 14.0f - center_pen * 12.0f - area_norm * 120.0f;
                                    if (score > best_score)
                                    {
                                        best_score = score;
                                        best = id;
                                    }
                                }
                                if (best <= 0)
                                {
                                    int best_area = 0;
                                    for (int id = 1; id < nlabels; ++id)
                                    {
                                        const int area = stats.at<int>(id, cv::CC_STAT_AREA);
                                        if (area > best_area)
                                        {
                                            best_area = area;
                                            best = id;
                                        }
                                    }
                                }
                                for (int y = 0; y < keep.rows; ++y)
                                {
                                    const int* lp = labels.ptr<int>(y);
                                    uint8_t* kp = keep.ptr<uint8_t>(y);
                                    for (int x = 0; x < keep.cols; ++x)
                                        if (lp[x] == best)
                                            kp[x] = 255;
                                }
                                if (best > 0)
                                {
                                    const int top = stats.at<int>(best, cv::CC_STAT_TOP);
                                    const int bw = stats.at<int>(best, cv::CC_STAT_WIDTH);
                                    const int bh = stats.at<int>(best, cv::CC_STAT_HEIGHT);
                                    const float wall_like = static_cast<float>(bw) / std::max(1.0f, static_cast<float>(bh));
                                    if (top < static_cast<int>(Hm * 0.05f) && bw > static_cast<int>(Wm * 0.58f) &&
                                        wall_like > 1.05f)
                                    {
                                        const int cut = static_cast<int>(Hm * 0.10f);
                                        for (int y = 0; y < cut; ++y)
                                            keep.row(y).setTo(0);
                                    }
                                }
                                cv::dilate(keep, keep, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
                            }
                            else
                            {
                                keep = person_bin_u8;
                            }

                            // RVM 그림자는 depth-shadow(fx darkening) 위치를 그대로 쓰되,
                            // 부위별 얼룩을 줄이기 위해 소프트/균일 마스크로 다시 정리한다.
                            cv::Mat sh_raw(alpha_use.size(), CV_32F, cv::Scalar(0));
                            for (int y = 0; y < sh_raw.rows; ++y)
                            {
                                const cv::Vec3b* cp = bgr_align.ptr<cv::Vec3b>(y);
                                const cv::Vec3b* fpix = fx.ptr<cv::Vec3b>(y);
                                float* sp = sh_raw.ptr<float>(y);
                                for (int x = 0; x < sh_raw.cols; ++x)
                                {
                                    const float src_l = (0.114f * cp[x][0] + 0.587f * cp[x][1] + 0.299f * cp[x][2]) + 1.0f;
                                    const float fx_l = (0.114f * fpix[x][0] + 0.587f * fpix[x][1] + 0.299f * fpix[x][2]) + 1.0f;
                                    sp[x] = std::clamp((src_l - fx_l) / std::max(8.0f, src_l), 0.0f, 1.0f);
                                }
                            }

                            cv::Mat sh_floor_lf, sh_soft;
                            cv::GaussianBlur(sh_raw, sh_floor_lf, cv::Size(0, 0), std::max(10.0, (double)ui.cast_shadow_blur * 2.0));
                            cv::GaussianBlur(sh_raw, sh_soft, cv::Size(0, 0), std::max(1.2, (double)ui.cast_shadow_blur * 0.85));
                            cv::Mat sh_bin;
                            cv::threshold(sh_soft, sh_bin, 0.045, 1.0, cv::THRESH_BINARY);
                            cv::dilate(sh_bin, sh_bin, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
                            cv::GaussianBlur(sh_bin, sh_bin, cv::Size(0, 0), std::max(0.8, (double)ui.cast_shadow_blur * 0.7));
                            cv::Mat cast_w =
                                rvm_floor::compute_cast_shadow_weight(bgr_align, d16_align, alpha_use, fp);
                            const bool use_3d_person_shadow = !cast_w.empty();
                            cv::Mat sh_roi_f;
                            if (!use_3d_person_shadow)
                            {
                                float alt_rad = std::clamp(ui.sun_altitude_deg, 6.0f, 88.0f) * 3.14159265f / 180.0f;
                                const float azi_rad = ui.sun_azimuth_deg * 3.14159265f / 180.0f;
                                float stretch = 52.f / std::tan(std::max(0.12f, alt_rad));
                                stretch = std::clamp(stretch, 6.f, 240.f);
                                const float tx = -std::cos(azi_rad) * stretch;
                                const float ty = -std::sin(azi_rad) * stretch * 0.42f;
                                const cv::Mat M = (cv::Mat_<double>(2, 3) << 1., 0., static_cast<double>(tx), 0., 1.,
                                                   static_cast<double>(ty));
                                cv::Mat sh_roi_u8;
                                cv::warpAffine(keep, sh_roi_u8, M, keep.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                               cv::Scalar(0));
                                cv::dilate(sh_roi_u8, sh_roi_u8,
                                           cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));
                                sh_roi_u8.convertTo(sh_roi_f, CV_32F, 1.0 / 255.0);
                                cv::GaussianBlur(sh_roi_f, sh_roi_f, cv::Size(0, 0),
                                                 std::max(1.0, (double)ui.cast_shadow_blur));
                            }

                            m_floor.create(bgr_align.size(), CV_8UC3);
                            for (int y = 0; y < m_floor.rows; ++y)
                            {
                                const cv::Vec3b* cp = bgr_align.ptr<cv::Vec3b>(y);
                                const float* ap = alpha_use.ptr<float>(y);
                                const uint8_t* kp = keep.ptr<uint8_t>(y);
                                const float* flp = sh_floor_lf.ptr<float>(y);
                                const float* shp = sh_soft.ptr<float>(y);
                                const float* sbp = sh_bin.ptr<float>(y);
                                const float* cwp =
                                    use_3d_person_shadow ? cast_w.ptr<float>(y) : sh_roi_f.ptr<float>(y);
                                cv::Vec3b* op = m_floor.ptr<cv::Vec3b>(y);
                                for (int x = 0; x < m_floor.cols; ++x)
                                {
                                    const float rp = cwp[x];
                                    const float al = kp[x] ? std::clamp(ap[x], 0.f, 1.f) : 0.0f;
                                    const float floor_w =
                                        ui.rvm_floor_overlay_on ? std::clamp(flp[x] * 0.45f, 0.0f, 1.0f) : 0.0f;
                                    // 팔다리/허리 얼룩을 줄이기 위해 "부드러운 맵 + 이진 소프트 맵"을 섞어 단일 톤화
                                    const float sh_unified = std::clamp(0.35f * shp[x] + 0.95f * sbp[x], 0.0f, 1.0f);
                                    const float shadow_w =
                                        ui.rvm_shadow_overlay_on
                                            ? (use_3d_person_shadow
                                                   ? std::clamp(rp * ui.cast_shadow_alpha * 1.25f, 0.0f, 1.0f)
                                                   : std::clamp(sh_unified * rp * ui.cast_shadow_alpha * 1.25f, 0.0f,
                                                                1.0f))
                                            : 0.0f;
                                    const float dark_w = 1.0f - (1.0f - floor_w) * (1.0f - shadow_w);
                                    cv::Vec3b bg_shadowed;
                                    bg_shadowed[0] = static_cast<uint8_t>(std::clamp(kGreenBg[0] * (1.0f - dark_w), 0.0f, 255.0f));
                                    bg_shadowed[1] = static_cast<uint8_t>(std::clamp(kGreenBg[1] * (1.0f - dark_w), 0.0f, 255.0f));
                                    bg_shadowed[2] = static_cast<uint8_t>(std::clamp(kGreenBg[2] * (1.0f - dark_w), 0.0f, 255.0f));
                                    const float person_w = std::clamp((al - 0.08f) / 0.30f, 0.0f, 1.0f);
                                    op[x][0] = static_cast<uint8_t>(cp[x][0] * person_w + bg_shadowed[0] * (1.f - person_w));
                                    op[x][1] = static_cast<uint8_t>(cp[x][1] * person_w + bg_shadowed[1] * (1.f - person_w));
                                    op[x][2] = static_cast<uint8_t>(cp[x][2] * person_w + bg_shadowed[2] * (1.f - person_w));
                                }
                            }
                            if (ui.rvm_reflection_overlay_on && have_depth_intrin && !m_floor.empty())
                                rvm_floor::apply_depth_mirror_person_reflection(m_floor,
                                                                              bgr_align,
                                                                              d16_align,
                                                                              alpha_use,
                                                                              fp,
                                                                              ui.floor_refl);
                        }
                        else
                        {
                            cv::Mat bg;
                            if (ui.rvm_bg_mode == 0)
                                bg = cv::Mat::zeros(bgr_align.size(), CV_8UC3);
                            else
                                bg = bgr_align;
                            m_floor.create(bgr_align.size(), CV_8UC3);
                            for (int y = 0; y < m_floor.rows; ++y)
                            {
                                const cv::Vec3b* cp = bgr_align.ptr<cv::Vec3b>(y);
                                const cv::Vec3b* bp = bg.ptr<cv::Vec3b>(y);
                                const float* ap = alpha_use.ptr<float>(y);
                                cv::Vec3b* op = m_floor.ptr<cv::Vec3b>(y);
                                for (int x = 0; x < m_floor.cols; ++x)
                                {
                                    const float al = std::clamp(ap[x], 0.f, 1.f);
                                    op[x][0] = static_cast<uint8_t>(cp[x][0] * al + bp[x][0] * (1.f - al));
                                    op[x][1] = static_cast<uint8_t>(cp[x][1] * al + bp[x][1] * (1.f - al));
                                    op[x][2] = static_cast<uint8_t>(cp[x][2] * al + bp[x][2] * (1.f - al));
                                }
                            }
                        }
                    }
                    else
                    {
                        m_color_fx.release();
                        m_floor.release();
                    }
                }
                else
                {
                    m_color_fx.release();
                    m_floor.release();
                }
            }
        }

        tex_ir.ensure();
        tex_depth.ensure();
        tex_color.ensure();
        tex_floor.ensure();
        if (ui.show_ir)
            upload_tex_rgba(tex_ir.id, m_ir);
        if (ui.show_depth)
            upload_tex_rgba(tex_depth.id, m_depth);
        if (ui.show_color)
        {
            cv::Mat color_disp;
            // raw color 모드는 항상 RGB 카메라 원본 비율/해상도를 기준으로 표시
            if (ui.color_view_mode == 0)
                color_disp = !m_color_mr_rgb.empty() ? m_color_mr_rgb : m_color;
            else if (!m_color_fx.empty())
                color_disp = m_color_fx;
            else
                color_disp = m_color;

            cv::Mat color_out = color_disp;
            if (ui.color_depth_view_mode == 2 && !m_depth_rgb_vis.empty() && color_disp.size() == m_depth_rgb_vis.size())
            {
                // Depth only에서도 MR 오브젝트/그림자 확인이 필요하므로 약하게 합성 결과를 남긴다.
                cv::addWeighted(m_depth_rgb_vis, 0.78, color_disp, 0.22, 0.0, color_out);
            }
            else if (ui.color_depth_view_mode == 1 && !m_depth_rgb_vis.empty() && !color_disp.empty() &&
                     color_disp.size() == m_depth_rgb_vis.size())
            {
            const float a = std::clamp(ui.color_depth_overlay_alpha, 0.f, 1.f);
            cv::addWeighted(color_disp, 1.0 - a, m_depth_rgb_vis, a, 0.0, color_out);
            }
            upload_tex_rgba(tex_color.id, color_out);
        }
        if (ui.show_floor_panel && !m_floor.empty())
            upload_tex_rgba(tex_floor.id, m_floor);

        ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x - 460, vp->WorkSize.y), ImGuiCond_FirstUseEver);
        ImGuiWindowFlags streams_flags = ImGuiWindowFlags_None;
        if (ui.view_mode == kView2D)
            streams_flags |= ImGuiWindowFlags_NoScrollbar;
        ImGui::Begin("Streams", nullptr, streams_flags);

        if (ui.view_mode == kView2D)
        {
            const float grid_h = ImGui::GetContentRegionAvail().y;
            ImGui::BeginChild("stream_grid_2d", ImVec2(0.0f, std::max(1.0f, grid_h)), false,
                              ImGuiWindowFlags_NoScrollbar);
            const float col_gap = ImGui::GetStyle().ItemSpacing.x;
            const float row_gap = ImGui::GetStyle().ItemSpacing.y;
            ImVec2 inner = ImGui::GetContentRegionAvail();
            const float half_w = std::max(80.0f, (inner.x - col_gap) * 0.5f);
            const float half_h = std::max(80.0f, (inner.y - row_gap) * 0.5f);

            ImGui::BeginChild("tl_floor_ir", ImVec2(half_w, half_h), true, ImGuiWindowFlags_NoScrollbar);
            const bool has_floor_img = ui.show_floor_panel && tex_floor.id && !m_floor.empty();
            if (has_floor_img)
            {
                if (ui.person_segment_depth_only)
                    ImGui::TextUnformatted("FLOOR합성 (DEPTH ONLY)");
                else if (ui.rvm_bg_mode == 2)
                    ImGui::TextUnformatted("FLOOR합성 (RVM MATTE ONLY)");
                else
                    ImGui::TextUnformatted("Floor 합성 (RVM / matte)");
            }
            else if (ui.show_ir)
                ImGui::TextUnformatted("IR (합성 없을 때 대체)");
            else
                ImGui::TextDisabled("(왼쪽 위 패널 끔)");
            if (!has_floor_img && ui.show_floor_panel && !ui.person_segment_depth_only)
            {
                if (!rvm_eng.net_loaded)
                    ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.35f, 1.0f), "RVM 미로드: ONNX를 로드하세요.");
                else
                    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
                                       "RVM matte 미검출: 조명/피사체 위치를 조정하거나 RVM 모델을 확인하세요.");
            }
            if (has_floor_img)
            {
                const ImVec2 img_sz = ImGui::GetContentRegionAvail();
                ImGui::Image((ImTextureID)(intptr_t)tex_floor.id, img_sz, ImVec2(0, 0), ImVec2(1, 1));
                if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                {
                    const ImVec2 mp = io.MousePos;
                    const ImVec2 r0 = ImGui::GetItemRectMin();
                    const ImVec2 r1 = ImGui::GetItemRectMax();
                    const float rw = std::max(1e-4f, r1.x - r0.x);
                    const float rh = std::max(1e-4f, r1.y - r0.y);
                    ui.floor_anchor_u = std::clamp((mp.x - r0.x) / rw, 0.f, 1.f);
                    ui.floor_anchor_v = std::clamp((mp.y - r0.y) / rh, 0.f, 1.f);
                    ui.floor_anchor_valid = true;
                }
            }
            else if (ui.show_ir && tex_ir.id && !m_ir.empty())
                ImGui::Image((ImTextureID)(intptr_t)tex_ir.id, ImGui::GetContentRegionAvail(), ImVec2(0, 0), ImVec2(1, 1));
            else if (ui.show_floor_panel && streaming)
                ImGui::TextWrapped(
                    "깊이+컬러 정합 필요. Floor 갱신은 N프레임마다 — 잠시 검은 화면일 수 있음.");
            ImGui::EndChild();
            ImGui::SameLine();
            ImGui::BeginChild("depth", ImVec2(half_w, half_h), true, ImGuiWindowFlags_NoScrollbar);
            if (ui.show_depth)
            {
                ImGui::TextUnformatted("Depth Camera");
                if (tex_depth.id && !m_depth.empty())
                    ImGui::Image((ImTextureID)(intptr_t)tex_depth.id, ImGui::GetContentRegionAvail(), ImVec2(0, 0), ImVec2(1, 1));
            }
            else
                ImGui::TextDisabled("Depth 패널 끔");
            ImGui::EndChild();
            ImGui::BeginChild("color", ImVec2(half_w, half_h), true, ImGuiWindowFlags_NoScrollbar);
            if (ui.show_color)
            {
                if (ui.color_view_mode == 0 && !m_color_fx.empty())
                    ImGui::TextUnformatted("Color Camera");
                else if (ui.color_view_mode == 0)
                    ImGui::TextUnformatted("Color Camera");
                else if (ui.color_view_mode == 1)
                    ImGui::TextUnformatted("Color Camera (RVM human matte)");
                else
                    ImGui::TextUnformatted("Color Camera (RVM matte + depth shadow)");
                if (ui.color_depth_view_mode == 1)
                    ImGui::TextDisabled("RGB depth overlay");
                else if (ui.color_depth_view_mode == 2)
                    ImGui::TextDisabled("RGB depth only (MR blended)");
                if (tex_color.id && ((ui.color_view_mode == 0 && (!m_color.empty() || !m_color_fx.empty())) ||
                                     (ui.color_view_mode > 0 && !m_color_fx.empty())))
                {
                    // RGB 비율 유지한 상태로 fit 렌더 + 같은 기준으로 클릭을 UV로 변환
                    const ImVec2 avail = ImGui::GetContentRegionAvail();
                    const int src_w = (!m_color.empty() ? m_color.cols : (!m_color_fx.empty() ? m_color_fx.cols : 0));
                    const int src_h = (!m_color.empty() ? m_color.rows : (!m_color_fx.empty() ? m_color_fx.rows : 0));
                    ImVec2 draw_sz = avail;
                    if (src_w > 0 && src_h > 0 && avail.x > 1.0f && avail.y > 1.0f)
                    {
                        const float ar = static_cast<float>(src_w) / static_cast<float>(src_h);
                        const float avail_ar = avail.x / avail.y;
                        if (avail_ar > ar)
                        {
                            draw_sz.y = avail.y;
                            draw_sz.x = draw_sz.y * ar;
                        }
                        else
                        {
                            draw_sz.x = avail.x;
                            draw_sz.y = draw_sz.x / ar;
                        }
                    }
                    // 가운데 정렬(레터박스)
                    const float off_x = std::max(0.0f, (avail.x - draw_sz.x) * 0.5f);
                    const float off_y = std::max(0.0f, (avail.y - draw_sz.y) * 0.5f);
                    {
                        const ImVec2 cur = ImGui::GetCursorPos();
                        ImGui::SetCursorPos(ImVec2(cur.x + off_x, cur.y + off_y));
                    }
                    ImGui::Image((ImTextureID)(intptr_t)tex_color.id, draw_sz, ImVec2(0, 0), ImVec2(1, 1));

                    if (ui.mr_virtual_object_enabled && ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                    {
                        const ImVec2 mp = io.MousePos;
                        const ImVec2 r0 = ImGui::GetItemRectMin();
                        const ImVec2 r1 = ImGui::GetItemRectMax();
                        const float rw_img = std::max(1e-4f, r1.x - r0.x);
                        const float rh_img = std::max(1e-4f, r1.y - r0.y);
                        // 레터박스가 이미 제외된 실제 이미지 rect이므로 그대로 UV 변환
                        ui.mr_object_u = std::clamp((mp.x - r0.x) / rw_img, 0.f, 1.f);
                        ui.mr_object_v = std::clamp((mp.y - r0.y) / rh_img, 0.f, 1.f);
                        // Ray picking 단순화: depth pixel + intrinsic으로 depth camera 3D 복원
                        const bool pick_is_rgb = !m_depth_rgb_u16.empty();
                        const cv::Mat& pick_depth = pick_is_rgb ? m_depth_rgb_u16 : m_depth_aligned_u16;
                        const bool have_intrin_for_pick = pick_is_rgb ? have_color_intrin : have_depth_intrin;
                        const float fx_pick = pick_is_rgb ? color_fx : depth_fx;
                        const float fy_pick = pick_is_rgb ? color_fy : depth_fy;
                        const float cx_pick = pick_is_rgb ? color_cx : depth_cx;
                        const float cy_pick = pick_is_rgb ? color_cy : depth_cy;
                        if (have_intrin_for_pick && !pick_depth.empty())
                        {
                            const int iw = pick_depth.cols;
                            const int ih = pick_depth.rows;
                            const int u0 = std::clamp(static_cast<int>(std::round(ui.mr_object_u * (iw - 1))), 0, iw - 1);
                            const int v0 = std::clamp(static_cast<int>(std::round(ui.mr_object_v * (ih - 1))), 0, ih - 1);
                            int u = u0, v = v0;
                            uint16_t dmm = pick_depth.at<uint16_t>(v, u);
                            // 픽셀 하나가 비어 있으면 주변에서 가장 가까운 유효 depth를 탐색 (작은 반경).
                            if (dmm <= 250)
                            {
                                bool found = false;
                                for (int r = 1; r <= 6 && !found; ++r)
                                {
                                    for (int yy = std::max(0, v0 - r); yy <= std::min(ih - 1, v0 + r) && !found; ++yy)
                                    {
                                        for (int xx = std::max(0, u0 - r); xx <= std::min(iw - 1, u0 + r); ++xx)
                                        {
                                            const uint16_t z = pick_depth.at<uint16_t>(yy, xx);
                                            if (z > 250)
                                            {
                                                dmm = z;
                                                u = xx;
                                                v = yy;
                                                found = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            if (dmm > 250)
                            {
                                const float d = static_cast<float>(dmm) * 0.001f;
                                mr_pick_x_m = (static_cast<float>(u) - cx_pick) * d / fx_pick;
                                mr_pick_y_m = (static_cast<float>(v) - cy_pick) * d / fy_pick;
                                mr_pick_z_m = d;
                                mr_pick_u = u;
                                mr_pick_v = v;
                                mr_pick_valid = true;
                                // 클릭 배치는 깊이 지점(바닥/벽 포함)으로 바로 반영되도록 월드 트랜스폼 자동 동기화
                                ui.mr_use_world_transform = true;
                                ui.mr_obj_tx_m = mr_pick_x_m;
                                ui.mr_obj_ty_m = mr_pick_y_m;
                                const float snap_z = mr_pick_z_m + ui.mr_ground_snap_offset_m;
                                ui.mr_obj_tz_m = std::max(0.3f, ui.mr_ground_snap_on_click ? snap_z : mr_pick_z_m);
                                ui.mr_object_depth_mm = ui.mr_obj_tz_m * 1000.0f;
                            }
                            else
                            {
                                mr_pick_valid = false;
                            }
                        }
                        else
                        {
                            mr_pick_valid = false;
                        }
                    }
                }
                else if (ui.color_view_mode > 0)
                    ImGui::TextDisabled("RVM/정합 데이터 대기 중...");
            }
            else
                ImGui::TextDisabled("Color 패널 끔");
            ImGui::EndChild();
            ImGui::SameLine();
            ImGui::BeginChild("imu", ImVec2(half_w, half_h), true, ImGuiWindowFlags_NoScrollbar);
            if (ui.show_imu_panel)
            {
                ImGui::TextUnformatted("IMU");
                ImGui::TextWrapped("%s", imu_line.c_str());
                if (ui.imu_on && imu_line.find("no sample") != std::string::npos)
                    ImGui::TextColored(ImVec4(1, 0.5f, 0.2f, 1), "Data source failed!");
            }
            else
                ImGui::TextDisabled("IMU 패널 끔");
            ImGui::EndChild();
            ImGui::EndChild();
        }
        else
        {
            ImGui::Text("%s: Point Cloud Viewer", serial.c_str());
            ImGui::RadioButton("Simple", &ui.cloud_style, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Shaded", &ui.cloud_style, 1);
            ImGui::SameLine();
            ImGui::RadioButton("Color", &ui.cloud_style, 2);
            ImGui::SliderFloat("Point Size", &ui.cloud_point_px, 1.0f, 10.0f);
            ImGui::SliderFloat("Orbit distance", &ui.orbit_dist, 0.8f, 8.0f);
            if (ImGui::Button("Reset view"))
            {
                yaw = 0.35f;
                pitch = 0.25f;
                ui.orbit_dist = 2.2f;
                ui.cloud_point_px = 2.0f;
            }
            ImVec2 r = ImGui::GetContentRegionAvail();
            const int rw = std::max(64, static_cast<int>(r.x));
            const int rh = std::max(64, static_cast<int>(r.y));
            if (rw != fbo_w || rh != fbo_h)
            {
                if (fbo)
                    glDeleteFramebuffers(1, &fbo);
                if (fbo_tex)
                    glDeleteTextures(1, &fbo_tex);
                if (rbo)
                    glDeleteRenderbuffers(1, &rbo);
                fbo = fbo_tex = rbo = 0;
                fbo_w = rw;
                fbo_h = rh;
                glGenFramebuffers(1, &fbo);
                glGenTextures(1, &fbo_tex);
                glBindTexture(GL_TEXTURE_2D, fbo_tex);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fbo_w, fbo_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_tex, 0);
                glGenRenderbuffers(1, &rbo);
                glBindRenderbuffer(GL_RENDERBUFFER, rbo);
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, fbo_w, fbo_h);
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            }
            ImGui::InvisibleButton("pc_canvas", ImVec2(static_cast<float>(rw), static_cast<float>(rh)));
            const bool hovered = ImGui::IsItemHovered();
            if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            {
                dragging_pc = true;
                drag_sx = io.MousePos.x;
                drag_sy = io.MousePos.y;
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
                dragging_pc = false;
            if (dragging_pc)
            {
                yaw += static_cast<float>(io.MousePos.x - drag_sx) * 0.01f;
                pitch += static_cast<float>(io.MousePos.y - drag_sy) * 0.01f;
                pitch = std::clamp(pitch, -1.4f, 1.4f);
                drag_sx = io.MousePos.x;
                drag_sy = io.MousePos.y;
            }
            if (fbo)
            {
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glViewport(0, 0, fbo_w, fbo_h);
                glClearColor(0.02f, 0.02f, 0.05f, 1);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                glEnable(GL_DEPTH_TEST);
                draw_point_cloud(pc_program, pcgl, yaw, pitch, ui.orbit_dist, ui.cloud_point_px, fbo_w, fbo_h);
                glDisable(GL_DEPTH_TEST);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            }
            glViewport(0, 0, fb_w, fb_h);
            ImGui::GetWindowDrawList()->AddImage(
                (ImTextureID)(intptr_t)fbo_tex,
                ImGui::GetItemRectMin(),
                ImGui::GetItemRectMax(),
                ImVec2(0, 0),
                ImVec2(1, 1));
        }
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        {
            viewer_io::KVMap m;
            ui_to_kv(ui, rvm_onnx_buf, m);
            static std::string g_last_settings_snap;
            const std::string snap = canonical_kv_string(m);
            if (snap != g_last_settings_snap)
            {
                viewer_io::save_ini(viewer_io::default_settings_path(), m);
                g_last_settings_snap = snap;
            }
        }

        glfwSwapBuffers(window);
    }

    save_app_settings(ui, rvm_onnx_buf);

    if (streaming && dev)
    {
        if (imu_running)
        {
            dev->stop_imu();
            imu_running = false;
        }
        dev->stop_cameras();
        streaming = false;
    }
    if (dev)
    {
        dev->close();
        dev.reset();
    }

    pcgl.destroy();
    if (pc_program)
        glDeleteProgram(pc_program);
    if (fbo)
        glDeleteFramebuffers(1, &fbo);
    if (fbo_tex)
        glDeleteTextures(1, &fbo_tex);
    if (rbo)
        glDeleteRenderbuffers(1, &rbo);
    tex_ir.destroy();
    tex_depth.destroy();
    tex_color.destroy();
    tex_floor.destroy();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
