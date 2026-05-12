#include "rvm_floor_fx.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

/**
 * Azure Kinect 정렬 컬러/깊이 + RVM(또는 깊이 보조) 알파를 이용한 바닥 합성.
 * - 바닥 평면 RANSAC, 바닥 마스크
 * - 접촉 그림자(가우시안), 2D 거울 반사, 3D 캐스트 그림자(평행광)
 * - ONNX Runtime 또는 OpenCV DNN으로 RVM 추론(선택)
 */
namespace rvm_floor {

namespace {

#ifdef RVM_USE_ONNXRUNTIME
std::wstring utf8_to_wide(const std::string& s)
{
    if (s.empty())
        return {};
    try
    {
        return std::filesystem::u8path(s).wstring();
    }
    catch (...)
    {
        return {};
    }
}

template <size_t N>
bool fetch_node_names(const Ort::Session& session,
                      bool is_input,
                      std::array<std::string, N>& out_names,
                      const std::array<const char*, N>& preferred)
{
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t count = is_input ? session.GetInputCount() : session.GetOutputCount();
    if (count < N)
        return false;
    std::vector<std::string> got;
    got.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        Ort::AllocatedStringPtr p = is_input ? session.GetInputNameAllocated(i, alloc)
                                             : session.GetOutputNameAllocated(i, alloc);
        got.emplace_back(p ? p.get() : "");
    }
    for (size_t i = 0; i < N; ++i)
    {
        auto it = std::find(got.begin(), got.end(), preferred[i]);
        if (it != got.end())
            out_names[i] = *it;
        else if (i < got.size())
            out_names[i] = got[i];
        else
            return false;
    }
    return true;
}

#endif

cv::Mat nchw_pha_to_hw(const cv::Mat& pha_nchw)
{
    if (pha_nchw.dims == 4 && pha_nchw.size[2] > 0 && pha_nchw.size[3] > 0)
    {
        const int h = pha_nchw.size[2];
        const int w = pha_nchw.size[3];
        cv::Mat f(h, w, CV_32F);
        const float* src = pha_nchw.ptr<float>(0);
        std::memcpy(f.data, src, static_cast<size_t>(h * w) * sizeof(float));
        return f;
    }
    if (pha_nchw.rows > 1 && pha_nchw.cols > 1 && pha_nchw.channels() == 1)
        return pha_nchw.clone();
    return {};
}

void ensure_same_size(const cv::Mat& a, cv::Mat& b, int interpolation = cv::INTER_LINEAR)
{
    if (a.size() == b.size())
        return;
    cv::Mat r;
    cv::resize(b, r, a.size(), 0, 0, interpolation);
    b = r;
}

/** 이미지 좌표 기반 근사 바닥: 깊이 z(mm) ≈ a*x + b*y + c (intrinsics 없을 때 보조) */
struct PlaneZ
{
    float a = 0.f, b = 0.f, c = 0.f;
    bool valid = false;
};

/** 카메라 좌표(미터) 평면 n·p + d = 0, 법선 n 단위벡터 */
struct Plane3
{
    cv::Vec3f n{0.f, 0.f, 0.f};
    float d = 0.f;
    bool valid = false;
};

/** 픽셀(u,v) + 깊이(mm) → 카메라 공간 3D점(미터) */
static inline bool backproject_m(int u, int v, uint16_t z_mm, const Params& p, cv::Vec3f& out)
{
    if (!p.cam_intrin_valid || z_mm < 250 || z_mm > 8000)
        return false;
    const float z = static_cast<float>(z_mm) * 0.001f;
    const float x = (static_cast<float>(u) - p.cam_cx) * z / p.cam_fx;
    const float y = (static_cast<float>(v) - p.cam_cy) * z / p.cam_fy;
    out = cv::Vec3f(x, y, z);
    return true;
}

static inline bool fit_plane3_cam(const cv::Vec3f& p1, const cv::Vec3f& p2, const cv::Vec3f& p3, Plane3& out)
{
    const cv::Vec3f v1 = p2 - p1;
    const cv::Vec3f v2 = p3 - p1;
    cv::Vec3f n(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
    const float nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (nn < 1e-6f)
        return false;
    n *= (1.0f / nn);
    /* 바닥 법선은 카메라 Y(화면 아래) 성분이 커야 함 — Y가 약하면 바닥이 아닌 벽/잡음일 가능성 */
    if (std::abs(n[1]) < 0.55f)
        return false;
    const float d = -(n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2]);
    out.n = n;
    out.d = d;
    out.valid = true;
    return true;
}

static cv::Vec3f normalized_safe(const cv::Vec3f& v)
{
    const float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n < 1e-6f)
        return cv::Vec3f(0.f, 1.f, 0.f);
    return v * (1.0f / n);
}

/**
 * 깊이+알파로 바닥 평면 추정(카메라 3D).
 * 하단에서 사람 알파로 발 위치 추정 → 발 주변 ROI의 바닥 깊이만 샘플 → RANSAC.
 * plane_temporal_smooth_on 이면 프레임 간 EMA로 흔들림 완화.
 */
static Plane3 estimate_floor_plane_ransac_cam(const cv::Mat& d, const cv::Mat& a, const Params& p)
{
    if (!p.cam_intrin_valid)
        return {};
    const int H = d.rows;
    const int W = d.cols;
    /* 발 쪽 x 중심: 화면 아래쪽부터 스캔해 알파가 큰 열의 평균 */
    float foot_x = 0.5f * static_cast<float>(W);
    bool foot_found = false;
    {
        double sx = 0.0;
        int cnt = 0;
        for (int y = H - 1; y >= std::max(0, static_cast<int>(H * 0.45f)); --y)
        {
            const float* ar = a.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                if (ar[x] > 0.35f)
                {
                    sx += static_cast<double>(x);
                    ++cnt;
                }
            }
            if (cnt > 12)
            {
                foot_x = static_cast<float>(sx / cnt);
                foot_found = true;
                break;
            }
        }
    }
    const float roi_half = std::clamp(p.plane_foot_roi_radius, 0.08f, 0.50f) * static_cast<float>(W);
    const float roi_x0 = std::clamp(foot_x - roi_half, 0.0f, static_cast<float>(W - 1));
    const float roi_x1 = std::clamp(foot_x + roi_half, 0.0f, static_cast<float>(W - 1));
    std::vector<cv::Vec3f> pts;
    pts.reserve(static_cast<size_t>(W) * 16);
    const int y0 = static_cast<int>(H * 0.70f);
    const int step = 3;
    for (int y = y0; y < H; y += step)
    {
        const uint16_t* dr = d.ptr<uint16_t>(y);
        const float* ar = a.ptr<float>(y);
        for (int x = 0; x < W; x += step)
        {
            const uint16_t z = dr[x];
            if (z < 250 || z > 8000 || ar[x] > 0.20f)
                continue;
            if (p.plane_use_foot_roi && foot_found)
            {
                if (x < static_cast<int>(roi_x0) || x > static_cast<int>(roi_x1))
                    continue;
            }
            cv::Vec3f pt;
            if (backproject_m(x, y, z, p, pt))
                pts.push_back(pt);
        }
    }
    /* 발 ROI만 쓰면 점이 너무 적을 때: ROI 제한 없이 바닥 후보 다시 모음 */
    if (pts.size() < 24 && p.plane_use_foot_roi)
    {
        pts.clear();
        for (int y = y0; y < H; y += step)
        {
            const uint16_t* dr = d.ptr<uint16_t>(y);
            const float* ar = a.ptr<float>(y);
            for (int x = 0; x < W; x += step)
            {
                const uint16_t z = dr[x];
                if (z < 250 || z > 8000 || ar[x] > 0.20f)
                    continue;
                cv::Vec3f pt;
                if (backproject_m(x, y, z, p, pt))
                    pts.push_back(pt);
            }
        }
    }
    if (pts.size() < 36)
        return {};

    cv::RNG rng(0x2442);
    Plane3 best{};
    int best_inliers = 0;
    /* 3점으로 평면 후보 → inlier 수가 최대인 평면 선택 */
    for (int it = 0; it < 140; ++it)
    {
        const int i1 = rng.uniform(0, static_cast<int>(pts.size()));
        const int i2 = rng.uniform(0, static_cast<int>(pts.size()));
        const int i3 = rng.uniform(0, static_cast<int>(pts.size()));
        if (i1 == i2 || i2 == i3 || i1 == i3)
            continue;
        Plane3 m;
        if (!fit_plane3_cam(pts[i1], pts[i2], pts[i3], m))
            continue;
        int inl = 0;
        for (const auto& q : pts)
        {
            const float dist = std::abs(m.n[0] * q[0] + m.n[1] * q[1] + m.n[2] * q[2] + m.d);
            if (dist < 0.09f)
                ++inl;
        }
        if (inl > best_inliers)
        {
            best_inliers = inl;
            best = m;
        }
    }
    if (best_inliers < 28)
        return {};

    if (!p.plane_temporal_smooth_on)
        return best;
    const float alpha = std::clamp(p.plane_temporal_alpha, 0.02f, 1.0f);
    /* 동일 카메라 해상도일 때만 이전 프레임 평면과 (n,d) 선형 보간 */
    struct PlaneHist
    {
        bool valid = false;
        int w = 0;
        int h = 0;
        float fx = 0.f;
        float fy = 0.f;
        Plane3 plane{};
    };
    static PlaneHist s_hist;
    const bool same_camera = s_hist.valid && s_hist.w == W && s_hist.h == H &&
                             std::abs(s_hist.fx - p.cam_fx) < 1e-3f &&
                             std::abs(s_hist.fy - p.cam_fy) < 1e-3f;
    if (!same_camera)
    {
        s_hist.valid = true;
        s_hist.w = W;
        s_hist.h = H;
        s_hist.fx = p.cam_fx;
        s_hist.fy = p.cam_fy;
        s_hist.plane = best;
        return best;
    }
    Plane3 smooth = best;
    cv::Vec3f prev_n = s_hist.plane.n;
    if (prev_n.dot(best.n) < 0.0f)
        prev_n = -prev_n;
    smooth.n = normalized_safe(prev_n * (1.0f - alpha) + best.n * alpha);
    smooth.d = s_hist.plane.d * (1.0f - alpha) + best.d * alpha;
    smooth.valid = true;
    s_hist.plane = smooth;
    return smooth;
}

/** 세 점 (x,y,z픽셀)으로 PlaneZ 계수 맞춤 */
bool fit_plane3(const cv::Point3f& p1, const cv::Point3f& p2, const cv::Point3f& p3, PlaneZ& out)
{
    const float x1 = p1.x, y1 = p1.y, z1 = p1.z;
    const float x2 = p2.x, y2 = p2.y, z2 = p2.z;
    const float x3 = p3.x, y3 = p3.y, z3 = p3.z;
    const float d = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
    if (std::abs(d) < 1e-6f)
        return false;
    const float a = (z1 * (y2 - y3) + z2 * (y3 - y1) + z3 * (y1 - y2)) / d;
    const float b = (z1 * (x3 - x2) + z2 * (x1 - x3) + z3 * (x2 - x1)) / d;
    const float c = (z1 * (x2 * y3 - x3 * y2) + z2 * (x3 * y1 - x1 * y3) + z3 * (x1 * y2 - x2 * y1)) / d;
    out.a = a;
    out.b = b;
    out.c = c;
    out.valid = true;
    return true;
}

/** intrinsics 없을 때: (x,y,z) 픽셀 좌표 RANSAC으로 z=ax+by+c 근사 */
PlaneZ estimate_floor_plane_ransac(const cv::Mat& d, const cv::Mat& a)
{
    const int H = d.rows;
    const int W = d.cols;
    std::vector<cv::Point3f> pts;
    pts.reserve(static_cast<size_t>(W) * 16);
    const int y0 = static_cast<int>(H * 0.70f);
    const int step = 3;
    for (int y = y0; y < H; y += step)
    {
        const uint16_t* dr = d.ptr<uint16_t>(y);
        const float* ar = a.ptr<float>(y);
        for (int x = 0; x < W; x += step)
        {
            const uint16_t z = dr[x];
            if (z < 250 || z > 8000 || ar[x] > 0.20f)
                continue;
            pts.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        }
    }
    if (pts.size() < 30)
        return {};

    cv::RNG rng(0x2442);
    PlaneZ best{};
    int best_inliers = 0;
    for (int it = 0; it < 120; ++it)
    {
        const int i1 = rng.uniform(0, static_cast<int>(pts.size()));
        const int i2 = rng.uniform(0, static_cast<int>(pts.size()));
        const int i3 = rng.uniform(0, static_cast<int>(pts.size()));
        if (i1 == i2 || i2 == i3 || i1 == i3)
            continue;
        PlaneZ m;
        if (!fit_plane3(pts[i1], pts[i2], pts[i3], m))
            continue;
        if (std::abs(m.a) > 35.f || std::abs(m.b) > 35.f)
            continue;

        int inl = 0;
        for (const auto& p : pts)
        {
            const float z_hat = m.a * p.x + m.b * p.y + m.c;
            if (std::abs(p.z - z_hat) < 90.f)
                ++inl;
        }
        if (inl > best_inliers)
        {
            best_inliers = inl;
            best = m;
        }
    }
    if (best_inliers < 28)
        return {};
    return best;
}

/**
 * 바닥 영역 이진 마스크(255=바닥).
 * intrinsics 있으면 3D 평면 거리로, 없으면 PlaneZ 깊이 차이로 후보 픽셀을 채운 뒤
 * 형태학·연결요소로 아래쪽 큰 덩어리만 남김.
 */
static void build_floor_mask_u8(int H, int W, const cv::Mat& d, const cv::Mat& a, const Params& p, cv::Mat& floor_mask)
{
    floor_mask.create(H, W, CV_8U);
    floor_mask.setTo(0);
    if (!d.empty() && d.type() == CV_16U)
    {
        const Plane3 plane3 = estimate_floor_plane_ransac_cam(d, a, p);
        if (plane3.valid)
        {
            const int y_top = static_cast<int>(H * 0.44f);
            for (int y = y_top; y < H; ++y)
            {
                const uint16_t* dr = d.ptr<uint16_t>(y);
                const float* ar = a.ptr<float>(y);
                uint8_t* fm = floor_mask.ptr<uint8_t>(y);
                for (int x = 0; x < W; ++x)
                {
                    const uint16_t z = dr[x];
                    if (z < 250 || z > 8000 || ar[x] > 0.20f)
                        continue;
                    cv::Vec3f pt;
                    if (!backproject_m(x, y, z, p, pt))
                        continue;
                    const float dist =
                        std::abs(plane3.n[0] * pt[0] + plane3.n[1] * pt[1] + plane3.n[2] * pt[2] + plane3.d);
                    if (dist < 0.080f)
                        fm[x] = 255;
                }
            }
            cv::morphologyEx(floor_mask,
                             floor_mask,
                             cv::MORPH_CLOSE,
                             cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 9)));
            cv::morphologyEx(floor_mask,
                             floor_mask,
                             cv::MORPH_OPEN,
                             cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

            cv::Mat labels, stats, centroids;
            const int nlabels = cv::connectedComponentsWithStats(floor_mask, labels, stats, centroids, 8, CV_32S);
            if (nlabels > 1)
            {
                std::vector<uint8_t> keep(static_cast<size_t>(nlabels), 0);
                const int y_seed = static_cast<int>(H * 0.90f);
                for (int y = y_seed; y < H; ++y)
                {
                    const int* lp = labels.ptr<int>(y);
                    for (int x = 0; x < W; ++x)
                    {
                        const int id = lp[x];
                        if (id > 0 && id < nlabels)
                            keep[static_cast<size_t>(id)] = 1;
                    }
                }
                for (int id = 1; id < nlabels; ++id)
                {
                    const int area = stats.at<int>(id, cv::CC_STAT_AREA);
                    if (area < std::max(140, (H * W) / 1800))
                        keep[static_cast<size_t>(id)] = 0;
                }
                floor_mask.setTo(0);
                for (int y = 0; y < H; ++y)
                {
                    const int* lp = labels.ptr<int>(y);
                    uint8_t* fm = floor_mask.ptr<uint8_t>(y);
                    for (int x = 0; x < W; ++x)
                    {
                        const int id = lp[x];
                        if (id > 0 && id < nlabels && keep[static_cast<size_t>(id)])
                            fm[x] = 255;
                    }
                }
            }
        }
        else
        {
            const PlaneZ plane = estimate_floor_plane_ransac(d, a);
            if (plane.valid)
            {
                const int y_top = static_cast<int>(H * 0.44f);
                for (int y = y_top; y < H; ++y)
                {
                    const uint16_t* dr = d.ptr<uint16_t>(y);
                    const float* ar = a.ptr<float>(y);
                    uint8_t* fm = floor_mask.ptr<uint8_t>(y);
                    for (int x = 0; x < W; ++x)
                    {
                        const uint16_t z = dr[x];
                        if (z < 250 || z > 8000 || ar[x] > 0.20f)
                            continue;
                        const float zh = plane.a * static_cast<float>(x) + plane.b * static_cast<float>(y) + plane.c;
                        if (std::abs(static_cast<float>(z) - zh) < 80.f)
                            fm[x] = 255;
                    }
                }
                cv::morphologyEx(floor_mask,
                                 floor_mask,
                                 cv::MORPH_CLOSE,
                                 cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 9)));
                cv::morphologyEx(floor_mask,
                                 floor_mask,
                                 cv::MORPH_OPEN,
                                 cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
            }
        }
    }
    const int floor_area = cv::countNonZero(floor_mask);
    const int min_floor_area = std::max(200, static_cast<int>(H * W * 0.08f));
    if (floor_area < min_floor_area)
    {
        floor_mask.create(H, W, CV_8U);
        floor_mask.setTo(0);
        cv::rectangle(floor_mask, cv::Rect(0, static_cast<int>(H * 0.50f), W, H - static_cast<int>(H * 0.50f)),
                      cv::Scalar(255), -1);
    }
}

/**
 * 3D 캐스트 그림자 가중치 맵(CV_32F 0..1): 사람 픽셀에서 평행광 방향으로 바닥 평면과 교차 →
 * 교차점을 다시 투영해 바닥에 스플랫 누적. 에너지 낮으면 2D 워프 폴백.
 * 마지막에 바닥 마스크·가우시안·최댓값 정규화.
 */
static cv::Mat build_cast_shadow_sh_m_normalized(const cv::Mat& d,
                                                 const cv::Mat& a,
                                                 const cv::Mat& floor_mask,
                                                 const Params& p,
                                                 int H,
                                                 int W)
{
    static constexpr float kPi = 3.14159265f;
    cv::Mat empty;
    if (p.cast_shadow_alpha <= 1e-4f || !p.cam_intrin_valid || d.empty() || d.type() != CV_16U || floor_mask.empty())
        return empty;

    const Plane3 plane3 = estimate_floor_plane_ransac_cam(d, a, p);
    if (!plane3.valid)
        return empty;

    const float alt = std::clamp(p.sun_altitude_deg, 6.f, 88.f) * kPi / 180.f;
    const float azi = p.sun_azimuth_deg * kPi / 180.f;
    /* 태양 방향 단위벡터 L (카메라 좌표계, 평행광) */
    cv::Vec3f L(std::cos(azi) * std::cos(alt), std::sin(azi) * std::cos(alt), std::sin(alt));
    const float Ln = std::sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
    if (Ln > 1e-6f)
        L *= (1.0f / Ln);

    /* 태양이 너무 수평에 가까우면 광선이 평면과 거의 평행해져 수치 불안 → z 성분으로 상한 거리 완화 */
    const float lz_abs = std::max(std::abs(L[2]), 0.05f);
    const float ray_t_max_m = std::clamp(36.f / lz_abs, 10.f, 220.f);

    cv::Mat sh(H, W, CV_32F, cv::Scalar(0));
    const int step = std::clamp(p.shadow_ray_step, 1, 6);
    for (int y = 0; y < H; y += step)
    {
        const uint16_t* dr = d.ptr<uint16_t>(y);
        const float* ar = a.ptr<float>(y);
        for (int x = 0; x < W; x += step)
        {
            const float al = ar[x];
            if (al < 0.08f)
                continue;
            uint16_t z = dr[x];
            /* 사람 마스크에서 깊이 구멍이 흔함 → 이웃에서 유효 깊이 탐색 */
            if (z <= 250)
            {
                const int rmax = 3;
                bool found = false;
                for (int r = 1; r <= rmax && !found; ++r)
                {
                    for (int yy = std::max(0, y - r); yy <= std::min(H - 1, y + r) && !found; ++yy)
                    {
                        const uint16_t* dr2 = d.ptr<uint16_t>(yy);
                        for (int xx = std::max(0, x - r); xx <= std::min(W - 1, x + r); ++xx)
                        {
                            const uint16_t zz = dr2[xx];
                            if (zz > 250 && zz < 8000)
                            {
                                z = zz;
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if (!found)
                    continue;
            }
            cv::Vec3f pt;
            if (!backproject_m(x, y, z, p, pt))
                continue;
            const float num = -(plane3.n[0] * pt[0] + plane3.n[1] * pt[1] + plane3.n[2] * pt[2] + plane3.d);
            /* 광선 pt + t*dir 가 바닥과 만나는 t. L 또는 -L 둘 중 유효한 쪽 */
            auto try_dir = [&](const cv::Vec3f& dir, float& out_t) -> bool {
                const float denom = plane3.n[0] * dir[0] + plane3.n[1] * dir[1] + plane3.n[2] * dir[2];
                if (std::abs(denom) < 1e-4f)
                    return false;
                out_t = num / denom;
                return (out_t > 0.0f && out_t < ray_t_max_m);
            };
            float t = 0.f;
            cv::Vec3f dir = L;
            if (!try_dir(dir, t))
            {
                dir = -L;
                if (!try_dir(dir, t))
                    continue;
            }
            const cv::Vec3f q = pt + dir * t;
            if (q[2] <= 0.05f)
                continue;
            const float uf = p.cam_fx * (q[0] / q[2]) + p.cam_cx;
            const float vf = p.cam_fy * (q[1] / q[2]) + p.cam_cy;
            const int u0 = static_cast<int>(std::floor(uf));
            const int v0 = static_cast<int>(std::floor(vf));
            const float du = uf - static_cast<float>(u0);
            const float dv = vf - static_cast<float>(v0);
            const float w0 = al * 0.35f;
            const float w00 = w0 * (1.f - du) * (1.f - dv);
            const float w10 = w0 * du * (1.f - dv);
            const float w01 = w0 * (1.f - du) * dv;
            const float w11 = w0 * du * dv;
            /* 바닥 2x2 셀에 알파 가중치 분산(얇은 그림자 방지) */
            const struct { int u, v; float w; } splat[4] = {{u0, v0, w00}, {u0 + 1, v0, w10}, {u0, v0 + 1, w01}, {u0 + 1, v0 + 1, w11}};
            for (const auto& s : splat)
            {
                if (s.u < 0 || s.u >= W || s.v < 0 || s.v >= H || s.w <= 1e-8f)
                    continue;
                float& dst = sh.at<float>(s.v, s.u);
                dst = std::min(1.0f, dst + s.w);
            }
        }
    }

    /* 3D 누적이 거의 없을 때: 사람 실루엣을 태양 방위로 약간 워프한 2D 그림자로 보강 */
    if (cv::sum(sh)[0] < 100.0)
    {
        cv::Mat ath;
        cv::threshold(a, ath, 0.08, 255.0, cv::THRESH_BINARY);
        cv::Mat person_m;
        ath.convertTo(person_m, CV_8U);
        const float alt_eff = std::max(0.12f, alt);
        float stretch = 55.f / std::tan(alt_eff);
        stretch = std::clamp(stretch, 4.f, 240.f);
        float tx = -std::cos(azi) * stretch;
        float ty = -std::sin(azi) * stretch * 0.42f;
        tx = std::clamp(tx, -0.42f * static_cast<float>(W), 0.42f * static_cast<float>(W));
        ty = std::clamp(ty, -0.30f * static_cast<float>(H), 0.08f * static_cast<float>(H));
        const cv::Mat M =
            (cv::Mat_<double>(2, 3) << 1., 0., static_cast<double>(tx), 0., 1., static_cast<double>(ty));
        cv::Mat shifted;
        cv::warpAffine(person_m, shifted, M, cv::Size(W, H), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::Mat sh_fb;
        shifted.convertTo(sh_fb, CV_32F, 1.0 / 255.0);
        cv::max(sh, sh_fb * 0.38f, sh);
    }

    cv::Mat sh_m = sh;
    cv::Mat fm_f;
    floor_mask.convertTo(fm_f, CV_32F, 1.0 / 255.0);
    sh_m = sh_m.mul(fm_f);
    const double bs = std::max(0.8, static_cast<double>(p.cast_shadow_blur));
    cv::GaussianBlur(sh_m, sh_m, cv::Size(0, 0), bs, bs);

    /* 바닥 마스크 안 최댓값으로 0..1 정규화 → 이후 cast_shadow_alpha와 곱해 어둡게 합성 */
    double smin = 0.0, smax = 0.0;
    cv::minMaxLoc(sh_m, &smin, &smax, nullptr, nullptr, floor_mask);
    if (smax > 1e-4)
        sh_m.convertTo(sh_m, CV_32F, 1.0 / smax);
    else
        return empty;
    return sh_m;
}

} // namespace

/** 외부용: 바닥 마스크 + 3D 캐스트 그림자 가중치(0..1) */
cv::Mat compute_cast_shadow_weight(const cv::Mat& color_bgr,
                                  const cv::Mat& depth_u16,
                                  const cv::Mat& alpha_f32_hw,
                                  const Params& p)
{
    if (color_bgr.empty())
        return {};
    cv::Mat a = alpha_f32_hw;
    ensure_same_size(color_bgr, a, cv::INTER_LINEAR);
    cv::Mat d = depth_u16;
    if (!d.empty())
        ensure_same_size(color_bgr, d, cv::INTER_NEAREST);
    const int H = color_bgr.rows;
    const int W = color_bgr.cols;
    cv::Mat floor_mask;
    build_floor_mask_u8(H, W, d, a, p, floor_mask);
    return build_cast_shadow_sh_m_normalized(d, a, floor_mask, p, H, W);
}

/**
 * 깊이+바닥 평면으로 사람을 바닥에 거울상처럼 반사 블렌딩.
 * strength는 호출부에서 floor_refl 등으로 전달.
 */
void apply_depth_mirror_person_reflection(cv::Mat& io_bgr,
                                          const cv::Mat& color_bgr,
                                          const cv::Mat& depth_u16,
                                          const cv::Mat& alpha_f32_hw,
                                          const Params& p,
                                          float strength)
{
    if (io_bgr.empty() || color_bgr.empty() || strength < 1e-4f || !p.cam_intrin_valid)
        return;
    if (depth_u16.empty() || depth_u16.type() != CV_16U)
        return;

    cv::Mat a = alpha_f32_hw;
    ensure_same_size(color_bgr, a, cv::INTER_LINEAR);
    cv::Mat d = depth_u16;
    ensure_same_size(color_bgr, d, cv::INTER_NEAREST);
    cv::Mat c = color_bgr;
    if (io_bgr.size() != color_bgr.size())
        return;

    const int H = color_bgr.rows;
    const int W = color_bgr.cols;
    cv::Mat floor_mask;
    build_floor_mask_u8(H, W, d, a, p, floor_mask);
    const Plane3 pl = estimate_floor_plane_ransac_cam(d, a, p);
    if (!pl.valid)
        return;

    cv::Vec3f n = pl.n;
    float dplane = pl.d;
    // 바닥 위쪽(카메라 쪽)으로 법선이 향하도록: 하단 샘플에서 n·P+d > 0 이면 (n,d) 동시 반전
    {
        int cnt = 0;
        double s = 0.0;
        const int y0 = static_cast<int>(H * 0.72f);
        for (int y = y0; y < H && cnt < 120; y += 3)
        {
            const uint16_t* dr = d.ptr<uint16_t>(y);
            const uint8_t* fm = floor_mask.ptr<uint8_t>(y);
            for (int x = 0; x < W && cnt < 120; x += 5)
            {
                if (!fm[x])
                    continue;
                const uint16_t z = dr[x];
                if (z < 250 || z > 8000)
                    continue;
                cv::Vec3f P;
                if (!backproject_m(x, y, z, p, P))
                    continue;
                s += static_cast<double>(n[0] * P[0] + n[1] * P[1] + n[2] * P[2] + dplane);
                ++cnt;
            }
        }
        if (cnt > 8 && s < 0.0)
        {
            n = -n;
            dplane = -dplane;
        }
    }

    cv::Mat acc0(H, W, CV_32F, cv::Scalar(0));
    cv::Mat acc1(H, W, CV_32F, cv::Scalar(0));
    cv::Mat acc2(H, W, CV_32F, cv::Scalar(0));
    cv::Mat accw(H, W, CV_32F, cv::Scalar(0));

    const int step = 1;
    for (int y = 0; y < H; y += step)
    {
        const uint16_t* dr = d.ptr<uint16_t>(y);
        const float* ar = a.ptr<float>(y);
        const cv::Vec3b* cp = c.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; x += step)
        {
            const float al = ar[x];
            // 마네킹·소품 등 RVM 알파가 낮을 때도 반사 소스로 쓰기 위해 하한 완화
            if (al < 0.04f)
                continue;
            uint16_t z = dr[x];
            if (z <= 250)
            {
                bool found = false;
                for (int r = 1; r <= 3 && !found; ++r)
                {
                    for (int yy = std::max(0, y - r); yy <= std::min(H - 1, y + r) && !found; ++yy)
                    {
                        const uint16_t* dr2 = d.ptr<uint16_t>(yy);
                        for (int xx = std::max(0, x - r); xx <= std::min(W - 1, x + r); ++xx)
                        {
                            const uint16_t zz = dr2[xx];
                            if (zz > 250 && zz < 8000)
                            {
                                z = zz;
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if (!found)
                    continue;
            }
            cv::Vec3f Q;
            if (!backproject_m(x, y, z, p, Q))
                continue;
            const float plane_dist = n[0] * Q[0] + n[1] * Q[1] + n[2] * Q[2] + dplane;
            cv::Vec3f Qv = Q - 2.0f * n * plane_dist;
            if (Qv[2] < 0.03f)
                continue;
            const float uf = p.cam_fx * (Qv[0] / Qv[2]) + p.cam_cx;
            const float vf = p.cam_fy * (Qv[1] / Qv[2]) + p.cam_cy;
            const int ui = static_cast<int>(std::lround(uf));
            const int vi = static_cast<int>(std::lround(vf));
            if (ui < 0 || ui >= W || vi < 0 || vi >= H)
                continue;
            if (!floor_mask.ptr<uint8_t>(vi)[ui])
                continue;
            const float wgt = al * al * (0.28f + 0.72f * std::clamp(std::abs(plane_dist) * 4.0f, 0.0f, 1.0f));
            float* p0 = acc0.ptr<float>(vi);
            float* p1 = acc1.ptr<float>(vi);
            float* p2 = acc2.ptr<float>(vi);
            float* pw = accw.ptr<float>(vi);
            p0[ui] += static_cast<float>(cp[x][0]) * wgt;
            p1[ui] += static_cast<float>(cp[x][1]) * wgt;
            p2[ui] += static_cast<float>(cp[x][2]) * wgt;
            pw[ui] += wgt;
        }
    }

    cv::GaussianBlur(acc0, acc0, cv::Size(0, 0), 2.2, 2.2);
    cv::GaussianBlur(acc1, acc1, cv::Size(0, 0), 2.2, 2.2);
    cv::GaussianBlur(acc2, acc2, cv::Size(0, 0), 2.2, 2.2);
    cv::GaussianBlur(accw, accw, cv::Size(0, 0), 2.2, 2.2);

    const float kBlend = std::clamp(strength * 1.85f, 0.0f, 1.0f);
    for (int y = 0; y < H; ++y)
    {
        const uint8_t* fm = floor_mask.ptr<uint8_t>(y);
        const float* ap = a.ptr<float>(y);
        const float* p0 = acc0.ptr<float>(y);
        const float* p1 = acc1.ptr<float>(y);
        const float* p2 = acc2.ptr<float>(y);
        const float* pw = accw.ptr<float>(y);
        cv::Vec3b* op = io_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x)
        {
            if (!fm[x])
                continue;
            // 형태학 확장으로 마스크에 섞인 경계는 스킵, 바닥 본체는 대부분 al 낮음
            if (ap[x] > 0.58f)
                continue;
            const float w = pw[x];
            if (w < 1e-2f)
                continue;
            const float inv = 1.0f / w;
            float rr = p0[x] * inv;
            float gg = p1[x] * inv;
            float bb = p2[x] * inv;
            // 물결 느낌: 살짝 푸른 틴 + 채도 약화
            const float lum = 0.114f * bb + 0.587f * gg + 0.299f * rr;
            rr = std::clamp(lum + (rr - lum) * 0.82f + 3.f, 0.f, 255.f);
            gg = std::clamp(lum + (gg - lum) * 0.82f + 5.f, 0.f, 255.f);
            bb = std::clamp(lum + (bb - lum) * 0.82f + 10.f, 0.f, 255.f);
            const float yfade = std::clamp(static_cast<float>(H - 1 - y) / (0.72f * H), 0.0f, 1.0f);
            const float fres = 0.38f + 0.62f * (1.0f - yfade);
            const float wn = std::sqrt(std::max(w, 0.f));
            float mix = kBlend * fres * std::min(1.0f, wn * 0.30f);
            mix = std::clamp(mix, 0.0f, 0.94f);
            op[x][0] = static_cast<uint8_t>(std::clamp(op[x][0] * (1.f - mix) + bb * mix, 0.f, 255.f));
            op[x][1] = static_cast<uint8_t>(std::clamp(op[x][1] * (1.f - mix) + gg * mix, 0.f, 255.f));
            op[x][2] = static_cast<uint8_t>(std::clamp(op[x][2] * (1.f - mix) + rr * mix, 0.f, 255.f));
        }
    }
}

/** RVM ONNX 로드. 가능하면 ONNX Runtime 세션, 아니면 OpenCV DNN 네트워크. */
bool try_load_onnx(Engine& e, const std::string& path)
{
    e.net_loaded = false;
    e.ort_loaded = false;
    e.backend_name.clear();
    e.last_error.clear();
    e.net = cv::dnn::Net();
    e.rec.assign(4, cv::Mat());
    if (path.empty())
    {
        e.last_error = "RVM ONNX path is empty.";
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f.good())
    {
        e.last_error = "RVM ONNX file not found: " + path;
        return false;
    }
    f.close();
#ifdef RVM_USE_ONNXRUNTIME
    try
    {
        auto env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rvm");
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        opts.SetIntraOpNumThreads(1);
        const std::wstring wpath = utf8_to_wide(path);
        if (wpath.empty())
            throw std::runtime_error("Path conversion to wide string failed.");
        auto sess = std::make_unique<Ort::Session>(*env, wpath.c_str(), opts);

        static constexpr std::array<const char*, 6> kInPref = {"src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio"};
        static constexpr std::array<const char*, 6> kOutPref = {"pha", "fgr", "r1o", "r2o", "r3o", "r4o"};
        if (!fetch_node_names(*sess, true, e.ort_input_names, kInPref) ||
            !fetch_node_names(*sess, false, e.ort_output_names, kOutPref))
        {
            throw std::runtime_error("RVM ORT IO names not available");
        }

        e.ort_env = std::move(env);
        e.ort_session = std::move(sess);
        for (int i = 0; i < 4; ++i)
        {
            e.ort_rec[i].assign(1, 0.0f);
            e.ort_rec_shape[i] = {1, 1, 1, 1};
        }
        e.rvm_input = cv::Size(512, 288);
        e.rvm_dtype = CV_32F;
        e.ort_loaded = true;
        e.net_loaded = true;
        e.backend_name = "ONNX Runtime";
        return true;
    }
    catch (const std::exception& ex)
    {
        e.last_error = std::string("ONNX Runtime load failed: ") + ex.what();
        e.ort_loaded = false;
        e.ort_session.reset();
        e.ort_env.reset();
    }
    catch (...)
    {
        e.last_error = "ONNX Runtime load failed: unknown exception.";
        e.ort_loaded = false;
        e.ort_session.reset();
        e.ort_env.reset();
    }
#endif
    try
    {
        e.net = cv::dnn::readNetFromONNX(path);
        if (e.net.empty())
            return false;
        e.net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        e.net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        e.rec.assign(4, cv::Mat());
        e.rvm_input = cv::Size(512, 288);
        e.rvm_dtype = CV_32F;
        e.net_loaded = true;
        e.backend_name = "OpenCV DNN";
        e.last_error.clear();
        return true;
    }
    catch (const std::exception& ex)
    {
        e.net_loaded = false;
        if (e.last_error.empty())
            e.last_error = std::string("OpenCV DNN load failed: ") + ex.what();
        else
            e.last_error += " | OpenCV DNN load failed: " + std::string(ex.what());
        return false;
    }
    catch (...)
    {
        e.net_loaded = false;
        if (e.last_error.empty())
            e.last_error = "OpenCV DNN load failed: unknown exception.";
        else
            e.last_error += " | OpenCV DNN load failed: unknown exception.";
        return false;
    }
}

/** RVM 한 프레임 추론: alpha_f32_hw를 컬러와 같은 크기 CV_32F(0..1)로 채움. */
bool infer_alpha(Engine& e, const cv::Mat& color_bgr, cv::Mat& alpha_f32_hw, float downsample_ratio)
{
    if (!e.net_loaded || color_bgr.empty())
        return false;
#ifdef RVM_USE_ONNXRUNTIME
    if (e.ort_loaded && e.ort_session)
    {
        try
        {
            cv::Mat blob;
            cv::dnn::blobFromImage(color_bgr,
                                   blob,
                                   1.0 / 255.0,
                                   e.rvm_input,
                                   cv::Scalar(),
                                   true,
                                   false,
                                   CV_32F);
            const int in_h = blob.size[2];
            const int in_w = blob.size[3];
            const std::array<int64_t, 4> src_shape = {1, 3, static_cast<int64_t>(in_h), static_cast<int64_t>(in_w)};
            std::array<Ort::Value, 6> inputs = {
                Ort::Value(nullptr), Ort::Value(nullptr), Ort::Value(nullptr),
                Ort::Value(nullptr), Ort::Value(nullptr), Ort::Value(nullptr)
            };
            Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            inputs[0] = Ort::Value::CreateTensor<float>(mem,
                                                        blob.ptr<float>(),
                                                        static_cast<size_t>(blob.total()),
                                                        src_shape.data(),
                                                        src_shape.size());
            for (int i = 0; i < 4; ++i)
            {
                if (e.ort_rec_shape[i].empty())
                    e.ort_rec_shape[i] = {1, 1, 1, 1};
                const size_t need = std::accumulate(e.ort_rec_shape[i].begin(),
                                                    e.ort_rec_shape[i].end(),
                                                    static_cast<size_t>(1),
                                                    [](size_t a, int64_t b) { return a * static_cast<size_t>(std::max<int64_t>(1, b)); });
                if (e.ort_rec[i].size() != need)
                    e.ort_rec[i].assign(need, 0.0f);
                inputs[1 + i] = Ort::Value::CreateTensor<float>(mem,
                                                                e.ort_rec[i].data(),
                                                                e.ort_rec[i].size(),
                                                                e.ort_rec_shape[i].data(),
                                                                e.ort_rec_shape[i].size());
            }
            float ds_v = downsample_ratio;
            const std::array<int64_t, 1> ds_shape = {1};
            inputs[5] = Ort::Value::CreateTensor<float>(mem, &ds_v, 1, ds_shape.data(), ds_shape.size());

            std::array<const char*, 6> in_names{};
            std::array<const char*, 6> out_names{};
            for (int i = 0; i < 6; ++i)
            {
                in_names[i] = e.ort_input_names[i].c_str();
                out_names[i] = e.ort_output_names[i].c_str();
            }
            auto outs = e.ort_session->Run(Ort::RunOptions{nullptr},
                                           in_names.data(),
                                           inputs.data(),
                                           inputs.size(),
                                           out_names.data(),
                                           out_names.size());
            if (outs.size() < 6 || !outs[0].IsTensor())
                return false;

            {
                auto info = outs[0].GetTensorTypeAndShapeInfo();
                const auto shape = info.GetShape();
                if (shape.size() < 4)
                    return false;
                const int h = static_cast<int>(shape[shape.size() - 2]);
                const int w = static_cast<int>(shape[shape.size() - 1]);
                const float* p = outs[0].GetTensorData<float>();
                cv::Mat hw(h, w, CV_32F);
                std::memcpy(hw.data, p, static_cast<size_t>(h * w) * sizeof(float));
                cv::resize(hw, alpha_f32_hw, color_bgr.size(), 0, 0, cv::INTER_LINEAR);
            }

            for (int i = 0; i < 4; ++i)
            {
                if (!outs[2 + i].IsTensor())
                    continue;
                auto info = outs[2 + i].GetTensorTypeAndShapeInfo();
                e.ort_rec_shape[i] = info.GetShape();
                const float* p = outs[2 + i].GetTensorData<float>();
                const size_t n = static_cast<size_t>(info.GetElementCount());
                e.ort_rec[i].assign(p, p + n);
            }
            return true;
        }
        catch (...)
        {
            for (int i = 0; i < 4; ++i)
            {
                e.ort_rec[i].assign(1, 0.0f);
                e.ort_rec_shape[i] = {1, 1, 1, 1};
            }
            // ORT 런타임 실패 시 OpenCV 경로로 한 번 더 시도
        }
    }
#endif
    try
    {
        cv::Mat blob;
        cv::dnn::blobFromImage(color_bgr,
                               blob,
                               1.0 / 255.0,
                               e.rvm_input,
                               cv::Scalar(),
                               true,
                               false,
                               CV_32F);
        const int dtype = e.rvm_dtype;
        int z4[] = {1, 1, 1, 1};
        for (int i = 0; i < 4; ++i)
        {
            if (e.rec[i].empty() || e.rec[i].type() != dtype || e.rec[i].dims != 4)
                e.rec[i] = cv::Mat(4, z4, dtype, cv::Scalar::all(0));
        }
        cv::Mat ds = (cv::Mat_<float>(1, 1) << downsample_ratio);

        e.net.setInput(blob, "src");
        e.net.setInput(e.rec[0], "r1i");
        e.net.setInput(e.rec[1], "r2i");
        e.net.setInput(e.rec[2], "r3i");
        e.net.setInput(e.rec[3], "r4i");
        e.net.setInput(ds, "downsample_ratio");

        std::vector<cv::String> outn = {"pha", "fgr", "r1o", "r2o", "r3o", "r4o"};
        std::vector<cv::Mat> outs;
        e.net.forward(outs, outn);
        if (outs.size() < 6)
            return false;
        cv::Mat pha = outs[0];
        cv::Mat hw = nchw_pha_to_hw(pha);
        if (hw.empty())
            return false;
        cv::resize(hw, alpha_f32_hw, color_bgr.size(), 0, 0, cv::INTER_LINEAR);
        e.rec[0] = outs[2];
        e.rec[1] = outs[3];
        e.rec[2] = outs[4];
        e.rec[3] = outs[5];
        return true;
    }
    catch (...)
    {
        e.rec.assign(4, cv::Mat());
        return false;
    }
}

/** RVM 실패 시: 화면 하단 깊이 통계로 전경(사람) 알파 근사. */
void infer_alpha_depth_fallback(const cv::Mat& color_bgr, const cv::Mat& depth_u16, cv::Mat& alpha_f32_hw)
{
    alpha_f32_hw.create(color_bgr.size(), CV_32F);
    alpha_f32_hw.setTo(0.f);
    if (color_bgr.empty() || depth_u16.empty() || depth_u16.type() != CV_16U)
        return;

    const int H = depth_u16.rows;
    const int W = depth_u16.cols;
    const int y0 = static_cast<int>(H * 0.72f);
    std::vector<uint16_t> samples;
    samples.reserve(static_cast<size_t>(W) * 8);
    for (int y = y0; y < H; ++y)
    {
        const uint16_t* row = depth_u16.ptr<uint16_t>(y);
        for (int x = 0; x < W; ++x)
        {
            const uint16_t d = row[x];
            if (d > 200 && d < 8000)
                samples.push_back(d);
        }
    }
    if (samples.empty())
        return;
    std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());
    const float floor_mm = static_cast<float>(samples[samples.size() / 2]);

    // 바닥 중앙값 하나만 쓰면 더비 인형/기울어진 바닥에서 누락이 발생할 수 있어
    // 로컬 바닥 깊이(평면)를 우선 사용하고, 실패 시 중앙값으로 폴백.
    cv::Mat a_zero(depth_u16.size(), CV_32F, cv::Scalar(0));
    const PlaneZ floor_plane = estimate_floor_plane_ransac(depth_u16, a_zero);
    const bool use_plane = floor_plane.valid;

    for (int y = 0; y < H; ++y)
    {
        const uint16_t* dr = depth_u16.ptr<uint16_t>(y);
        float* ar = alpha_f32_hw.ptr<float>(y);
        const float row_w = static_cast<float>(y) / static_cast<float>(std::max(1, H - 1));
        for (int x = 0; x < W; ++x)
        {
            const uint16_t d = dr[x];
            if (d < 200 || d > 8000)
            {
                ar[x] = 0.f;
                continue;
            }
            const float dd = static_cast<float>(d);
            const float floor_local_mm =
                use_plane ? (floor_plane.a * static_cast<float>(x) + floor_plane.b * static_cast<float>(y) + floor_plane.c)
                          : floor_mm;
            const float closer = floor_local_mm - dd;
            const float cx = std::abs((static_cast<float>(x) / static_cast<float>(W)) - 0.5f);
            float a = 0.f;
            // 더비 인형/얇은 팔다리도 잡히도록 임계값 완화
            if (closer > 20.f && closer < 3600.f)
                a = std::clamp((closer - 20.f) / 700.f, 0.f, 1.f);
            a *= (0.55f + 0.45f * row_w);
            a *= (1.0f - 0.45f * cx * cx);
            ar[x] = std::clamp(a, 0.f, 1.f);
        }
    }
    cv::Mat tmp;
    cv::GaussianBlur(alpha_f32_hw, tmp, cv::Size(0, 0), 3.0);
    alpha_f32_hw = tmp;
    cv::morphologyEx(alpha_f32_hw, tmp, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
    alpha_f32_hw = tmp;

    // 벽/배경 누설 억제: "하단 시드와 연결된 인체 성분"만 유지
    cv::Mat a_bin;
    cv::threshold(alpha_f32_hw, a_bin, 0.14, 255.0, cv::THRESH_BINARY);
    a_bin.convertTo(a_bin, CV_8U);
    // 얇은 연결(벽으로 새는 브리지)을 먼저 끊고 컴포넌트 선택
    cv::erode(a_bin, a_bin, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    cv::Mat labels, stats, cents;
    const int nlabels = cv::connectedComponentsWithStats(a_bin, labels, stats, cents, 8, CV_32S);
    if (nlabels > 1)
    {
        int best_label = 0;
        float best_score = -1.0f;
        const int y_seed0 = static_cast<int>(H * 0.58f);
        for (int y = y_seed0; y < H; ++y)
        {
            const int* lp = labels.ptr<int>(y);
            const float* ap = alpha_f32_hw.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                const int id = lp[x];
                if (id <= 0)
                    continue;
                // 하단/중앙에 가까운 고알파 픽셀을 시드로 우선
                const float center_pen = std::abs(x - 0.5f * W) / static_cast<float>(W);
                const float score = ap[x] * (1.0f - 0.35f * center_pen);
                if (score > best_score)
                {
                    best_score = score;
                    best_label = id;
                }
            }
        }
        if (best_label <= 0)
        {
            // 시드 실패 시 가장 큰 컴포넌트로 폴백
            int best_area = 0;
            for (int id = 1; id < nlabels; ++id)
            {
                const int area = stats.at<int>(id, cv::CC_STAT_AREA);
                if (area > best_area)
                {
                    best_area = area;
                    best_label = id;
                }
            }
        }

        cv::Mat keep(alpha_f32_hw.size(), CV_32F, cv::Scalar(0));
        for (int y = 0; y < H; ++y)
        {
            const int* lp = labels.ptr<int>(y);
            float* kp = keep.ptr<float>(y);
            for (int x = 0; x < W; ++x)
                if (lp[x] == best_label)
                    kp[x] = 1.0f;
        }
        cv::dilate(keep, keep, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        alpha_f32_hw = alpha_f32_hw.mul(keep);
    }
}

/** Kinect 변환: 컬러를 depth 카메라 기준으로 정렬한 BGR + 깊이(CV_16U) */
bool color_depth_aligned_mats(const k4a::transformation& tr,
                              const k4a::image& depth,
                              const k4a::image& color,
                              cv::Mat& out_bgr,
                              cv::Mat& out_depth_u16)
{
    if (!depth || !color)
        return false;
    try
    {
        k4a::image c2d = tr.color_image_to_depth_camera(depth, color);
        if (!c2d || c2d.get_format() != K4A_IMAGE_FORMAT_COLOR_BGRA32)
            return false;
        const int w = c2d.get_width_pixels();
        const int h = c2d.get_height_pixels();
        cv::Mat bgra(h, w, CV_8UC4, (void*)c2d.get_buffer(), c2d.get_stride_bytes());
        cv::cvtColor(bgra, out_bgr, cv::COLOR_BGRA2BGR);
        const int dw = depth.get_width_pixels();
        const int dh = depth.get_height_pixels();
        cv::Mat d16(dh, dw, CV_16UC1, (void*)depth.get_buffer(), depth.get_stride_bytes());
        if (out_bgr.size() != d16.size())
            return false;
        d16.copyTo(out_depth_u16);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

/** compose_floor_with_fx와 동일한 2D 수평 거울 반사(반사면 y = reflection_plane_y). */
static void apply_vertical_mirror_floor_reflect_work(cv::Mat& work,
                                                     const cv::Mat& color_bgr,
                                                     const cv::Mat& alpha_f32_hw,
                                                     const cv::Mat& floor_mask,
                                                     const Params& p)
{
    if (work.empty() || !p.reflection_on || p.reflection_strength <= 1e-4f)
        return;
    const int H = work.rows;
    const int W = work.cols;
    const int y_mirror = std::clamp(static_cast<int>(p.reflection_plane_y * static_cast<float>(H)), 0, H - 1);
    const float denom_h = static_cast<float>(std::max(1, H - 1));
    for (int y = 0; y < H; ++y)
    {
        const uint8_t* fm = floor_mask.ptr<uint8_t>(y);
        cv::Vec3b* wp = work.ptr<cv::Vec3b>(y);
        const int ys = std::clamp(2 * y_mirror - y, 0, H - 1);
        const cv::Vec3b* cp = color_bgr.ptr<cv::Vec3b>(ys);
        const float* ap = alpha_f32_hw.ptr<float>(ys);
        const float yfac = static_cast<float>(y) / denom_h;
        const float mirror_fade =
            std::clamp(static_cast<float>(y - y_mirror) / static_cast<float>(std::max(1, H - 1 - y_mirror)), 0.f, 1.f);
        const float wref =
            p.reflection_strength * (0.40f + 0.60f * yfac) * (0.35f + 0.65f * mirror_fade);
        for (int x = 0; x < W; ++x)
        {
            if (!fm[x])
                continue;
            const float al = std::clamp(ap[x], 0.f, 1.f);
            if (al < 1e-3f)
                continue;
            const float w = std::clamp(wref * al, 0.f, 1.f);
            const cv::Vec3b src = cp[x];
            wp[x][0] = static_cast<uint8_t>(std::clamp(wp[x][0] * (1.f - w) + src[0] * w, 0.f, 255.f));
            wp[x][1] = static_cast<uint8_t>(std::clamp(wp[x][1] * (1.f - w) + src[1] * w, 0.f, 255.f));
            wp[x][2] = static_cast<uint8_t>(std::clamp(wp[x][2] * (1.f - w) + src[2] * w, 0.f, 255.f));
        }
    }
}

void apply_compose_style_floor_reflection(cv::Mat& io_bgr,
                                          const cv::Mat& color_bgr,
                                          const cv::Mat& depth_u16,
                                          const cv::Mat& alpha_f32_hw,
                                          const Params& p)
{
    if (io_bgr.empty() || color_bgr.empty() || !p.reflection_on || p.reflection_strength <= 1e-4f)
        return;
    cv::Mat a = alpha_f32_hw;
    ensure_same_size(color_bgr, a, cv::INTER_LINEAR);
    cv::Mat d = depth_u16;
    if (!d.empty())
        ensure_same_size(color_bgr, d, cv::INTER_NEAREST);
    if (io_bgr.size() != color_bgr.size())
        return;
    const int H = color_bgr.rows;
    const int W = color_bgr.cols;
    cv::Mat floor_mask;
    build_floor_mask_u8(H, W, d, a, p, floor_mask);
    apply_vertical_mirror_floor_reflect_work(io_bgr, color_bgr, a, floor_mask, p);
}

/**
 * 바닥 합성 한 번에: 접촉 그림자 → 거울 반사 → 3D 캐스트 그림자 → 알파로 원본 합성.
 * 선택적으로 바닥 격자(녹색) 오버레이.
 */
cv::Mat compose_floor_with_fx(const cv::Mat& color_bgr,
                               const cv::Mat& depth_u16,
                               const cv::Mat& alpha_f32_hw,
                               const Params& p)
{
    if (color_bgr.empty())
        return {};
    cv::Mat a = alpha_f32_hw;
    ensure_same_size(color_bgr, a, cv::INTER_LINEAR);
    cv::Mat d = depth_u16;
    if (!d.empty())
        ensure_same_size(color_bgr, d, cv::INTER_NEAREST);

    const int H = color_bgr.rows;
    const int W = color_bgr.cols;
    cv::Mat work;
    color_bgr.copyTo(work);

    cv::Mat floor_mask;
    build_floor_mask_u8(H, W, d, a, p, floor_mask);

    /* 사람 무게중심(cx,cy) — 바닥 접촉 그림자 감쇠 거리용 */
    double sumx = 0, sumy = 0, wa = 0;
    for (int y = 0; y < H; ++y)
    {
        const float* ap = a.ptr<float>(y);
        for (int x = 0; x < W; ++x)
        {
            const float al = ap[x];
            if (al > 0.25f)
            {
                sumx += x * al;
                sumy += y * al;
                wa += al;
            }
        }
    }
    const float cx = wa > 1e-3 ? static_cast<float>(sumx / wa) : W * 0.5f;
    const float cy = wa > 1e-3 ? static_cast<float>(sumy / wa) : H * 0.35f;
    const float sigma = static_cast<float>(std::max(W, H)) * 0.22f;
    /* 발 위치: 화면 아래에서 알파 높은 행을 찾아 발 쪽 x,y 추정 */
    float foot_x = cx;
    float foot_y = static_cast<float>(H - 1);
    {
        double sfx = 0.0;
        double sw = 0.0;
        for (int y = H - 1; y >= 0; --y)
        {
            const float* ap = a.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                if (ap[x] > 0.45f)
                {
                    sfx += static_cast<double>(x);
                    sw += 1.0;
                    foot_y = static_cast<float>(y);
                }
            }
            if (sw > 18.0)
                break;
        }
        if (sw > 1.0)
            foot_x = static_cast<float>(sfx / sw);
    }

    for (int y = 0; y < H; ++y)
    {
        cv::Vec3b* wp = work.ptr<cv::Vec3b>(y);
        const uint8_t* fm = floor_mask.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x)
        {
            if (!fm[x])
                continue;
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - static_cast<float>(H - 1);
            const float dist2 = dx * dx + dy * dy * 0.35f;
            const float sh = std::exp(-dist2 / (2.f * sigma * sigma));
            const float fdx = static_cast<float>(x) - foot_x;
            const float fdy = static_cast<float>(y) - foot_y;
            const float foot_contact = std::exp(-(fdx * fdx + fdy * fdy * 0.8f) /
                                                (2.f * (sigma * 0.18f) * (sigma * 0.18f)));
            const float far_fade = 1.0f - 0.45f * std::clamp((static_cast<float>(H - 1 - y)) / (0.65f * H), 0.0f, 1.0f);
            /* shadow_strength: 전역 가우시안(sh) + 발 주변(foot_contact) */
            const float dark = 1.f - p.shadow_strength * (0.75f * sh + 0.55f * foot_contact) * far_fade;
            wp[x][0] = static_cast<uint8_t>(std::clamp(wp[x][0] * dark, 0.f, 255.f));
            wp[x][1] = static_cast<uint8_t>(std::clamp(wp[x][1] * dark, 0.f, 255.f));
            wp[x][2] = static_cast<uint8_t>(std::clamp(wp[x][2] * dark, 0.f, 255.f));
        }
    }

    // 바닥 반사: 수평선 y=y_mirror 기준으로 원본 행 ys=2*y_mirror-y 색을 블렌딩
    apply_vertical_mirror_floor_reflect_work(work, color_bgr, a, floor_mask, p);

    // 3D 캐스트 그림자: 깊이 + 바닥 평면 + 태양 방향 L
    if (p.cast_shadow_alpha > 1e-4f && p.cam_intrin_valid)
    {
        cv::Mat sh_m = build_cast_shadow_sh_m_normalized(d, a, floor_mask, p, H, W);
        if (!sh_m.empty())
        {
            const float ca = std::clamp(p.cast_shadow_alpha, 0.f, 1.f);
            for (int y = 0; y < H; ++y)
            {
                const uint8_t* fm = floor_mask.ptr<uint8_t>(y);
                const float* sv = sh_m.ptr<float>(y);
                cv::Vec3b* wp = work.ptr<cv::Vec3b>(y);
                for (int x = 0; x < W; ++x)
                {
                    if (!fm[x])
                        continue;
                    const float shv = std::clamp(sv[x], 0.f, 1.f);
                    if (shv < 1e-4f)
                        continue;
                    const float far_fade =
                        1.0f - 0.28f * std::clamp((static_cast<float>(H - 1 - y)) / (0.70f * H), 0.0f, 1.0f);
                    /* sh_m: 0=밝음 1=그림자 진함 → dark로 곱해 어둡게 */
                    const float dark = 1.f - shv * ca * far_fade;
                    wp[x][0] = static_cast<uint8_t>(std::clamp(wp[x][0] * dark, 0.f, 255.f));
                    wp[x][1] = static_cast<uint8_t>(std::clamp(wp[x][1] * dark, 0.f, 255.f));
                    wp[x][2] = static_cast<uint8_t>(std::clamp(wp[x][2] * dark, 0.f, 255.f));
                }
            }
        }
    }

    /* 알파 블렌드: 사람은 원본 컬러, 배경은 work(바닥 효과 적용) */
    cv::Mat out(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y)
    {
        const cv::Vec3b* cp = color_bgr.ptr<cv::Vec3b>(y);
        const cv::Vec3b* wp = work.ptr<cv::Vec3b>(y);
        const float* ap = a.ptr<float>(y);
        cv::Vec3b* op = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x)
        {
            const float al = std::clamp(ap[x], 0.f, 1.f);
            op[x][0] = static_cast<uint8_t>(cp[x][0] * al + wp[x][0] * (1.f - al));
            op[x][1] = static_cast<uint8_t>(cp[x][1] * al + wp[x][1] * (1.f - al));
            op[x][2] = static_cast<uint8_t>(cp[x][2] * al + wp[x][2] * (1.f - al));
        }
    }

    if (p.floor_grid_show && p.floor_grid_spacing >= 4)
    {
        const int sp = std::clamp(p.floor_grid_spacing, 4, 200);
        const cv::Vec3b green(0, 255, 0);
        const float kGridAlpha = 0.68f; /* 1.0이면 격자가 그림자를 가림 */
        auto blend_grid = [&](cv::Vec3b& px) {
            px[0] = static_cast<uint8_t>(std::clamp(px[0] * (1.f - kGridAlpha) + green[0] * kGridAlpha, 0.f, 255.f));
            px[1] = static_cast<uint8_t>(std::clamp(px[1] * (1.f - kGridAlpha) + green[1] * kGridAlpha, 0.f, 255.f));
            px[2] = static_cast<uint8_t>(std::clamp(px[2] * (1.f - kGridAlpha) + green[2] * kGridAlpha, 0.f, 255.f));
        };
        const int y_grid0 = static_cast<int>(H * 0.55f);
        for (int y = y_grid0; y < H; y += sp)
        {
            const uint8_t* fm = floor_mask.ptr<uint8_t>(y);
            cv::Vec3b* op = out.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; ++x)
                if (fm[x])
                    blend_grid(op[x]);
        }
        for (int x = 0; x < W; x += sp)
        {
            for (int y = y_grid0; y < H; ++y)
                if (floor_mask.at<uint8_t>(y, x))
                    blend_grid(out.at<cv::Vec3b>(y, x));
        }
    }

    return out;
}

} // namespace rvm_floor
