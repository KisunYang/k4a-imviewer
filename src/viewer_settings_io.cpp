#include "viewer_settings_io.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace viewer_io {

namespace {

std::string trim(std::string s)
{
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
        s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t'))
        ++i;
    return s.substr(i);
}

} // namespace

bool load_ini(const std::string& path, KVMap& out)
{
    out.clear();
    std::ifstream f(path, std::ios::in);
    if (!f.good())
        return false;
    std::string line;
    while (std::getline(f, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';')
            continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::string k = trim(line.substr(0, eq));
        std::string v = trim(line.substr(eq + 1));
        if (!k.empty())
            out[k] = v;
    }
    return true;
}

bool save_ini(const std::string& path, const KVMap& in)
{
    std::error_code ec;
    std::filesystem::path p(path);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path(), ec);
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    if (!f.good())
        return false;
    f << "# k4a_imviewer settings (auto)\n";
    for (const auto& kv : in)
        f << kv.first << '=' << kv.second << '\n';
    return true;
}

std::string default_settings_path()
{
#ifdef _WIN32
    char buf[MAX_PATH];
    if (GetEnvironmentVariableA("APPDATA", buf, sizeof(buf)) > 0)
    {
        std::filesystem::path p(buf);
        p /= "k4a_imviewer";
        std::filesystem::create_directories(p);
        p /= "settings.ini";
        return p.string();
    }
#endif
    const char* home = std::getenv("HOME");
    if (home)
    {
        std::filesystem::path p(home);
        p /= ".k4a_imviewer";
        std::filesystem::create_directories(p);
        p /= "settings.ini";
        return p.string();
    }
    return "k4a_imviewer_settings.ini";
}

} // namespace viewer_io
