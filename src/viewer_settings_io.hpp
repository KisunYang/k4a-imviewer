#pragma once

#include <map>
#include <string>

namespace viewer_io {

using KVMap = std::map<std::string, std::string>;

bool load_ini(const std::string& path, KVMap& out);
bool save_ini(const std::string& path, const KVMap& in);

std::string default_settings_path();

} // namespace viewer_io
