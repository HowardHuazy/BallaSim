#include "scene/evimo2/catalogue.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "scene/npo.h"

namespace balla::scene::evimo2 {

namespace fs = std::filesystem;

namespace {

std::string read_text(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::invalid_argument(p.string() + ": cannot open");
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// The length of a Python whitespace character (str.isspace) starting at s[i] in UTF-8, or 0.
size_t space_len(const std::string& s, size_t i) {
  const auto c = static_cast<unsigned char>(s[i]);
  if (c == ' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f)) return 1;
  if (c == 0xc2 && i + 1 < s.size()) {
    const auto d = static_cast<unsigned char>(s[i + 1]);
    return d == 0x85 || d == 0xa0 ? 2 : 0;
  }
  if (c == 0xe1 && i + 2 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0x9a &&
      static_cast<unsigned char>(s[i + 2]) == 0x80) {
    return 3;   // U+1680
  }
  if (c == 0xe2 && i + 2 < s.size()) {
    const auto d = static_cast<unsigned char>(s[i + 1]), e = static_cast<unsigned char>(s[i + 2]);
    if (d == 0x80 && ((e >= 0x80 && e <= 0x8a) || e == 0xa8 || e == 0xa9 || e == 0xaf)) return 3;   // U+2000-200A,2028,2029,202F
    if (d == 0x81 && e == 0x9f) return 3;   // U+205F
  }
  if (c == 0xe3 && i + 2 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0x80 &&
      static_cast<unsigned char>(s[i + 2]) == 0x80) {
    return 3;   // U+3000
  }
  return 0;
}

std::vector<std::string> split_words(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t n;
    while (i < s.size() && (n = space_len(s, i)) > 0) i += n;
    const size_t j = i;
    while (i < s.size() && space_len(s, i) == 0) ++i;
    if (i > j) out.push_back(s.substr(j, i - j));
  }
  return out;
}

std::string strip(const std::string& s) {
  size_t b = 0, e = s.size();
  size_t n;
  while (b < e && (n = space_len(s, b)) > 0) b += n;
  while (e > b) {
    size_t k = e - 1;
    while (k > b && (static_cast<unsigned char>(s[k]) & 0xc0) == 0x80) --k;   // the last character's start
    if (space_len(s, k) == e - k) {
      e = k;
    } else {
      break;
    }
  }
  return s.substr(b, e - b);
}

// str.splitlines() of UTF-8 text.
std::vector<std::string> split_lines(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0, start = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    size_t n = 0;
    if (c == '\n' || c == 0x0b || c == 0x0c || c == 0x1c || c == 0x1d || c == 0x1e) {
      n = 1;
    } else if (c == '\r') {
      n = i + 1 < s.size() && s[i + 1] == '\n' ? 2 : 1;
    } else if (c == 0xc2 && i + 1 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0x85) {
      n = 2;
    } else if (c == 0xe2 && i + 2 < s.size() && static_cast<unsigned char>(s[i + 1]) == 0x80 &&
               (static_cast<unsigned char>(s[i + 2]) == 0xa8 || static_cast<unsigned char>(s[i + 2]) == 0xa9)) {
      n = 3;
    }
    if (n) {
      out.push_back(s.substr(start, i - start));
      i += n;
      start = i;
    } else {
      ++i;
    }
  }
  if (start < s.size()) out.push_back(s.substr(start));
  return out;
}

// Python's float() and int() of one word: an optional sign, underscores only between digits.
std::string digits_only(const std::string& w, const char* what) {
  std::string out;
  for (size_t i = 0; i < w.size(); ++i) {
    if (w[i] == '_') {
      if (i == 0 || i + 1 == w.size() || !std::isdigit(static_cast<unsigned char>(w[i - 1])) ||
          !std::isdigit(static_cast<unsigned char>(w[i + 1]))) {
        throw std::invalid_argument(std::string(what) + ": '" + w + "'");
      }
      continue;
    }
    out += w[i];
  }
  return out;
}

double py_float(const std::string& text) {
  const std::string w = digits_only(strip(text), "could not convert string to float");
  const char* b = w.data();
  const char* e = w.data() + w.size();
  if (b != e && *b == '+') ++b;
  double d = 0;
  const auto r = std::from_chars(b, e, d);
  if (w.empty() || r.ec != std::errc() || r.ptr != e) {
    throw std::invalid_argument("could not convert string to float: '" + text + "'");
  }
  return d;
}

int64_t py_int(const std::string& text) {
  const std::string w = digits_only(strip(text), "invalid literal for int() with base 10");
  const char* b = w.data();
  const char* e = w.data() + w.size();
  if (b != e && *b == '+') ++b;
  int64_t i = 0;
  const auto r = std::from_chars(b, e, i);
  if (w.empty() || r.ec != std::errc() || r.ptr != e) {
    throw std::invalid_argument("invalid literal for int() with base 10: '" + text + "'");
  }
  return i;
}

// --- the vicon.vsk patterns, scanned (no regex engine: its recursion overflowed the stack on long tags) ---------------
// Each scanner returns the matches Python's re.finditer gives for the dataset's pattern, in order, non-overlapping.

// The end of a [^"]+ run starting at b, closed by '"' (its position), or npos when the run is empty or unclosed.
size_t quoted_end(const std::string& t, size_t b) {
  const size_t q = t.find('"', b);
  return q == std::string::npos || q == b ? std::string::npos : q;
}

// <Parameter NAME="([^"]+)" VALUE="([^"]+)"
std::vector<std::pair<std::string, std::string>> scan_parameters(const std::string& t) {
  static const std::string head = "<Parameter NAME=\"", mid = "\" VALUE=\"";
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = 0;
  while ((i = t.find(head, i)) != std::string::npos) {
    const size_t n0 = i + head.size(), n1 = quoted_end(t, n0);
    if (n1 != std::string::npos && t.compare(n1, mid.size(), mid) == 0) {
      const size_t v0 = n1 + mid.size(), v1 = quoted_end(t, v0);
      if (v1 != std::string::npos) {
        out.emplace_back(t.substr(n0, n1 - n0), t.substr(v0, v1 - v0));
        i = v1 + 1;
        continue;
      }
    }
    ++i;
  }
  return out;
}

// The last `key="` at or after b and before the first '>' from b ([^>]* greedy) whose quoted run (and, for `then`,
// the rest of the pattern) matches; returns {key start, run start, run end} or npos.
struct Hit {
  size_t key = std::string::npos, v0 = 0, v1 = 0;
};

Hit last_attr(const std::string& t, size_t b, const std::string& key) {
  size_t e = t.find('>', b);
  if (e == std::string::npos) e = t.size();
  for (size_t p = e; p-- > b;) {
    if (t.compare(p, key.size(), key) != 0) continue;
    const size_t v0 = p + key.size(), v1 = quoted_end(t, v0);
    if (v1 != std::string::npos) return {p, v0, v1};
  }
  return {};
}

// <TAG\s[^>]*NAME="([^"]+)" (Segment) and <TAG\s[^>]*NAME="([^"]+)"[^>]*RADIUS="([^"]+)" (Marker): the greedy [^>]*
// runs take the last NAME (then the last RADIUS after it) that completes the pattern.
std::vector<std::pair<std::string, std::string>> scan_tags(const std::string& t, const std::string& tag, bool radius) {
  static const std::string name_key = "NAME=\"", radius_key = "RADIUS=\"";
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = 0;
  while ((i = t.find(tag, i)) != std::string::npos) {
    const size_t sp = i + tag.size();
    const size_t ws = sp < t.size() ? space_len(t, sp) : 0;
    bool matched = false;
    if (ws > 0) {
      const size_t b = sp + ws;
      size_t e = t.find('>', b);
      if (e == std::string::npos) e = t.size();
      for (size_t p = e; p-- > b && !matched;) {
        if (t.compare(p, name_key.size(), name_key) != 0) continue;
        const size_t n0 = p + name_key.size(), n1 = quoted_end(t, n0);
        if (n1 == std::string::npos) continue;
        if (!radius) {
          out.emplace_back(t.substr(n0, n1 - n0), std::string());
          i = n1 + 1;
          matched = true;
          break;
        }
        const Hit r = last_attr(t, n1 + 1, radius_key);
        if (r.key == std::string::npos) continue;
        out.emplace_back(t.substr(n0, n1 - n0), t.substr(r.v0, r.v1 - r.v0));
        i = r.v1 + 1;
        matched = true;
      }
    }
    if (!matched) ++i;
  }
  return out;
}

}  // namespace

std::vector<std::pair<std::string, std::string>> parse_settings(const std::string& text) {
  std::vector<std::pair<std::string, std::string>> out;
  for (const std::string& line : split_lines(text)) {
    const size_t c = line.find(':');
    if (c == std::string::npos) continue;
    const std::string k = strip(line.substr(0, c)), v = strip(line.substr(c + 1));
    const auto it = std::find_if(out.begin(), out.end(), [&k](const auto& kv) { return kv.first == k; });
    if (it != out.end()) {
      it->second = v;   // a dict keeps the first position and the last value
    } else {
      out.emplace_back(k, v);
    }
  }
  return out;
}

std::vector<rot::V> parse_config(const std::string& text) {
  std::vector<rot::V> out;
  for (const std::string& line : split_lines(text)) {
    const std::vector<std::string> w = split_words(line);
    if (w.size() >= 4) out.push_back({py_float(w[1]), py_float(w[2]), py_float(w[3])});
  }
  return out;
}

void parse_vsk(const std::string& text, const std::string& path, std::vector<rot::V>& markers,
               std::vector<std::string>& names, double& radius) {
  std::map<std::string, double> params;
  for (const auto& [k, v] : scan_parameters(text)) params[k] = py_float(v);
  const std::vector<std::pair<std::string, std::string>> mk = scan_tags(text, "<Marker", true);
  std::vector<std::string> segs;
  for (const auto& m : scan_tags(text, "<Segment", false)) segs.push_back(m.first);
  markers.clear();
  names.clear();
  std::vector<double> radii;
  for (const auto& [name, rad] : mk) {
    std::string key;
    for (const std::string& seg : segs) {
      if (params.count(seg + "_" + name + "_x")) {
        key = seg + "_" + name;
        break;
      }
    }
    if (key.empty()) continue;
    auto get = [&params, &key](const char* axis) {
      const auto it = params.find(key + axis);
      if (it == params.end()) throw std::invalid_argument("'" + key + axis + "'");   // the Python's KeyError
      return it->second;
    };
    markers.push_back({get("_x"), get("_y"), get("_z")});
    names.push_back(name);
    radii.push_back(py_float(rad));
  }
  if (markers.empty()) throw std::invalid_argument(path + ": no marker positions");
  radius = npo::median(radii);
}

std::vector<std::string> object_folders(const std::string& base) {
  if (!fs::is_directory(base)) {
    throw CatalogueMissing(base + ": no EVIMO object catalogue (scene.evimo2.objects_root)");
  }
  std::vector<std::string> out;
  for (const auto& e : fs::directory_iterator(base)) {
    if (fs::exists(e.path() / "settings.txt")) out.push_back(e.path().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> catalogue_files(const std::string& base) {
  std::vector<std::string> out;
  for (const std::string& folder : object_folders(base)) {
    for (const char* f : {"settings.txt", "config.txt", "vicon.vsk"}) {
      const fs::path p = fs::path(folder) / f;
      if (fs::exists(p)) out.push_back(p.string());
    }
  }
  return out;
}

std::vector<CatalogueEntry> catalogue(const std::string& base) {
  static const std::map<std::string, double> units = {{"mm", 1e-3}, {"cm", 1e-2}, {"m", 1.0}};
  std::vector<CatalogueEntry> out;
  for (const std::string& folder : object_folders(base)) {
    const fs::path dir(folder);
    const auto s = parse_settings(read_text(dir / "settings.txt"));
    auto get = [&s](const std::string& k) -> const std::string* {
      for (const auto& [key, v] : s) {
        if (key == k) return &v;
      }
      return nullptr;
    };
    if (!get("id") || !fs::exists(dir / "config.txt") || !fs::exists(dir / "vicon.vsk")) continue;
    CatalogueEntry e;
    parse_vsk(read_text(dir / "vicon.vsk"), (dir / "vicon.vsk").string(), e.vsk_markers_mm, e.vsk_names,
              e.marker_radius_mm);
    e.id = py_int(*get("id"));
    e.name = get("name") ? *get("name") : dir.filename().string();
    e.folder = folder;
    const std::string unit = get("unit") ? *get("unit") : "m";
    const auto u = units.find(unit);
    if (u == units.end()) throw std::invalid_argument("'" + unit + "'");   // the Python's KeyError
    e.unit_scale = u->second;
    e.mesh_path = (dir / (get("mesh") ? *get("mesh") : std::string("model.ply"))).string();
    e.config_markers = parse_config(read_text(dir / "config.txt"));
    e.settings = s;
    for (const CatalogueEntry& o : out) {
      if (o.id == e.id) {
        throw std::invalid_argument("duplicate EVIMO object id " + std::to_string(e.id) + ": " + o.folder + " and " +
                                    folder);
      }
    }
    out.push_back(std::move(e));
  }
  return out;
}

}  // namespace balla::scene::evimo2
