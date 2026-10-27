// The EVIMO object catalogue (ballasim.evimo2.dataset.catalogue; docs/dev/CPP_PORT.md item 5, S5): each top-level
// folder's settings.txt, config.txt (marker positions in the mesh frame) and vicon.vsk (marker positions in the Vicon
// subject frame, mm), parsed as the Python parsed them (str.split / splitlines, float(), int(), the same regular
// expressions).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "scene/rotation.h"

namespace balla::scene::evimo2 {

struct CatalogueEntry {
  int64_t id = 0;
  std::string name, folder, mesh_path;
  double unit_scale = 1.0;
  std::vector<rot::V> config_markers, vsk_markers_mm;
  std::vector<std::string> vsk_names;
  double marker_radius_mm = 0;
  std::vector<std::pair<std::string, std::string>> settings;
};

// No catalogue folder (Python's FileNotFoundError).
struct CatalogueMissing : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// dataset._object_folders: the folders of `base` holding a settings.txt, sorted; CatalogueMissing when base is not a
// directory.
std::vector<std::string> object_folders(const std::string& base);
// dataset.catalogue_files: each object folder's settings.txt, config.txt and vicon.vsk that exist.
std::vector<std::string> catalogue_files(const std::string& base);
// dataset.catalogue(objects=base): the objects in folder order (a folder without an id, config.txt or vicon.vsk is
// skipped); a duplicate id throws.
std::vector<CatalogueEntry> catalogue(const std::string& base);

// The parsers (dataset._read_settings, _read_config, _read_vsk) on file contents.
std::vector<std::pair<std::string, std::string>> parse_settings(const std::string& text);
std::vector<rot::V> parse_config(const std::string& text);
void parse_vsk(const std::string& text, const std::string& path, std::vector<rot::V>& markers,
               std::vector<std::string>& names, double& radius);

}  // namespace balla::scene::evimo2
