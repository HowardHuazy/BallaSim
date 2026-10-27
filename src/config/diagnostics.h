// Structured diagnostics of the resolver (docs/design/phase3/ui.md §3.10, UI-4a): every error and warning of
// docs/dev/P0_PARAMS.md §2.4 as a kind, a severity, the key, camera and layer it concerns, every key a cross-key rule
// involves, the message (the text the resolver always printed) and, for an unknown key, the nearest one. Errors stay
// fail-fast: the resolver throws a DiagnosticError, a std::invalid_argument carrying one Diagnostic, so every existing
// catch (config.py's ValueError -> ConfigError, the CLI's exit 2) is unchanged. Header-only (SensorParams::validate
// throws it too, from balla_sensor_params).
#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace balla::config {

// The P0_PARAMS.md §2.4 table, plus kFlaggedDefault (the D4 / D4c warnings naming the defaults that come from EVIMO2
// data or are off for want of a literature value).
enum class DiagKind {
  kUnknownKey,
  kUnknownCamera,
  kWrongScope,
  kType,
  kHardRange,
  kSoftRange,
  kRequired,
  kFamilyMissing,
  kInert,
  kCrossKey,
  kRunLevel,
  kDerivedConflict,
  kRunWideDisagree,
  kProfile,
  kFormat,
  kFlaggedDefault,
};
enum class Severity { kError, kWarning };

inline const char* name(DiagKind k) {
  switch (k) {
    case DiagKind::kUnknownKey: return "unknown_key";
    case DiagKind::kUnknownCamera: return "unknown_camera";
    case DiagKind::kWrongScope: return "wrong_scope";
    case DiagKind::kType: return "type";
    case DiagKind::kHardRange: return "hard_range";
    case DiagKind::kSoftRange: return "soft_range";
    case DiagKind::kRequired: return "required";
    case DiagKind::kFamilyMissing: return "family_missing";
    case DiagKind::kInert: return "inert";
    case DiagKind::kCrossKey: return "cross_key";
    case DiagKind::kRunLevel: return "run_level";
    case DiagKind::kDerivedConflict: return "derived_conflict";
    case DiagKind::kRunWideDisagree: return "run_wide_disagree";
    case DiagKind::kProfile: return "profile";
    case DiagKind::kFormat: return "format";
    case DiagKind::kFlaggedDefault: return "flagged_default";
  }
  return "?";
}
inline const char* name(Severity s) { return s == Severity::kError ? "error" : "warning"; }

// key: the schema key (camera keys without cameras.<camera>.; empty when no single key is concerned); camera: the
// camera block or camera name ("default" for cameras.default; empty for run keys); layer: the layer as run.json
// records it (config, cli, profile:<id>, default, prior:<family>, accuracy:<profile>, derived, bundle; empty when no
// layer sets the key); keys: every key a cross-key rule or a warning involves (schema keys, `key` first).
struct Diagnostic {
  DiagKind kind = DiagKind::kRunLevel;
  Severity severity = Severity::kError;
  std::string key, camera, layer;
  std::vector<std::string> keys;
  std::string message, nearest;
};

struct DiagnosticError : std::invalid_argument {
  Diagnostic diag;
  explicit DiagnosticError(Diagnostic d) : std::invalid_argument(d.message), diag(std::move(d)) {}
};

}  // namespace balla::config
