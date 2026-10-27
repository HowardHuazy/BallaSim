# Loading and resolution

`ballasim.config` loads files and records their provenance; `ballasim._core.config`
resolves plain values and validates their dependencies. Read
[Configuration](../../guide/configuration.md) for the layered file format.

## Loading a run

(configuration-load)=
### `config.load`

```python
config.load(source, *, sets=(), profiles=(), base_dir=".", scene_path=None)
```

Read a format-2 YAML path or a mapping and return `RunSpec`. The loader expands
path values, loads the versioned profiles that are actually referenced, records
their content hashes, and resolves scene-stage settings. It does not build the
scene or simulate events. Use it to prepare one explicit set of layered inputs
before choosing or supplying a bundle.

`sets` accepts dotted-key mappings or `KEY=VALUE` strings parsed as YAML values.
`profiles` appends root profile IDs. Mapping-relative paths use `base_dir`;
file-relative paths use the file's directory; CLI-style paths use the working
directory. `scene_path` replaces the scene source with a concrete path. Invalid
layers raise `ConfigError`; missing files raise `FileNotFoundError`.

(configuration-runspec)=
### `RunSpec` and `RunSpec.get`

`inputs` contains the plain-value resolver inputs; `record` retains the original
configuration, CLI overlays, and profile identities; `scene` contains the
scene-stage resolution. `get(key)` reads that scene resolution. Full camera
settings are not yet available because they depend on the scene cameras and
bundle facts. Pass the spec to `ballasim.run`, or resolve it against a bundle.

(configuration-resolve)=
### `config.resolve`

```python
config.resolve(spec, bundle, *, supplied=True, digest=None, low_light=None,
               device_available=None)
```

Return a native `Resolved` object after applying camera defaults, sensor priors,
accuracy derivations, source facts, and cross-key checks. This is the point where
values such as pixel-pitch-dependent blur and selected camera names become
concrete. Resolution does not perform the reference-photometry image solve;
that scale is obtained when the camera starts.

Set `supplied=False` for a bundle built from the configured scene source.
`digest` and `low_light` preserve source-specific facts; `device_available`
can supply an independently probed device result. A supplied bundle replaces the
configured scene source, rather than merging two scenes.

## Reading resolved values

(configuration-get)=
### `Resolved.get`

`get(key, camera=None)` returns one value. Camera-scoped keys such as
`sensor.theta_on` require a camera name; scene, run, and dataset keys do not.
`KeyError` means the key has no resolved value in that context. This avoids
silently using a default from a different camera or from a different resolution
stage.

(configuration-values)=
### `Resolved.values`

Return plain nested mappings: `run` contains run, scene, and dataset keys;
`cameras` maps each selected name to its camera values. This is the native
engine's value input. It intentionally omits provenance wrappers so numerical
consumers receive values of the schema's declared types.

(configuration-record)=
### `Resolved.record`

Return the same layout with each entry represented by its value, winning layer,
status, and optional note. Use this to explain a run or persist its resolution.
A recorded value is not evidence of a device measurement: its status can be a
prior, placeholder, user choice, or derived quantity.

(configuration-digests)=
### `Resolved.digest` and `Resolved.digests`

`digest(stage, camera)` returns the SHA-256 identity of the selected stage and
its preceding stages. `digests()` returns all camera/stage identities. Stage
order is scene, render, schedule, sensor, then output. This represents cache
invalidation dependencies, not numerical similarity: even a small physical
parameter change can require a new stage result. Full digests require resolution
against a bundle.

(configuration-diagnostics)=
### Resolution metadata

`cameras` lists names in bundle order; `span` gives the aligned output interval.
`warnings` contains readable messages, while `diagnostics` carries structured
keys, layers, and related inputs. `facts` contains derived informational values
that are not themselves stage inputs. Inspect these alongside a value when an
automatic derivation or an inactive setting is unexpected.

## Validation and provenance

(configuration-bundle-info)=
### `config.bundle_info`

Build the plain facts required by the native resolver from a `SceneBundle`,
including camera dimensions, pitch, clocks, source digest, and lighting facts.
The `supplied` keyword records whether the bundle replaces the scene source.
Use this when a separate interface needs to validate a spec before constructing
a full engine.

(configuration-missing-keys)=
### `config.missing_keys`

`missing_keys(spec, info, *, diagnostics=False)` collects all unresolved required
values as `{key, hint, help}` records. Enabling diagnostics adds the structured
error each missing value would otherwise produce. Other malformed values still
raise immediately. This is intended for configuration forms and import checks:
it lets the user fill missing calibration without masking unrelated errors.

(configuration-parse-sets)=
### `config.parse_sets`

Convert CLI-style `KEY=VALUE` strings into a dotted-key mapping, or copy an
existing mapping. Values use YAML scalar and flow-list syntax; this function
parses them but does not perform full cross-key validation. An empty right-hand
side becomes `null`, which is a value rather than a request to restore defaults.

(configuration-read-profile)=
### `config.read_profile`

Read a packaged profile by its versioned ID and return its YAML document as
plain values. A profile includes provenance and mount information in addition
to settings. Loading this document does not by itself apply the profile to a
run; resolution determines precedence and whether it is valid for the context.

(configuration-schema)=
### `_core.config.schema`

Return every parameter descriptor, including scope, type, units, default kind,
choices, ranges, relevance, affected stage, and source. This is the machine-readable
counterpart of the topic-specific settings pages. Use descriptors to build
controls that distinguish a physical parameter from an execution choice and a
required value from a family prior.

(configuration-families)=
### `_core.config.families`

Return the expanded priors for each sensor family and their inheritance/status
metadata. Expansion makes copied values visible, rather than claiming they were
separately calibrated. See [Sensor families](../parameters/families.md) for the
model assumptions behind those values.

(configuration-accuracy-profiles)=
### `_core.config.accuracy_profiles`

Return each accuracy preset's literal entries and derivation descriptions. These
are numerical policies, not device profiles. Use full resolution to obtain their
values on a particular camera set: formulas can depend on selected thresholds,
noise, and timestamp quanta.

(configuration-default)=
### `_core.config.default`

`default(key, family=None)` returns a literal default or a requested family's
prior. Required and derived defaults raise `ValueError`, because they have no
context-free value. This distinction prevents an importer or form from inventing
a neutral illumination scale or using an unresolved bundle-dependent value.

(configuration-check)=
### `_core.config.check`

`check(key, value)` validates one value's type, nullable status, choices, list
length, finite numbers, and hard range. It does not establish compatibility with
other keys or a scene. Use full resolution for constraints such as threshold
floor versus threshold mean or a crop versus camera dimensions.

(configuration-explain)=
### `_core.config.explain`

`explain(spec.inputs, resolved, key, camera=None)` lists candidate sources in
increasing precedence, including priors, derivations, profiles, configuration,
and CLI inputs. The winning layer is recorded by `Resolved.record()`. Use it
when a visible value differs from a schema default rather than manually
reconstructing profile inheritance.

(configuration-relevant)=
### `_core.config.relevant`

`relevant(key, resolved, camera=None)` evaluates a descriptor's applicability
conditions on resolved values. An inactive value may still be present and
well-typed; it simply does not affect the selected mode. This lets interfaces
explain why, for example, a diffraction wavelength is inactive with diffraction
off, without deleting the user's setting.

(configuration-derived-from)=
### `_core.config.derived_from`

`derived_from(profile)` returns dependencies of derived accuracy settings,
including transitive dependencies. Setting a derived key explicitly replaces
its own derivation. The dependency map is useful for predicting which values a
sensor edit may recompute; it is not an independent resolver or a physical
sensitivity estimate.

## Errors

`config.ConfigError` is a `ValueError` with an optional `diagnostic` mapping.
Native resolution uses `_core.config.DiagnosticError`, also a `ValueError`.
Diagnostics contain `kind`, `severity`, `key`, `camera`, `layer`, `keys`,
`message`, and `nearest`. Missing input files remain `FileNotFoundError`.

## Cache and clock integration

These low-level `_core.config` functions support integrations that already own
cached signals and readout records. Their presence does not provide a complete
interactive replay workflow.

### `from_record`

`from_record(resolved, stage_digests)` reconstructs a `Resolved` object from a
finished run's recorded entries and identities. It preserves recorded values;
it does not rebuild the scene or verify that current input files are unchanged.

### `classify`

`classify(base, edit, camera, facts={})` compares resolutions and returns a cache
tier, changed stages and keys, and refusal reasons. The facts describe the cached
signal's photometry and precision. An earlier-stage change can invalidate a
later cached result even when its own local settings are unchanged.

### `replay_scale`

`replay_scale(edit, camera, facts)` returns the scalar photocurrent scale
appropriate to a cached signal, including reference-photometry normalization.
The caller applies that scale; the function does not modify cached arrays and
cannot recover a changed scene or missed temporal samples.

### `probe_digest`

`probe_digest(resolved, camera)` computes the sensor identity with the event-only
edit keys masked. This distinguishes comparator-state compatibility from changes
that can be derived afterward from raw crossings and attribution increments.

### Clock mappings and lower bounds

`sensor_clock_ns(readout, t_true_ns)` applies the sensor clock.
`sensor_time_lower_bound(readout, s_ns)` finds the corresponding true-clock
lower bound. `crossing_lower_bound(readout, s_ns)` also accounts for delay and
quantization. Use lower bounds for clock-aware interval cuts rather than
subtracting a constant offset when drift or quantization is present.

A readout mapping accepts `quantum_ns`, `delay_ns`, `phase_ns`, `clock_offset_ns`,
and `clock_drift_ppm`. These scalar helpers retain integer-nanosecond arithmetic.
