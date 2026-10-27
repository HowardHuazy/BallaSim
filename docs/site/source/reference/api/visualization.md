# Event reading and visualization methods

Read events on an explicit comparison clock, select time windows, and derive
images or statistics. File contracts are described in
[Outputs & ground truth](../../guide/outputs.md).

## Visualization

Names below are available from `ballasim.viz`; install the `viz` extra. Event
sources retain integer nanoseconds internally. Viewer time shifts change the
comparison coordinate system; they do not calibrate the sensor's physical
latency.

(open-events)=
### `open_events`

```python
open_events(path, *, label=None, polarity="auto")
```

Open BallaSim/DSEC HDF5, an EVIMO2 sequence or NumPy event data as an
`EventSource`. `label` supplies display text. `polarity` is `auto`, `normal` or
`inverted`; automatic detection handles the EVIMO2 Samsung polarity convention.
HDF5 ground-truth extras and neighboring dense files remain available through
the source interface.

(arrays-to-source)=
### `arrays_to_source`

`arrays_to_source(columns, label="events")` wraps in-memory `x`, `y`, `p` and
one supported time column: integer `t_ns`, integer `t_us`, or `t` (floating-point
seconds or integer microseconds). Include resolution metadata for sparse streams
whose event extent does not cover the sensor. Return `ArraySource`.

(load-arrays)=
### `load_arrays`

`load_arrays(path, label=None)` reads `.npz`, a supported dictionary `.npy`, or a
directory of named NumPy columns and returns `ArraySource`. Use `open_events`
when the format is not known in advance. `H5Events` and `Evimo2Sequence` expose
the corresponding format-specific reader classes.

(event-source)=
### `EventSource` and `Events`

`EventSource` provides indexed and time-window access without requiring every
event in memory. Properties `size`, `t_start_ns`, `t_end_ns`, `extra_fields`
and `gt_kinds` describe the stream. `Events` contains aligned `x`, `y`, Boolean
`p`, `int64 t_ns` and optional `extras`; ON is `True`. It supports slicing,
which preserves all extra columns.

#### `EventSource.read`

`read(i0, i1, extras=())` returns events in the half-open index interval. Request
only needed extra fields to avoid unnecessary HDF5 reads.

#### `EventSource.window`

`window(t0_ns, t1_ns, extras=())` returns events with
`t0_ns <= t < t1_ns`. The source applies its comparison offsets before selection.
This is an event-time query, distinct from dense geometry queries.

#### `EventSource.search` and `EventSource.count`

`search(t_ns)` returns the first index at or after the timestamp.
`count(t0_ns, t1_ns)` returns the number in the half-open interval without
constructing an `Events` batch.

#### `EventSource.iter_chunks`

`iter_chunks(t0_ns, t1_ns, max_events=4194304, extras=())` iterates bounded-size
`Events` batches. Use it for long recordings and summary calculations where a
full time window would exceed memory.

#### `EventSource.gt_frame`

`gt_frame(kind, t_ns)` returns the nearest valid `GtFrame`, or `None` when no
sufficiently close frame exists. `GtFrame` holds its own time and data. Dense
frame association uses the file's spacing tolerance and should not be treated
as exact per-event ground truth.

#### Source views

`with_offset(offset_ns)` shifts event and ground-truth times together;
`with_gt_offset(gt_offset_ns)` sets ground-truth timing separately.
`with_polarity(inverted)` and `with_label(label)` change interpretation or
presentation. Each returns a shallow view, leaving stored files unchanged.
Use separate GT offsets when a recording's event and pose clocks differ.

(common-time-range)=
### `common_time_range`

`common_time_range(sources)` returns the interval covered by all sources.
Use a deliberate explicit range when comparing a stream with no events in part
of a sequence; overlapping event extents do not establish clock synchronization.

(timeline)=
### `Timeline`

`Timeline.from_range(t0_ns, t1_ns, dt_ns, window_ns=None)` constructs equally
spaced windows wholly inside the interval. Window length defaults to `dt_ns`;
a shorter step creates overlapping windows. `Timeline.from_windows(starts_ns,
ends_ns)` accepts explicit half-open intervals, including empty ones.
`window(k)` returns a pair of integer bounds and `len(timeline)` gives count.

(viz-render-options)=
### `RenderOptions`

```python
RenderOptions(kind="events", theme="dark", clip=None, gamma=0.5,
              tau_ms=20.0, mask="valid", third="none", sigma_px=None)
```

Choose how event windows are displayed. Kinds are signed `events`, separate
`polarity`, `timesurface` and `counts`; themes are `dark` or `white`.
`clip` sets color saturation; `None` estimates one shared value from the reference
stream. `gamma<1` keeps isolated events visible near dense activity.

`tau_ms` controls time-surface decay. `mask` chooses validity, instance boundaries
(`labels`) or `none`. `third` selects extra `diff`, `ratio`, `depth` or
`irradiance` panels, separated by commas. `sigma_px` smooths difference/ratio
panels; its defaults are zero for difference and 2 px for ratio. These display
operations do not modify the events used for file output or statistics.

(viz-renderer)=
### `Renderer`

`Renderer(sources, timeline, options=None, roles=None)` constructs a panel
renderer. `.render(k)` returns the window's panels and event counts as a
`RenderedFrame`; panel images are RGB `uint8` at sensor resolution before
composition. Roles provide labels for each source.

Difference and ratio views require exactly two equal-resolution streams.
Depth and irradiance panels require available dense products. Color and range
estimates are held across frames to preserve visual comparison; they do not
normalize every window independently.

(write-comparison)=
### `write_comparison`

```python
write_comparison(renderer, outputs, *, fps=25.0, layout="h", scale=1.0,
                 strip_frames=8, workers=None, progress=None)
```

Render each timeline window once and write all requested outputs. File suffix
selects video, GIF or PNG strip; a suffixless path selects a PNG directory.
`fps` is playback rate, independently of simulation step; `layout` is horizontal
or vertical. `progress(done, total)` reports completed windows. Return `None`.

(viz-write-frames)=
### `write_frames`

`write_frames(renderer, out_dir, workers=None, progress=None, *, flat=False,
file_names=None)` writes each panel as a native-resolution PNG without headers
and returns its panel directories. `flat=True` requires one panel; optional
`file_names` must supply distinct plain filenames for every window. This writer
produces event-display images, not `ballasim.frames/1` sensor frames.

(write-html)=
### `write_html`

```python
write_html(renderer, out_path, *, title="BallaSim event viewer", fps=25.0,
           image_format="webp", embed=True, workers=None, progress=None)
```

Write a static time-slider viewer and return its `Path`. Embedded images make
a self-contained HTML artifact. `embed=False` writes images to adjacent
`<stem>_frames/`, which must move with the page and is preferable for long
sequences. Supported image encodings are WebP and PNG.

(compute-stats)=
### `compute_stats`

```python
compute_stats(source, t0_ns, t1_ns, options=None, region=None, lights=None)
```

Return `StreamStats` for complete spectrum bins inside the requested interval.
The result includes ON/OFF counts, rate spectrum, vector strengths, inter-event
intervals and per-pixel rates. `.summary()` returns JSON-compatible scalar
statistics; undefined values are `None`.

`lights` maps fundamental frequencies to names; defaults come from stream
metadata. `region` restricts selected events and covered time. Rates are divided
by covered duration, not the entire requested span when GT has gaps.

(stats-options)=
### `StatsOptions`

`StatsOptions(rate_bin_ms=1.0, on_fraction_bin_ms=10.0,
spectrum_bin_us=100.0, spectrum_segment_s=1.0,
peak_band_hz=(20.0, 1000.0), max_interval_events=10000000)` controls statistical
resolution and interval-memory cost. Finer bins resolve faster rate variation
but contain fewer events; longer spectrum segments improve frequency resolution.
The interval limit truncates interval sampling, not overall event counts.

(gt-region)=
### `GtRegion`

`GtRegion(source)` selects pixels inside the source's available GT validity
mask. Intervals without a sufficiently close mask are uncovered and omitted.
The spectrum fills uncovered bins with the covered mean to avoid introducing
an artificial gap edge; this can reduce measured spectral peaks. Moving masks
also introduce modulation at the GT frame rate, so inspect whole-sensor spectra
when assessing lighting flicker.

(write-stats)=
### `write_stats`

`write_stats(stats, out_dir, f_max_hz=1000.0)` writes rate, polarity, spectrum,
interval and pixel-activity figures plus `summary.json`. Pass a sequence of
`StreamStats` objects to overlay streams on shared plots. Return written paths.
`f_max_hz` changes plotting extent rather than recomputing the underlying bins.
