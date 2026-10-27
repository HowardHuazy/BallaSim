# Event-output settings

These top-level `output.*` settings govern run storage and event annotations.
The event-file schema and timestamp reconstruction are described in
[Outputs & ground truth](../../guide/outputs.md).

(output-dir)=
## `output.dir`

Choose the run directory for event files, resolved configuration, provenance,
and progress logs. A path in YAML is relative to that file; a command-line path
is relative to the working directory. The directory identifies one run's
artifacts, including later dense, frame, IMU, and export products.

(output-overwrite)=
## `output.overwrite`

Permit replacement of an existing recognized run directory. Scene loading,
configuration resolution, and engine construction complete before it is created
or replaced. Camera initialization and warm-up can still fail afterward.
An occupied directory that is not a run, or one locked by another run, is refused.
This prevents a typo from treating an unrelated directory as simulation output.
The command-line `--overwrite` supplies the same permission for the selected run.

(output-gt)=
## `output.gt`

Write per-event ground truth alongside the event columns. The geometry query
uses the pixel center ray at the physical crossing time, including depth,
instance/semantic identity, and motion field. Turning this off reduces annotation
work and storage without changing the sensor's crossings. It removes the saved
annotations; it does not make dense ground truth unavailable for a later query
against the recorded scene.

(output-label-cut)=
## `output.label_cut`

Set the minimum share of the comparator's attributed increment required for a
SIGNAL or LEAK label. The sensor separates deterministic signal, temporal noise,
and leak contributions; eligible signal/leak shares are compared at this cut,
with tie and first-event rules. HOT overrides a non-signal label on a hot pixel.
Current emitted labels are SIGNAL, NOISE, LEAK, and HOT. MIXED has a reserved
file-format code but is not emitted by the current classifier.

This is attribution within the pixel model, not a post-hoc event-denoising
threshold. Raising it makes it harder to claim a dominant signal or leak cause;
it does not remove crossings or change their times. Exact relabeling requires
the raw attribution increments, which the standard event HDF5 does not store.
See [readout and label recomputation](../api/sensor.md#readout-and-label-recomputation)
for the low-level interface.

The label rules are implemented in
{download}`pixel.h <../../../../../src/sensor/pixel.h>` and
{download}`output.h <../../../../../src/engine/output.h>`.

## Quick reference

| Parameter | Type / unit | Base default | Description |
| --- | --- | --- | --- |
| `output.label_cut` | `float` | `0.5` | Minimum increment share required for a SIGNAL or LEAK event label. Range: (0, 1). |
| `output.gt` | `bool` | `true` | Write per-event ground-truth datasets. |
| `output.dir` | `path` | required | Destination run directory; must be specified. |
| `output.overwrite` | `bool` | `false` | Replace an existing run directory. |
