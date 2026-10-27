# Browser interface

The local browser interface configures simulations, manages jobs and inspects
event streams. Install the `ui` extra, then open a run or start the server:

```bash
ballasim ui --open runs/example
ballasim ui --root runs=/path/to/runs
```

`--open` reuses a running server or starts one in the background. Chrome or
Chromium 147 or later is expected. **Browser check** reports WebGPU availability;
the basic event viewer uses server-rendered PNG panels when WebGPU is unavailable.
Real-versus-sim comparison panels require WebGPU.

## Pages

| Page | Use |
| --- | --- |
| **Server** | Inspect the build, device availability, memory, jobs directory and file roots |
| **Runs** | Browse registered roots and open jobs, runs or configuration files |
| **Setup** | Edit parameters, resolve configurations, save YAML and submit runs |
| **Viewer** | Open an event source and inspect its timeline |
| **Real vs sim** | Compare two streams with shared timing and display settings |
| **Browser check** | Inspect browser, isolation and WebGPU capabilities |

## Configure and run

1. Open **Setup** and choose **New configuration**, **Configuration file** or
   **Run** as the source.
2. Select the scene and profiles. Edit run settings and camera settings in their
   respective sections. Filter parameters by key or help text.
3. Resolve missing required values. Use **Apply scene edits** after changing
   scene-stage parameters; use **Build the scene bundle** when requested.
4. Choose a writable destination and select **Save configuration** or **Run**.

Parameter edits use the same native resolver as the command line. The form shows
resolved values, provenance, warnings and missing inputs. Advanced accuracy
settings are grouped separately; heuristic parameters require `ui --dev` to edit.

The job page reports its state, current stage, progress, performance and runner
logs. **Cancel** requests a graceful stop. **Kill** is available for stalled jobs
and leaves their output incomplete. A paused queue can be restarted
with **Resume queue**.

## Inspect results

Open a run from **Runs**, then choose a tab:

| Tab | Contents |
| --- | --- |
| **Events** | Camera selector and event viewer, with available frame and IMU timeline tracks |
| **Performance** | Per-window performance plots |
| **Provenance** | Inputs, CLI overrides, profiles, resolved values, stage digests and software versions |
| **Files** | Run files available for download |

Progress and logs remain visible while a run is active. Event files become
available after their writer closes; partial files cannot be opened as streams.

## View events

The viewer accepts BallaSim or DSEC HDF5 files, NumPy event files or column
directories, and EVIMO2 sequence directories. Paths must lie inside a registered
root; `ui --open PATH` registers the attached directory as read-only.

Select **events**, **polarity**, **counts**, **time surface**, **decay** or
**(x, y, t) cloud**. Window length controls the displayed interval; clip and gamma
control the color mapping. **Auto** estimates a suitable clip value. Time-surface
and decay views also expose the time constant τ.

Drag the timeline to seek and scroll over it to change the window length. With
the timeline focused, use the arrow keys to step and Space to play or pause.
Click a timeline track or use `[` and `]` to move between recorded marks.
The pixel inspector reports the values under the pointer.

The viewer badge distinguishes **all events**, sampled events and blocks still
loading. Counts from sampled blocks are estimates. Shorten the window, or τ for
history views, when the event pool cannot hold the requested interval. Multiple
viewer tabs allocate separate GPU pools.

## Compare streams

In **Real vs sim**, enter the reference and simulation paths, choose polarity
handling and select **Compare**. The streams must have equal resolution. Panels
show both streams, signed count difference, activity ratio and an alternating
blink view.

Select a region using **viewer** ground-truth validity, a rectangular **crop** or
**none**. Inspect the proposed simulation clock offset, `δ_stream`, before
applying it; the page reports missing or inconsistent clock metadata. Choose
**Statistics of the window** to compute statistics within the selected region.

For saved videos, HTML viewers and plots, use the
[visualization CLI](../reference/cli.md#event-visualization).

## Server and session

The server binds to loopback only: `127.0.0.1` or `::1`. Named roots control file
access, and configuration and output writes require a writable root. Browser
sessions use local credentials; after signing out or restarting the server,
reopen the interface with `ballasim ui --open`.

```bash
ballasim ui --status
ballasim ui --stop
```

The server log is stored under `$XDG_STATE_HOME/ballasim/ui/server.log`, defaulting
to `~/.local/state/ballasim/ui/server.log`. See the
[CLI reference](../reference/cli.md#browser-interface-and-jobs) for root, browser,
cache and job-directory options.
