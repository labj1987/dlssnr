# Changelog

## 0.1.14 — 2026-09-10

- **Cross-frame async pipelining** (`GpuCompose::dispatch_into_image_async`, new):
  submits the compute dispatch + write-back with a signal semaphore and returns
  immediately instead of blocking on its own fence. `capture::run` now returns
  `Option<vk::Semaphore>`; `device.rs`'s present hook chains it into the real
  `vkQueuePresentKHR` call's own wait-semaphore list (combined with the app's own,
  never replacing them) so the presentation engine — not our code — waits for the GPU
  work before displaying the frame. Explicitly authorized: "do it if it gives us the
  most frames when NR is on."
- Genuinely double-buffered (`ASYNC_SLOTS = 2`, independent images/staging
  buffer/command buffer/fence/semaphore each) to avoid a real data race between two
  in-flight dispatches; the only wait is on a slot's *own* fence from its *previous*
  use, immediately before reuse, never before returning the current result. Verified
  the binary-semaphore reuse discipline is sound both by reasoning (a wait is always
  chained into that same frame's present call before the same slot could ever be
  reused) and by a new test driving it across many more iterations than there are
  slots, against a real device.
- Deliberately did not add reprojection/stale-answer tricks to chase a bigger win:
  without real motion vectors, that would cause real visible ghosting on moving
  content. Every frame's presented image still comes from that same frame's own
  capture and model answer — only *when* the CPU learns the work is done changed.
- Verified thoroughly on real hardware before trusting it: a short run first
  (watching specifically for hangs/crashes), then a real 10s measurement, then a real
  `capture_request` dump (still visually correct), then a 45-second/601-frame stress
  run to rule out a slot-reuse issue only surfacing after many cycles. Zero crashes,
  zero hangs, zero fallbacks except when a capture_request was genuinely pending.
- Real gain: 143 frames/10s, up from 128 (~134/10s sustained over 45s). Documented
  this is likely close to the practical ceiling for the current architecture — the
  remaining gap to the 244-frame no-composition baseline is real GPU bandwidth/work
  volume, not something more scheduling cleverness can remove.
- 3 new tests, 30 tests in this crate now, full suite green.

## 0.1.13 — 2026-09-10

- **Merged GPU compose + write-back into one submission** (`GpuCompose::dispatch_into_image`,
  new): writes the composited result straight into the real swapchain image, in the
  same command buffer as the compute dispatch, when nothing needs to see the bytes on
  the CPU (no pending `capture_request` dump — checked via a new non-consuming
  `ShmClient::capture_request_pending`). Two GPU submissions per frame instead of
  three, no CPU round-trip for the composited bytes in the common case. Deliberately
  routes through an intermediate buffer rather than a raw image-to-image copy, since
  the latter would silently corrupt colors if a real swapchain's format ever differs
  from this module's own hardcoded format (nothing here can vary/test that in this
  environment) — a buffer has no format attached, so the final copy always targets the
  real image's own true format, exactly like the existing stage 2 it replaces.
- Verified byte-for-byte identical to the already-verified separate-dispatch path by a
  new local test, before ever touching real hardware. Verified correct and measured on
  real hardware after: every frame took the fast path except the one a real
  `capture_request` was pending for, which correctly fell back and still dumped a
  visually correct frame. Real gain: 128 frames/10s, up from 118.
- Documented what this reveals: the no-composition baseline already does the same
  number of submissions per frame, so the larger remaining gap (244 vs. 128) is real
  GPU bandwidth/work-volume, not submission count — genuinely closing it further needs
  cross-frame pipelining, which trades in a frame of latency and is a real product
  decision, not attempted here.
- 1 new test, full suite green.

## 0.1.12 — 2026-09-10

- **`shaders/compose.comp` is now really dispatched on the GPU** (`crates/layer/src/composition/gpu.rs`,
  new), tried first in the write-back whenever `debug_view == 0`, falling back to the
  CPU path otherwise. The shader itself moved from `rgba16f` to `rgba8` storage images
  with explicit sRGB decode/encode added (storage-image loads never apply an sRGB
  curve regardless of format), precompiled to SPIR-V and embedded via `include_bytes!`.
- **A real shader bug was found and fixed before it ever touched real hardware**, by a
  new local test that needs only a software Vulkan ICD (lavapipe): `OklabFromLinearSrgb`
  multiplied a matrix by `sign(lms)` before the cube root instead of after, due to GLSL
  operator precedence — diverged from the CPU reference by up to 90/255 with the
  default `colour_strength = 1.0`. Fixed and reverified.
- Verified correct on real hardware after the fix (same real, structured composition
  effect as the CPU path). Real, honest performance finding: 118 frames/10s on the GPU
  vs. 97 for multi-threaded CPU and 244 for no composition — a real but modest gain,
  since the current synchronous one-submit-one-wait-per-frame pattern (three CPU-GPU
  round trips per frame now) is the real remaining cost, not the shader itself.
  Pipelining that is genuinely still-open work.
- 2 new tests (GPU-vs-CPU parity, resize handling), both skip gracefully with no
  Vulkan ICD present rather than breaking the build. Full suite green.

## 0.1.11 — 2026-09-10

- **Expanded the GUI settings surface**: a new "Compare and debug" group
  (`compare_mode`, `compare_split`, `compare_zoom`, `compare_swap`, `debug_view`) and
  five new Composition rows (`colour_mode`, `transfer`, `unlock_passes`,
  `apply_model`, `hold_frame`) — all real, tested, and verified by screenshot. Still
  not bound: white-point HDR tuning and the raw hotkey (`toggle_key`), which needs a
  proper key-capture widget.
- Extended `ShmHeader::persisted_settings`/`apply_persisted_setting` from 21 to 31
  entries so every new row actually survives a reboot through `config.ini`, not just
  appears to save.
- Found and fixed a real bug while verifying the above by actually running the GUI:
  `AdwPreferencesGroup::title` is parsed as Pango markup, and naming the new group
  "Compare & debug" broke it outright. Renamed to "Compare and debug".
- `dlssnr-cli shmctl` now covers all 31 persisted settings (previously 21, with
  `debug_view`/`apply_model`/`compare_mode`/`hold_frame` handled as a separate
  non-persisted special case — folded into the main list now that the GUI needs them
  to persist too).

## 0.1.10 — 2026-09-10

- **`dlssnr-cli shmctl`** (`crates/cli/src/shmctl.rs`, new): the real equivalent of
  upstream's separate `dlssnr-shmctl` debug/introspection tool this project had none
  of before — `status` (all 21 persisted settings plus live `helper_state`/`model_up`/
  `helper_frames`/`debug_view`/`apply_model`/`compare_mode`/`hold_frame`/
  `capture_request`), `set <name> <value>`, `toggle <name>`, and `capture [view]` (the
  same real frame-dump this project used to first visually verify its composition
  output). 10 new tests on the pure resolve/store/toggle logic; full suite green.

## 0.1.9 — 2026-09-10

- Real `ShmHeader::capture_request` support (`crates/layer/src/dump.rs`, new):
  writes a matched before/after PNG pair on request. Plus
  `crates/protocol/examples/trigger_capture.rs`, a small manual tool to trigger one
  (and optionally set `debug_view`) against a running instance from the outside.
- Three real bugs found (via `strings` against the real `nvngx_dlssnr.dll`) and fixed
  in the helper's NGX evaluation, all independently justified, none the actual cause
  of what they were found while chasing (see below): `DLSSNR.Depth`/`DepthInverted`
  and every resource's `*Subrect*` scalar were never bound; `DLSSNR.Reset` was set
  once at creation and never toggled to 0 for subsequent frames; the device extension
  list was copied from a native-Linux reference binary's own strings output without
  adjusting for this crate's actual Windows/Wine platform
  (`VK_EXT_external_memory_dma_buf`/`VK_KHR_external_memory_fd` -> `VK_KHR_external_memory`/
  `VK_KHR_external_memory_win32`; device extension count went 6/9 -> 8/9).
- Found and fixed the actual bug behind an initially-alarming solid-white dumped model
  answer: it was this session's own new PNG dump tool passing through a real answer's
  alpha channel (0 across the whole image) unmodified, which a PNG viewer renders as
  blank/transparent — not a broken model output. A real opaque-composite-mode present
  never reads alpha at all, so this could never have affected an actual displayed
  frame. Fixed by forcing alpha to 255 before encoding.
- **First real visual confirmation the full pipeline produces correct output**: a
  pixel diff between a real captured frame and its real composited answer shows a
  mean per-channel difference of ~17/255 across 100% of sampled pixels — real,
  structured work, not a no-op or garbage. Full writeup in CLAUDE.md.

## 0.1.8 — 2026-09-10

- **This project's own composition math now actually reaches the presented frame**
  (`crates/layer/src/composition/apply.rs`, new): a CPU port of `shaders/compose.comp`'s
  pipeline, built on the already-tested `upgrade_tone_map`/`gamut_compress_reversible`
  functions, wired into `capture.rs`'s real write-back for the `RGBA8` proxy format.
  `debug_view` (composited/original/raw-answer/amplified-diff) and `apply_model`'s
  off-switch are both real now, read live from the SHM header every frame.
- Real, measured performance finding: the naive single-threaded version cost ~800ms/
  frame at 1080p on real hardware (244 -> 11 frames in a real 10s `vkcube` run).
  Parallelized the (fully independent, per-pixel) work across threads —
  ~97 frames in the same real 10s run, an ~8x measured improvement. Still short of the
  244-frame no-composition baseline; real GPU dispatch of `compose.comp` remains the
  actual fix for game-ready framerates and is still open work.
- 5 new tests for `apply_rgba8`'s real invariants; full suite (30 tests) still green.
  Verified on real hardware: round trip still succeeds every frame with composition
  active.

## 0.1.7 — 2026-09-10

- **First confirmed real DLSS 5 Neural Rendering success from this project's own
  code.** With the 0.1.6 crash fix in place, ran a fresh `dlssnr_helper.exe` under real
  Proton + real `vkcube`/`VK_LAYER_dlssnr_neural` (implicit activation) against the
  real, legitimately-signed `nvngx_dlssnr.dll` on `lordnikon`. Result: `VULKAN_CreateFeature(18)
  -> 0x1` (real non-null handle, 1920x1080) and `EvaluateFeature -> 0x1` on 243/244
  captured frames (the one miss is frame 1, before `CreateFeature` had run) — zero
  evaluation failures across a full 10-second run. This is the exact success shape
  previously only ever produced by upstream's C++ build. Doesn't yet prove the visual
  output is correct (composition math still isn't wired in, real optical flow still
  isn't) — see CLAUDE.md's new "First confirmed neural-rendering success" section for
  exactly what this does and doesn't close.
- Repo visibility changed to private.

## 0.1.6 — 2026-09-10

- **Fixed the critical layer crash documented in 0.1.5**: `VK_LAYER_dlssnr_neural`
  segfaulted 100% of the time under real, implicit activation (`VKLayer_DLSS5=1`) in
  the presence of Mesa's `device_select` implicit layer. Root cause, found by
  reproducing the crash locally against Google's own pristine, unmodified
  `vulkan-layer` `hello-world` example (same crash, same backtrace — proving this was
  never a dlssnr-specific bug): `vulkan_layer::Global::create_instance`'s default
  fallback path eagerly resolves all three Vulkan 1.0 global entry points
  (`vkCreateInstance`/`vkEnumerateInstanceExtensionProperties`/
  `vkEnumerateInstanceLayerProperties`) through the chained, `VK_NULL_HANDLE`-instance
  `vkGetInstanceProcAddr`, even though a layer that doesn't hook those extra two never
  calls them afterward. Resolving `vkEnumerateInstanceExtensionProperties` that way
  segfaults inside `libVkLayer_MESA_device_select.so` on this Mesa build 100% of the
  time; `vkCreateInstance` resolves fine through the exact same chained pointer right
  before it. Worked around by implementing `GlobalHooks::create_instance` ourselves
  (`crates/layer/src/lib.rs`'s new `DlssnrGlobalHooks`) and resolving only the one
  entry point this layer actually needs, never making the query that crashes.
  Verified fixed: 5/5 clean runs locally (this sandbox has the identical Mesa
  `device_select` present) and 3/3 clean real `vkcube` runs on `lordnikon` (real
  GPU/driver, the machine the original crash was found on) — capturing and
  round-tripping real frames the whole time, no crash. Full workspace test suite and
  both smoke tests (explicit and implicit activation) still green.

## 0.1.5 — 2026-09-10

- No fix in this release — documenting a critical, confirmed bug found while testing
  against real hardware with a legitimate NGX DLL for the first time: this project's
  Vulkan layer segfaults 100% of the time when loaded the way it will always actually
  be loaded (implicit activation via `VKLayer_DLSS5=1`), in the presence of Mesa's
  `device_select` implicit layer. Upstream, tested side by side on the same machine,
  works correctly end to end. See CLAUDE.md for the full, gdb-verified writeup — what's
  ruled out, what isn't yet, and how to reproduce it.
- Runtime SHM directory permissions note: if DLSS5 NR silently refuses to work, check
  `/tmp/dlssnr-$UID` is `0700` (owner-only) — both this project's and upstream's own
  security check correctly reject a group-writable runtime directory.

## 0.1.4 — 2026-09-10

- The write-back now uses the helper's actual answer instead of always
  re-presenting the untouched capture. Not yet verified against a real present
  cycle (no display/swapchain in this dev sandbox) — the underlying change is
  small and carefully reasoned through, but real confirmation still needs to
  happen against an actual game on real hardware.

## 0.1.3 — 2026-09-10

- Settings changed in the GUI now survive a reboot. Previously they lived only in the
  SHM mapping under `/tmp`, which doesn't persist — found by comparing against a real,
  installed upstream instance's `config.ini`, which does persist them.

## 0.1.2 — 2026-09-09

- Fixes a real crash found on first-ever real-hardware testing: `create_feature_at()`
  set every `DLSSNR.*` parameter through the raw NGX vtable outside any `guarded()`
  call, the only DLL-touching code in `ngx.rs` that wasn't. On a real GPU + real
  `nvngx_dlssnr.dll`/`nvngx.dll`, a fault in that block took the whole helper down
  silently a couple seconds after a successful `VULKAN_Init_Ext`, with no log line and
  no crash dialog. Now guarded like every other call in the file, so a fault latches
  `disabled` and logs `seh=...` instead of killing the process.

## 0.1.1 — 2026-09-09

- Adds a Start/Stop button for the helper to the GUI's Status group. Backed by a new
  shared `dlssnr-supervisor` crate (extracted from `dlssnr-cli`) so both the CLI and
  GUI start/stop the helper through the same code instead of duplicating it.
- Adds an About dialog to the GUI (there wasn't one before) crediting Claude Code
  (Anthropic) in its acknowledgements, matching GreenLight/KernelPop/SteamPunk.
- Adds a "NGX binaries" import button to the GUI's Status group — previously only
  `dlssnr-cli import-binaries` could do this.

## 0.1.0 — 2026-09-09

- First release.
