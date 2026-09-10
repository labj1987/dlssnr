# Changelog

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
