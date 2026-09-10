# Changelog

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
