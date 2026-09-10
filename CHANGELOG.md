# Changelog

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
