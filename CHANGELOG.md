# Changelog

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
