# Attribution

`dlssnr` is a from-scratch Rust/GTK4 rebuild of the architecture behind
[DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (bmitch87) — a Linux Vulkan
layer plus Windows NGX helper that runs NVIDIA DLSS 5 Neural Rendering on Linux and
Proton games. No source from that project (or anything it in turn depends on) was
copied or transliterated into this codebase. This file records what was actually
taken, and from where.

## What's taken, and how

| Source | What's taken |
|---|---|
| [DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) | The overall architecture: a Vulkan implicit layer talking to a Wine/Proton-hosted Windows helper over shared memory, the NGX call sequence and vtable shape (NVIDIA's own public-facing API surface), the shared-memory protocol's field layout and semantics, and the settings taxonomy exposed in the GUI/CLI. All reimplemented independently in Rust from this description of the behavior, not from reading the C++/Qt source line-by-line. |
| [RenoDX](https://github.com/clshortfuse/renodx) (clshortfuse) — MIT | The color composition design: the two-branch luminance/headroom rule, the OkLab hue-correction step, and the reversible neutral-axis gamut compression. This is RenoDX's own DLSS 5 addon design, MIT-licensed, read and reimplemented directly from RenoDX's own public source — not through any GPL-licensed intermediary. |
| Björn Ottosson ([bottosson.github.io/posts/oklab](https://bottosson.github.io/posts/oklab/)) | The OkLab conversion matrices, published as public reference constants. |
| Public domain / standard color science | sRGB transfer function, SMPTE ST.2084 (PQ) encode/decode, Hunt-Pointer-Estevez LMS conversion — textbook formulas, attributable to no one project. |
| Public domain / standard resampling literature | The Lanczos, Catmull-Rom, Mitchell-Netravali, and Kaiser-windowed-sinc kernels used for the supersampling downscale leg — implemented from their mathematical definitions. |

## What's deliberately not taken

DLSS5VKLayer's own `ATTRIBUTION.md` documents that its composition shader was carried
in through [OptiScaler](https://github.com/cdozdil/OptiScaler) and two DLSS-NR forks of
it ([Dagherbou/OptiScaler_DLSSNR](https://github.com/Dagherbou/OptiScaler_DLSSNR),
[y4my4my4m/OptiScaler_DLSSNR_Multipass_MFG](https://github.com/y4my4my4m/OptiScaler_DLSSNR_Multipass_MFG)),
all GPL-3.0, and that DLSS5VKLayer itself ships without a license file of its own.
None of that shader source, HLSL or compiled SPIR-V, was read, copied, or ported here.
This project's composition math was rederived solely from RenoDX's own MIT-licensed
source and the public color-science/resampling references listed above, specifically
so that no GPL-3.0 obligation attaches to this codebase.

## The NGX caller-identity spoof

The helper's import-table patch that spoofs the calling module's identity to NVIDIA's
`nvngx_dlssnr.dll` (`crates/helper/src/spoof.rs`) is a reimplementation of a mechanism
DLSS5VKLayer's helper also uses — a PE import-address-table hook, a well-known and
generic Windows technique, implemented here directly against the Win32 API rather than
read from DLSS5VKLayer's C++. Unlike the composition math above, this is a case where
this project *is* deliberately reproducing DLSS5VKLayer's underlying approach (bypassing
NVIDIA's caller-identity check), by design and with the legal exposure that implies —
see [README.md's Legal section](README.md#legal). That decision is about what the
mechanism does, not about copying anyone's expression of it.

## License texts

This project's own code is MIT — see [LICENSE](LICENSE). RenoDX's MIT license applies
to the design this project's composition math is derived from; DLSS5VKLayer, OptiScaler,
and its DLSS-NR forks are referenced above for provenance only and no code from them
appears in this repository.
