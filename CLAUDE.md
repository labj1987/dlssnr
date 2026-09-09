# dlssnr (working codename)

A from-scratch Rust/GTK4/libadwaita rebuild of
[DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (C++/Qt6): a Vulkan implicit
layer + Windows NGX helper that runs NVIDIA DLSS 5 Neural Rendering on Linux/Proton
games, forwarding frames to a Windows NGX helper running under Wine/Proton today (and,
per upstream's own stated roadmap, a native-Linux helper once NVIDIA ships one — the
whole point of the shared-memory seam below is that nothing on this side of it needs to
change when that happens). Distributed as a single AppImage, same as GreenLight/
KernelPop/SteamPunk — but unlike those two, this app needs **no root/pkexec step at
all**: everything lives under `~/.local/share`, `~/.config`, `/tmp/dlssnr-$UID/`.

**Full implementation plan**: `/home/alex/.claude/plans/breezy-napping-waffle.md` — read
that before touching anything below. It covers the licensing ground rules (upstream has
no LICENSE file and admits GPL-3.0 contamination in parts of its own tree — nothing here
may be produced by reading upstream's source and translating it), the NGX
authorization-bypass mechanism this rebuild knowingly carries forward (accepted risk,
isolate it, never disguise it), and the per-crate design.

## Current state

Every crate has real code now (milestones 1-3 solid and real-tested; 4 partial — the
math is tested, the GPU pipeline isn't wired up yet; 5 and 6 real and working, with the
caveats below). Read each crate's own gotchas section before touching it — "compiles"
and "the plan says this milestone is done" are not the same claim anywhere in this repo.

`helper` loads `nvngx_dlssnr.dll`, installs the caller-identity spoof, initializes NGX,
and creates Feature 18 (`ngx::load_and_init`) -- the path that actually exercises the
spoof and the SEH guard, wrapped in `crates/helper/src/guard.rs`'s VEH+`setjmp`/
`longjmp` mechanism. There is no per-frame `EvaluateFeature` loop yet: that needs bound
Vulkan image resources, which arrive with milestone 4 (the same resources the
composition pass reads/writes). **As of 2026-09-09 this is real-tested, not just
compiled**: the cross-compile toolchain now exists on this dev machine (see below), and
three purpose-built examples (`examples/guard_test.rs`, `spoof_test.rs`,
`spoof_install_test.rs`) run the SEH guard and the caller-identity spoof for real under
Wine, plus the full `dlssnr_helper.exe` binary itself was run under Wine and its
shared-memory mapping read back correctly from Linux-built code (true cross-toolchain
interop, not just matching offsets on paper). One real bug was caught and fixed this
way — an infinite-recursion stack overflow that `cargo check` structurally could not
have found — see `helper` gotchas below.

`layer` hooks `vkCreateSwapchainKHR`/`vkDestroySwapchainKHR`/`vkQueuePresentKHR` via
Google's `vulkan_layer` crate and runs the real shared-memory round trip on present, but
does **not** touch swapchain image contents yet — `queue_present_khr` always presents
the original, unmodified frame. Real capture/compose, device-extension injection for
the dma-buf transport, and the queue-family bookkeeping that capture needs are all
milestone 4. Verified via `cargo test -p dlssnr-layer` (the SHM round-trip state
machine, including a simulated echo-helper thread and the private-directory security
check) and `scripts/smoke-test.sh` (a real `VkInstance`/`VkDevice` through the system
Vulkan loader with the layer actually negotiated and inserted into the call chain —
this is what caught and fixed a real bug: `DlssnrDeviceInfo::new` originally panicked
on any device that didn't enable `VK_KHR_swapchain`, which would have crashed every
compute-only Vulkan app the layer got loaded into).

## Workspace layout

```
crates/
  protocol/   #[repr(C)] SHM wire format — the whole layer<->helper contract, no
              GTK/tokio deps, builds for both x86_64-unknown-linux-gnu and
              x86_64-pc-windows-gnu with identical layout.
  layer/      cdylib, x86_64-unknown-linux-gnu only. The Vulkan implicit layer.
  helper/     bin, x86_64-pc-windows-gnu only, runs under Wine/Proton. Never built or
              run natively on a Linux dev machine.
  gui/        bin, gtk4 + libadwaita, flat src/*.rs modules (same convention as
              GreenLight/KernelPop/SteamPunk).
  cli/        bin, the `dlssnr-cli` helper-manager (init/start/stop/status/doctor/
              runners/detect-gpu/import-binaries).
```

`helper` is excluded from the workspace's `default-members` (see root `Cargo.toml`), so
a **plain, flag-less** `cargo build`/`cargo test` on a Linux dev box never tries to link
Windows-only code against the host toolchain. **`--workspace`/`--all` ignore
`default-members` entirely** — that's documented Cargo behavior, not a bug here — so
`cargo build --workspace`/`cargo test --workspace` *will* try to build `helper` for the
host and fail at the link step the moment `helper` has any real `#[link(name =
"kernel32")]` code (confirmed: it linked fine back when `helper` was still an empty
stub with nothing to resolve, and fails now). **Use the plain, flag-less form for
"build/test everything except helper"** — `cargo build`, `cargo test`, `cargo check`
all correctly skip it that way. Reach for `-p dlssnr-helper` explicitly (with
`cargo check` for a quick pass, or the cross-compile toolchain below for the real
thing) rather than `--workspace` when `helper` itself is what you're working on.

`helper` needs `mingw-w64` installed (`apt install rustup gcc-mingw-w64-x86-64
binutils-mingw-w64-x86-64`, `rustup toolchain install stable`, `rustup target add
--toolchain stable-x86_64-unknown-linux-gnu x86_64-pc-windows-gnu`) plus a linker
override in `~/.cargo/config.toml` (`[target.x86_64-pc-windows-gnu] linker =
"x86_64-w64-mingw32-gcc"`, `ar = "x86_64-w64-mingw32-ar"`, `runner = "wine"`) — all now
set up on this dev machine (2026-09-09). **Important:** apt's `rustup` package
replaces `/usr/bin/cargo`/`/usr/bin/rustc` with its own toolchain-dispatch shims and
sets whichever toolchain you first `rustup toolchain install` as `default` — installing
a second (`stable`) toolchain just to get the `x86_64-pc-windows-gnu` target would have
silently switched the *whole project's* everyday Rust version out from under it. Fixed
by linking the pre-existing system install as its own named toolchain
(`rustup toolchain link system /usr/lib/rust-1.93`, then `rustup default system`) and
keeping `stable` around only for explicit cross-compiles
(`cargo +stable-x86_64-unknown-linux-gnu build --target x86_64-pc-windows-gnu -p
dlssnr-helper`) — plain `cargo`/`rustc` still resolve to 1.93.1 for everything else.
Real cross-compiling + linking against the mingw CRT is now verified working (see
`helper` gotchas below) — this is no longer "written but unverified."

## `protocol` gotchas

- **The three free-text fields** (`helper_reason`, `layer_reason`, `game_name`) are
  `UnsafeCell<[u8; N]>`, not plain byte arrays, specifically so mutating them through
  `&ShmHeader` (which is all any caller ever has — the header lives in an mmap'd region
  another process can write at any time, never behind a `&mut`) is defined behavior.
  Always go through `set_helper_reason`/`helper_reason`/etc., never touch the fields
  directly — they're private for exactly this reason. See `store_seq_guarded`/
  `load_seq_guarded` in `header.rs` for the seqlock-style protocol that makes a
  concurrent read-during-write never observe a torn string.
- **`ShmHeader` has no normal constructor.** Every real instance is a raw-pointer cast
  onto an existing mmap'd region (zero-filled by `ftruncate`, same as upstream's
  approach) — never a `ShmHeader { .. }` struct literal. `Default` exists only to give
  tests (and anything that wants a heap-allocated instance not backed by shared memory)
  the same all-zero starting point a fresh mapping already has; it is implemented via
  `mem::zeroed()` precisely because that's what a real mapping's bytes look like before
  `init_defaults()` runs, not because zero happens to be a convenient default. Don't
  "simplify" this into `#[derive(Default)]` — it doesn't compile (`[u8; 192]` has no
  `Default` impl in std beyond N=32), and even where it would, deriving would produce a
  *different* meaning than "matches a fresh mapping's raw bytes" for any field whose
  real, `init_defaults()`-assigned default isn't zero (most of the float fields
  default to `1.0`, not `0.0`).
- **The offset/size `const _: () = assert!(...)` block near the top of `header.rs`
  is load-bearing, not decorative.** If you add, remove, or reorder a field and one of
  these fails to compile, that's the check doing its job — bump `SHM_VERSION` in
  `lib.rs` in the same change, then recompute the asserted numbers (the easiest way:
  temporarily add a `#[test]` that prints `size_of`/`offset_of!` for the fields you
  need, run it, copy the numbers in, delete the test).
- **This protocol is not wire-compatible with upstream's**, on purpose — different
  magic (`SHM_MAGIC = "DSN1"` vs. upstream's `"GNR2"`), independent versioning starting
  at 1. A stale mapping from anything else must never be half-read as if it were ours.

## `layer` gotchas

- **`vulkan-layer` is a git dependency pinned to an exact commit**, not a crates.io
  release (it has none). `ash` in `crates/layer/Cargo.toml` is pinned to the *exact*
  version `vulkan-layer` itself depends on (`=0.37.3`) — Cargo treats different 0.x
  versions of the same crate as distinct, incompatible types, so a newer `ash` here
  would fail to compile against `vulkan_layer`'s trait signatures. If bumping the
  `vulkan-layer` git rev, check what `ash` version it depends on first and match it.
- **`panic = "abort"` is set workspace-wide** (root `Cargo.toml`, not per-crate — Cargo
  profiles other than a few per-package-overridable knobs apply to the whole build).
  Required for `dlssnr-layer`: it's a cdylib the Vulkan loader calls back into across a
  plain C ABI boundary, and an unwinding panic crossing that boundary is undefined
  behavior. `vulkan-layer`'s own example layers set the same thing for the same reason.
  `cargo test` still works fine with this set (confirmed on this toolchain).
- **Never resolve a next-in-chain device function pointer and assume it's there.**
  `device::resolve()` returns `Option<F>`, not `F` — a device that never enabled
  `VK_KHR_swapchain` (any compute-only app, or any device an app just never presents
  from) legitimately has no `vkCreateSwapchainKHR` to resolve. The first version of this
  panicked on a missing pointer; `scripts/smoke-test.sh`, which creates a device with no
  extensions enabled at all, caught it immediately. Every hook that depends on a
  resolved pointer checks for `None` first and returns `LayerResult::Unhandled` — that
  tells the `vulkan_layer` framework to fall through to its own next-dispatch exactly as
  if this crate weren't there, which is always correct when we have nothing useful to do
  anyway (an app that never enabled the extension will also never call the function).
- **`vulkan_layer`'s `DeviceInfo`/`InstanceInfo` traits don't hand a device hook access
  to its owning `ash::Instance`.** `create_device_info` gets `Arc<ash::Device>` and
  `vk::PFN_vkGetDeviceProcAddr` only — enough to resolve device/extension function
  pointers directly (what `device::resolve()` does), but not enough to call an
  instance-level function like `vkGetPhysicalDeviceProperties` from inside device-level
  code. This is why device-extension injection and the inert-on-non-NVIDIA-device check
  (both need `InstanceHooks::create_device`, which *does* get the physical device and
  whatever the `InstanceInfo`/`InstanceHooks` implementor stored from
  `create_instance_info`) are deferred to milestone 4 rather than solved here with a
  workaround.
- **Testing an implicit-type layer via `VK_LAYER_PATH` needs `VK_INSTANCE_LAYERS` too.**
  `VK_LAYER_PATH`/`VK_ADD_LAYER_PATH` only add manifests to the *explicit*-layer search;
  an implicit-type manifest found there is not auto-enabled the way a real one dropped
  into an actual `implicit_layer.d` directory would be (confirmed with
  `VK_LOADER_DEBUG=layer` — the manifest was found but never inserted into the call
  chain until `VK_INSTANCE_LAYERS=VK_LAYER_dlssnr_neural` was also set).
  `scripts/smoke-test.sh` sets both; a real install only ever needs
  `enable_environment` (see `data/VK_LAYER_dlssnr_neural.json`).
- **This sandbox has a real (software) Vulkan loader + ICD** (`libvulkan.so.1` +
  lavapipe, no `vulkaninfo` binary though) — `scripts/smoke-test.sh` genuinely creates a
  `VkInstance`/`VkDevice` through it, so it's worth running again after any change to
  `device.rs`/`lib.rs`, not just `cargo test`.

## `helper` gotchas

- **No `windows`/`windows-sys` dependency, on purpose.** Every WinAPI call
  (`LoadLibraryExW`, `GetProcAddress`, `VirtualProtect`, `AddVectoredExceptionHandler`,
  `CreateFileMappingW`, ...) is declared directly against `kernel32.dll` via
  `#[link(name = "kernel32")] extern "system" { ... }` blocks. These are among the
  oldest, most stable parts of the Win32 ABI (unchanged since Windows XP/Vista), and
  declaring them directly means correctness never depends on how some wrapper crate
  happens to shape its own bindings for them -- which mattered a lot while writing this
  with zero ability to compiler-check the Windows-specific paths (see below).
- **`cargo check -p dlssnr-helper` on the Linux host actually works and catches real
  bugs** -- `cargo check` only type-checks (`--emit=metadata`), never links, so it never
  needs to resolve the `kernel32` symbols the way `cargo build`/a real cross-compile
  would. It already caught two real mistakes (a private-struct-leaked-through-a-public-
  API error, an unused import) before any toolchain existed to build this for real.
  **What it does NOT catch**: anything about whether the raw offsets in `spoof.rs`'s
  `pe` module actually match a real PE image's layout, whether `setjmp`/`longjmp` even
  link successfully against the mingw CRT (see the next point), or any runtime
  behavior at all. Run it after every change to this crate regardless -- it's free and
  it's already proven itself worth running -- but it is not a substitute for the real
  cross-compile + Wine smoke test this crate still needs once the toolchain exists.
- **The IAT-patching recursion bug — the one real runtime bug found so far, and why
  `spoof.rs` has `REAL_GET_MODULE_FILE_NAME_W`.** The first version of
  `spoofed_get_module_file_name_w`'s pass-through branch (for a query about some module
  other than the spoofed one) called the plain `extern "system" fn GetModuleFileNameW`
  declared at the top of `spoof.rs`. That compiles fine and type-checks fine — the bug
  is purely about *which IAT slot that call resolves through at runtime*, invisible to
  the type system and therefore to `cargo check` no matter how carefully it's read.
  `examples/spoof_install_test.rs` (deliberately, for a self-contained test with no
  second DLL needed) installs the spoof against *its own* module — and every call to
  the plain import inside the same binary resolves through that binary's own single IAT
  slot for `GetModuleFileNameW`, which `install()` had just repointed at
  `spoofed_get_module_file_name_w` itself. Result: the pass-through branch called
  itself, forever, and the process stack-overflowed under Wine within milliseconds. Real
  usage (`ngx.rs` always patches a *different*, separately-loaded DLL — `nvngx_dlssnr.dll`
  or `nvngx.dll`, never this helper's own .exe) never hits this specific scenario, but
  relying on "the module I patch is never the one my own fallback call resolves through"
  as an unstated invariant was exactly the kind of latent landmine worth fixing outright
  rather than rationalizing away — matches what upstream's own C++ already does
  (`g_realGetModuleFileNameW`, captured once, used for every pass-through call,
  regardless of which module was patched). Fixed by capturing the *real* function
  pointer once in `install()` (before any patch) into a dedicated static and calling
  that for pass-through, never the plain import. **The general lesson**: when patching
  a well-known import in-process, never assume your own fallback call to "the real
  function" resolves anywhere other than through the exact slot you might have just
  patched — capture the original value explicitly instead of trusting which name you
  wrote in the source.
- **`guard.rs`'s `jmp_buf` (256 bytes, generously over-sized) and raw `setjmp`/`longjmp`
  linked and worked on the first real test** — `examples/guard_test.rs` triggers a real
  `0xC0000005` (`EXCEPTION_ACCESS_VIOLATION`) inside a `guard::guarded` closure (reading
  through a deliberately unmapped, non-null address — a *literal* null dereference gets
  intercepted by Rust's own debug-mode "unsafe precondition checks" and turned into a
  clean panic+abort before the CPU ever really faults, which was this test's own first,
  wrong version) and confirms the process survives, the guard returns the fail value,
  and execution continues normally afterward. Wine logs a benign
  `err:seh:RtlUnwindEx invalid frame` warning during this — expected and harmless: the
  whole point of the `longjmp`-based recovery is that it bypasses the *normal* SEH
  unwind bookkeeping Wine is noticing went missing, not that anything actually broke.
- **The PE-parsing offsets in `spoof.rs` are hardcoded from documented, ABI-stable
  struct layouts** (`IMAGE_DOS_HEADER`/`IMAGE_NT_HEADERS64`/`IMAGE_OPTIONAL_HEADER64`/
  `IMAGE_IMPORT_DESCRIPTOR`), not derived from modeling those structs as Rust types —
  deliberately, since a single misplaced field in the ~30-field `IMAGE_OPTIONAL_HEADER64`
  would silently shift every offset after it. **Confirmed correct against a real PE
  image**: `examples/spoof_test.rs` runs `find_imported_function_slot` against its own
  compiled `x86_64-pc-windows-gnu` binary (every such binary imports
  `GetModuleFileNameW` from `KERNEL32.dll` itself, so this needs no NVIDIA DLL at all)
  and confirms it finds a real, already-loader-resolved IAT slot — no NVIDIA DLL needed
  for this specific check.
- **`ngx.rs` proves the call sequence through `CreateFeature(18)` only.**
  `EvaluateFeature` needs real bound `DLSSNR.Color`/`Output`/`MVec` Vulkan resources
  (upstream's `NgxSetResources`) to mean anything; wiring that up is milestone 4's job,
  paired with the composition pass that owns those same resources. The 1x1 placeholder
  size in `create_feature_at` exists only to exercise the create call chain (and
  therefore the spoof and the guard) end to end -- it is not a real frame size and
  nothing should try to make it one. This part is still genuinely untested end-to-end
  (no real `nvngx_dlssnr.dll` available) — only the load/spoof-install/init plumbing
  around it has been.
- **`lib.rs` exists alongside `main.rs` specifically so `examples/` can exercise
  individual modules** (`guard`, `spoof`) directly without a full helper run — `main.rs`
  is now a thin binary wrapper around the `dlssnr_helper` library crate. Keep this
  split; it's what makes the Wine-based example tests above possible at all.

## `composition` (milestone 4, partial — math tested, GPU pipeline not wired up)

- **`crates/layer/src/composition/color.rs` and `downscale.rs` are real, tested pure
  Rust** (15 + 9 `#[test]`s covering OkLab/HPE-LMS round trips, the two-branch
  luminance rule's both branches, hue preservation at `colour_strength=0`, gamut
  compression being an exact identity when already in-gamut and luminance-preserving
  when not, every resampling kernel's defining properties). Two of those tests caught
  real mistakes *in the tests themselves*, not the implementation — worth reading
  `catmull_rom_matches_known_closed_form_at_half_sample`'s comment and
  `mitchell_netravali_default_is_deliberately_non_interpolating` before assuming a
  filter that doesn't equal 1 at `x=0` is broken; Mitchell-Netravali's own
  recommended default (B=1/3, C=1/3) is *supposed* to not interpolate exactly, trading
  that for smoothness — that's the whole point of the two-parameter family.
- **`crates/layer/shaders/compose.comp` is a hand-translated GLSL expression of that
  same, already-tested Rust math — but it has never been compiled, dispatched, or
  checked against the Rust reference it's supposed to match.** Neither `glslc` nor
  `glslangValidator` is installed on this dev machine (`glslang-tools` is
  apt-available, not installed); nothing has verified this file even parses as valid
  GLSL, let alone produces the same numbers as `color.rs`. Before relying on it:
  install `glslang-tools`, run `glslangValidator -S comp compose.comp` for a syntax
  check, then (the real test) dispatch it against a handful of known
  original/proxy/model triples and diff the GPU output against `color.rs`'s own
  functions run on the same inputs on the CPU.
- **Nothing calls this shader.** `queue_present_khr` in `crates/layer/src/device.rs`
  still always presents the original, unmodified frame (milestone 2's scope, unchanged).
  Wiring it in needs: a compute pipeline + descriptor sets bound to the transport's
  three image regions, `shaderc`/build-time SPIR-V compilation (`build.rs`), the
  queue-family bookkeeping and command pool `device.rs`'s module doc comment already
  flags as deferred, and the device-extension injection for the dma-buf transport. All
  of that is still open.

## `gui` (milestone 5, real — settings read/render correctly; write-back visually
## unconfirmed for an environment reason, not a code one)

- **Every settings row in `ui.rs` genuinely binds to a live `dlssnr_protocol::ShmHeader`
  field** via `shm.rs`'s `bind_float`/`bind_bool`/`bind_u32` (thin wrappers around the
  exact same `AtomicU32::load`/`store` + `control_seq` bump already unit-tested in
  `dlssnr_protocol`). Confirmed by screenshot (see the session transcript) that every
  group (Model, Motion, Composition, Status) renders and every value shown matches
  `ShmHeader::init_defaults()`'s real defaults exactly — this is reading the live
  mapping, not a static mock.
- **Could not confirm the write-back path by actually clicking a widget.** `xdotool`
  clicks (window-relative *and* absolute-screen-coordinate, on the correct window ID)
  did not register on this window at all in this dev sandbox — confirmed by testing
  against the title bar's own close button, which also failed to close the window.
  That isolates it to this sandbox's X11/input-routing setup (matching the
  `_NET_WM_DESKTOP` warning `xdotool` printed), not a bug in the UI: the write path is
  the same trivial `AtomicU32::store` the read path already proved works, wired through
  GTK's own standard `connect_active_notify`/`connect_selected_notify`/
  `connect_value_changed` signal callbacks. Worth an actual click-test outside this
  sandbox before fully trusting it, but there is no specific reason to expect it's wrong.
- **`dlssnr_protocol::mapping`** (Linux-only, `cfg(unix)`) is the "just open the
  mapping and read/write settings" utility both `gui` and `cli` share, deliberately
  kept separate from `dlssnr_layer::shm::ShmClient` (which is entangled with the
  request/response round-trip state machine the GUI/CLI have no reason to depend on).
- **NGX binaries import row, added later (2026-09-09)**: the Status group's "NGX
  binaries" row has a real "Import…" button — this was missing when README.md first
  claimed it existed (a documentation bug, caught when Alex went looking for it in the
  running app and couldn't find it). `crates/gui/src/binaries.rs` holds the path
  (`XDG_DATA_HOME/dlssnr/binaries`, duplicated from `cli/src/paths.rs::binaries_dir`
  rather than shared — four lines, not worth a shared crate) and the copy logic, real-
  tested by `cargo test -p dlssnr-gui` (`import_from_copies_known_files_and_skips_unknown_ones`,
  confirms known DLLs are copied and unrelated files are not). The button opens a
  `gtk4::FileDialog::select_folder`, copies via that same function, and shows an
  `adw::Toast` with the result. **Same sandbox limitation as the write-back path
  above blocks confirming the actual click-through**: verified instead by temporarily
  reducing `build_ui` to just the Status group so it would render without needing the
  scroll this sandbox also can't inject, screenshotting it (row and subtitle render
  correctly, "nvngx_dlssnr.dll missing" reflecting the real absence of the file), then
  reverting that temporary reduction — the underlying `import_from` logic is what the
  test above actually exercises.

## `cli` (milestone 5, real and tested — including one real bug caught by an actual
## process-group kill test)

- **Every subcommand was actually run** (`init`/`config`/`runners`/`detect-gpu`/
  `status`/`doctor`/`import-binaries`), not just compiled — output matches what each
  is supposed to report (correctly fell back to system `wine` with no Proton compat
  tools present, correctly reported no NVIDIA GPU on this dev machine's actual iGPU,
  `doctor` correctly exits non-zero while `nvngx_dlssnr.dll`/the helper exe are
  missing, exactly as it should).
- **`process.rs`'s `stop_kills_the_whole_process_group_not_just_the_leader` test is
  `#[ignore]`d, for a confirmed environment reason, not because the code is wrong.**
  Direct reproduction (see the session transcript) showed `kill()` — direct pid,
  negative-pid process-group, with or without a prior `setsid()`, all report `Ok(())`
  — never actually reaches a child spawned via `std::process::Command` from a
  compiled Rust binary *in this specific dev sandbox*, while an identical signal to a
  plain shell background job (`sleep 30 &` from a bash tool call, no Rust involved)
  **does** get delivered and acted on in the same environment. That isolates the gap to
  this sandbox's handling of signals to `Command`-spawned children specifically — the
  actual `signal_group`/`stop` implementation is the standard, portable POSIX pattern
  and needs no change to work on a real desktop running the real `helper.exe` under
  Wine/Proton. Un-ignore and rerun outside this sandbox before shipping if that's ever
  worth re-confirming.
- **`start`/`stop` are otherwise unexercised against a real helper** (no
  `nvngx_dlssnr.dll` on this machine to start a real session with) — the process
  supervision mechanics are tested per the point above; the actual runner-invocation
  command line (`proton run <helper.exe>` with the right env vars) has not been run
  for real.

## Docs and release (done)

`README.md`/`ATTRIBUTION.md`/`LICENSE` written fresh — not copied from upstream's own
README/ATTRIBUTION.md, which were read (cloned to `/tmp/dlss5vklayer-review` in an
earlier session) only to know what to credit, per the same "shape not expression" rule
as the code. `ATTRIBUTION.md` explicitly separates what's really taken (DLSS5VKLayer's
architecture/protocol shape, RenoDX's MIT-licensed composition design, Ottosson's OkLab
constants) from what's deliberately not taken (the GPL-3.0 OptiScaler/DLSS-NR shader
code upstream's own `ATTRIBUTION.md` admits it carries) and from the one place this
project *does* knowingly reproduce upstream's approach on purpose (the NGX
caller-identity spoof) — that section says so plainly rather than blending it in.
`v0.1.0` tagged and pushed as a GitHub release with the AppImage + `.zsync` sidecar as
assets — repo and release are both public, matching GreenLight/SteamPunk's convention
(confirmed via `gh repo view` before creating it).

## Build process (milestone 6, real and verified end-to-end — this actually produced
## a working AppImage in this session)

`build-appimage.sh` builds the release binaries (native `protocol`/`layer`/`gui`/`cli`,
cross-compiled `helper`), assembles the `AppDir`, packs it with `appimagetool`, and
generates the `.zsync` sidecar — modeled on GreenLight's own script, with **no
polkit/pkexec step at all** (this app needs one nowhere) and a `CARGO_HELPER`
environment override for this dev machine's own multi-toolchain setup (see "Current
state" above; a clean CI image needs no override, since it has only one toolchain to
begin with). **Actually run successfully this session**: produced a real, valid
`dlssnr-0.1.0-x86_64.AppImage` (confirmed `file`-typed as a real ELF PIE AppImage
runtime), which was then extracted and run (`--appimage-extract-and-run`) and launched
the real settings GUI with no errors — this is the one part of milestones 4-6 that got
genuinely full, real, end-to-end verification, precisely because it doesn't depend on
anything this dev machine lacks (a real NVIDIA GPU, `nvngx_dlssnr.dll`, working
`xdotool` input routing). The Vulkan layer manifest's `library_path` is correctly
rewritten from the checked-in `./libdlssnr_layer.so` (right for a manifest sitting
beside the `.so`) to the AppImage's real relative layout
(`../../lib/dlssnr/libdlssnr_layer.so`) during packaging — check this rewrite still
matches if the `AppDir` layout ever changes.
