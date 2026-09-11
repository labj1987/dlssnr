#!/usr/bin/env bash
# build-appimage.sh — build the dlssnr AppImage.
# Run from the repo root on Ubuntu (matches GreenLight/KernelPop/SteamPunk's own CI
# assumption). Run as root in CI.
#
# Unlike GreenLight/KernelPop, this app needs no polkit/pkexec step at all -- every
# path it touches (~/.local/share, ~/.config, /tmp/dlssnr-$UID/) is already
# user-owned, so AppRun just execs the GUI directly.
set -euo pipefail

APP="dlssnr"
VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
ARCH="x86_64"
BUILD_DIR="build-appimage"
APPDIR="$BUILD_DIR/AppDir"
WIN_TARGET="x86_64-pc-windows-gnu"

# On a from-scratch CI image, `cargo`/`rustup` are already the only Rust toolchain, so
# plain `cargo` and `rustup target add` are correct there. On this repo's own dev
# machine specifically, a pre-existing non-rustup toolchain was deliberately kept as
# the `system` default (see CLAUDE.md's "helper gotchas") and the cross-compile
# toolchain lives under a separate `stable-x86_64-unknown-linux-gnu` rustup toolchain
# instead -- override CARGO_HELPER via the environment if building there.
CARGO_HELPER="${CARGO_HELPER:-cargo}"

echo "==> Building $APP $VERSION AppImage"

# ── Build dependencies ────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1 || ! pkg-config --exists gtk4 2>/dev/null; then
    echo "==> Installing build dependencies"
    # Tolerate an unrelated third-party repo (e.g. a runner image's preinstalled
    # Google Chrome source) failing to refresh -- apt falls back to its cached index
    # for that repo and still refreshes everything else; only `apt-get install`
    # failing on a package we actually need should be fatal.
    apt-get update -qq || true
    apt-get install -y -qq cargo rustc libgtk-4-dev libadwaita-1-dev \
        pkg-config libssl-dev wget file desktop-file-utils zsync \
        mingw-w64 gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64
fi
# Add the target to whichever toolchain `$CARGO_HELPER` will actually invoke, not
# necessarily the default one -- on a clean CI image there's exactly one toolchain and
# `+something` never appears in `$CARGO_HELPER`, so this reduces to the obvious
# `rustup target add`. On this repo's own dev machine, the default toolchain is a
# linked, non-rustup-managed one that `rustup target add` can't touch at all (see
# CLAUDE.md) -- `CARGO_HELPER="cargo +stable-..."` there means the target has to be
# added to *that* toolchain instead.
rustup_toolchain_arg=""
for word in $CARGO_HELPER; do
    case "$word" in
        +*) rustup_toolchain_arg="--toolchain ${word#+}" ;;
    esac
done
if ! rustup target list --installed ${rustup_toolchain_arg:+$rustup_toolchain_arg} 2>/dev/null | grep -q "$WIN_TARGET"; then
    # shellcheck disable=SC2086 -- word-splitting $rustup_toolchain_arg is intentional here.
    rustup target add $rustup_toolchain_arg "$WIN_TARGET"
fi

# ── Release build ─────────────────────────────────────────────────────
echo "==> cargo build --release (protocol/layer/gui/cli)"
cargo build --release

echo "==> $CARGO_HELPER build --release --target $WIN_TARGET -p dlssnr-helper"
$CARGO_HELPER build --release --target "$WIN_TARGET" -p dlssnr-helper

# ── AppDir layout ─────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
mkdir -p "$APPDIR/usr/bin" \
         "$APPDIR/usr/lib/$APP/helper" \
         "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/scalable/apps" \
         "$APPDIR/usr/share/metainfo" \
         "$APPDIR/usr/share/vulkan/implicit_layer.d"

cp "target/release/$APP-gui"                       "$APPDIR/usr/bin/"
cp "target/release/$APP-cli"                       "$APPDIR/usr/bin/"
cp "target/release/lib${APP}_layer.so"             "$APPDIR/usr/lib/$APP/"
cp "target/$WIN_TARGET/release/${APP}_helper.exe"  "$APPDIR/usr/lib/$APP/helper/"
sed "s#\./lib${APP}_layer\.so#../../lib/$APP/lib${APP}_layer.so#" \
    "data/VK_LAYER_${APP}_neural.json" > "$APPDIR/usr/share/vulkan/implicit_layer.d/VK_LAYER_${APP}_neural.json"
cp data/$APP.desktop                               "$APPDIR/usr/share/applications/"
cp data/icon.svg                                   "$APPDIR/usr/share/icons/hicolor/scalable/apps/$APP.svg"
cp data/io.github.labj1987.Dlssnr.appdata.xml       "$APPDIR/usr/share/metainfo/"

# Top-level AppImage requirements
cp data/$APP.desktop "$APPDIR/"
cp data/icon.svg "$APPDIR/$APP.svg"

# ── AppRun ────────────────────────────────────────────────────────────
cat > "$APPDIR/AppRun" << 'APPRUN'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
# The Vulkan layer manifest ships inside the AppImage's own read-only tree, so the
# loader needs an explicit path to it -- there is no writable implicit_layer.d this
# install owns to drop it into (this app needs no root/install step at all).
export VK_ADD_LAYER_PATH="$HERE/usr/share/vulkan/implicit_layer.d${VK_ADD_LAYER_PATH:+:$VK_ADD_LAYER_PATH}"
exec "$HERE/usr/bin/dlssnr-gui" "$@"
APPRUN
chmod 755 "$APPDIR/AppRun"

# ── appimagetool ──────────────────────────────────────────────────────
TOOL="$BUILD_DIR/appimagetool"
if [[ ! -f "$TOOL" ]]; then
    echo "==> Downloading appimagetool"
    wget -q -O "$TOOL" \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x "$TOOL"
fi

echo "==> Packing AppImage"
OUT="$APP-$VERSION-$ARCH.AppImage"

# Real repo is "dlssnr" (lowercase) -- "Dlssnr" here was a real, confirmed bug found
# 2026-09-11: GitHub's own API/web redirects resolve a case-mismatched repo name
# fine (confirmed via `gh api repos/labj1987/Dlssnr/...`), but Gear Lever's own
# update-check client apparently does not -- a real user report ("no updates
# found" despite a genuinely newer release existing) is what caught this, not
# inspection alone. Match the real repo's exact casing, don't rely on any client
# redirecting a mismatch correctly.
UPDATE_INFORMATION="gh-releases-zsync|labj1987|dlssnr|latest|$APP-*-x86_64.AppImage.zsync"
VERSION="$VERSION" ARCH="$ARCH" "$TOOL" --appimage-extract-and-run \
    -u "$UPDATE_INFORMATION" "$APPDIR" "$OUT"

echo "==> Done: $OUT"
ls -lh "$OUT"

# appimagetool's built-in zsync generation silently no-ops on some CI runners (see
# KernelPop's CLAUDE.md); build the sidecar directly instead. Non-fatal.
#
# `-u <url>` here is a *second*, different piece of update metadata than
# `UPDATE_INFORMATION` above: it's the .zsync file's own internal "URL:" header,
# read by whatever HTTP client actually fetches the new AppImage bytes once a zsync
# client has decided (via UPDATE_INFORMATION's gh-releases-zsync scheme) that an
# update exists. Without it, zsyncmake defaults to a bare relative filename, which
# only resolves correctly if a client does real relative-URL resolution against
# wherever it fetched this .zsync from -- not guaranteed. Point it at this exact
# release's real, absolute GitHub download URL instead of relying on that.
ZSYNC_URL="https://github.com/labj1987/dlssnr/releases/download/v$VERSION/$OUT"
echo "==> Generating .zsync sidecar"
if zsyncmake -u "$ZSYNC_URL" "$OUT"; then
    echo "==> .zsync generated: $OUT.zsync"
else
    echo "==> WARNING: zsyncmake failed — continuing without .zsync"
fi
