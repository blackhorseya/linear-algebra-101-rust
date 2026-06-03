#!/usr/bin/env bash
# Vercel 部署 — 安裝階段。
#
# 重要:Vercel 的 build image 已預裝 Rust 於 /rust(CARGO_HOME/RUSTUP_HOME=/rust)。
# 千萬別用 rustup installer 重裝 —— build container 的 $HOME(/vercel)與 euid
# home(/root)不一致,重裝會走到 /vercel/.cargo/env 這條不存在的路而失敗。
# 直接把內建 toolchain 的 bin 掛上 PATH 即可。
set -euo pipefail

export PATH="/rust/bin:$PATH"
export CARGO_HOME="${CARGO_HOME:-/rust}"
export RUSTUP_HOME="${RUSTUP_HOME:-/rust}"

# 補上 wasm 編譯目標 + wasm-pack(官方 installer 下載預編譯 binary,落在 $CARGO_HOME/bin)。
rustup target add wasm32-unknown-unknown
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# 前端依賴。Vercel build image 內建的 pnpm 太舊,讀不懂 lockfileVersion 9.0
#(會「忽略 lockfile」再因 frozen install 失敗)。光靠 `corepack enable` + package.json
# 的 packageManager 欄位不夠 —— 內建舊 pnpm 在 PATH 更前面,會蓋過 corepack shim,
# 害版本欄位整個被忽略。所以把 shim 裝進自有目錄、prepend 到 PATH,確定跑的是我們的版本。
# COREPACK_ENABLE_DOWNLOAD_PROMPT=0:headless 環境讓 corepack 直接下載、不卡互動提示。
export COREPACK_ENABLE_DOWNLOAD_PROMPT=0
COREPACK_BIN="$(pwd)/.corepack-bin"
mkdir -p "$COREPACK_BIN"
corepack enable --install-directory "$COREPACK_BIN" pnpm
export PATH="$COREPACK_BIN:$PATH"
corepack prepare pnpm@11.5.1 --activate

cd web
# 診斷:確認真的是 11.5.1 在跑(不是內建舊版)。失敗時這行會直接點出兇手。
echo ">> pnpm 來源:$(command -v pnpm) / 版本:$(pnpm --version)"
pnpm install --frozen-lockfile
