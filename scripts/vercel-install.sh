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

# 前端依賴(Vercel build image 內建 corepack 提供 pnpm)。
corepack enable
cd web && pnpm install --frozen-lockfile
