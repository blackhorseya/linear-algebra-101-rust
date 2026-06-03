#!/usr/bin/env bash
# Vercel 部署 — 安裝階段。
#
# Vercel 的 build image 只有 Node/pnpm,沒有 Rust 工具鏈,所以這裡補齊
# Rust + wasm32 target + wasm-pack,最後才裝前端依賴。
# (抽成 script 是因為 vercel.json 的 installCommand 有 256 字元上限。)
set -euo pipefail

# minimal profile:只裝 rustc/cargo,略過 docs/clippy 等,加快安裝。
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"

# wasm 編譯目標 + wasm-pack(官方 installer 下載預編譯 binary,比 cargo install 快)。
rustup target add wasm32-unknown-unknown
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# 前端依賴(Vercel build image 內建 corepack 提供 pnpm)。
corepack enable
cd web && pnpm install --frozen-lockfile
