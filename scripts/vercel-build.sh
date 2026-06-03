#!/usr/bin/env bash
# Vercel 部署 — 建置階段。
#
# 與 install 是不同的 shell,沿用同一個內建 Rust(/rust),重新把 bin 掛上 PATH。
set -euo pipefail

export PATH="/rust/bin:$PATH"
export CARGO_HOME="${CARGO_HOME:-/rust}"
export RUSTUP_HOME="${RUSTUP_HOME:-/rust}"

# 1) core 編成 WASM,產物落在 web/src/lib/wasm(gitignore)。--out-dir 相對 crate root。
wasm-pack build --target web --out-dir web/src/lib/wasm -- --features wasm

# 2) Vite 把上面的 .wasm 當 asset 打包進 web/dist。
cd web && pnpm build
