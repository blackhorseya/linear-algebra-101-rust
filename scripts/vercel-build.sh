#!/usr/bin/env bash
# Vercel 部署 — 建置階段。
#
# 與 install 階段是不同的 shell,install 階段 source 進來的 cargo 環境
# 不會延續,所以這裡要重新 source 一次才找得到 wasm-pack。
set -euo pipefail

. "$HOME/.cargo/env"

# 1) 先把 core 編成 WASM,產物落在 web/src/lib/wasm(gitignore,衍生物不入版控)。
#    --out-dir 相對 crate root,故在 repo 根目錄執行。
wasm-pack build --target web --out-dir web/src/lib/wasm -- --features wasm

# 2) 再讓 Vite 把上面的 .wasm 當 asset 打包進 web/dist。
cd web && pnpm build
