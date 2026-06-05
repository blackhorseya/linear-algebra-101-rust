# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 這個 repo 是什麼

用 Rust 從零手刻 linear algebra 的**學習型 library**,目的是建立對線性代數的直覺,不是做高效能數值庫。

- **刻意不依賴** `nalgebra` / `ndarray` 之類的數值 crate,vector、matrix 及其運算全部親手寫。除非有極強理由,否則不要引入外部數值依賴。唯一的 dev-dependency 是 `proptest`(測試工具,非數值依賴,見下方測試慣例)。
- 純 **library crate**:沒有 `main.rs`、沒有 bin target。`Cargo.lock` 被 `.gitignore` 排除且未追蹤(library 慣例)—— 不要把它 commit 進去。
- edition 2024,需要 Rust 1.85 以上。

## 最重要的工作流程:逐 commit 移植 Go 版

core(`src/` 的數學模組)是 [linear-algebra-101](https://github.com/blackhorseya/linear-algebra-101)(Go 版)的 Rust 改寫,**嚴格依 Go 專案 git log 正序逐步移植 —— 一個 Rust commit 對應一個 Go commit**。這就是為何原始碼註解常標「對應原始 Go 專案的哪個 commit」—— 延續這個習慣。

**Go 可移植內容已全數移完**(收尾於 `coordinates` 模組)。若還要移植某個 Go commit,先確認它做了什麼,再用 Rust 的型別與慣例重寫(不是逐行翻譯,而是「同一個數學概念,換 Rust 的方式表達」)。**Go 之外的延伸(如下方的視覺化軌道)不受此移植順序約束**,不必硬去對應某個 Go commit。

**移植完成後的 core 新功能由使用者的線性代數學習筆記驅動**(如 `multiply` / `power` / `DiagonalMatrix` 這條乘法章):題目由使用者從筆記給出,不要自行推測下一章該做什麼。

學習路徑與進度勾選見 `README.md`(Vector → Matrix → 向量空間 → 線性方程組與分解 → 進階主題)。**新增數學功能時,同一個 commit 內同步更新 README 的勾選與條目**(歷來 feature commit 都這麼做)。

## 常用指令

專案用 [Taskfile](https://taskfile.dev)(go-task)包裝 cargo 指令。

| 指令 | 作用 |
|------|------|
| `task` | 列出所有 task |
| `task test` | 跑全部測試(`cargo test`) |
| `task test:v` | 顯示捕捉的 stdout(對應 Go 的 `go test -v`) |
| `task check` | **pre-commit gate**:依序跑 `fmt:check` → `lint` → `test` |
| `task lint` | Clippy,`-D warnings`(warnings 視為錯誤) |
| `task fmt` / `task fmt:check` | rustfmt 格式化 / 僅檢查 |
| `task cover` | 覆蓋率表(需先 `cargo install cargo-llvm-cov`) |
| `task cover:lines` | 列出未覆蓋的行 |
| `task cover:html` | 開 HTML 覆蓋率報告 |
| `task wasm:build` | 用 wasm-pack 建 WASM 套件到 `web/src/lib/wasm`(需 wasm-pack 0.15+ 與 `wasm32-unknown-unknown` target) |

跑單一測試:`cargo test add_sums_elementwise`;跑整個模組:`cargo test matrix::tests`。

提交前務必通過 `task check`。

## 程式碼結構與慣例

- **一個概念一個模組**:`vector` / `matrix` / `diagonal` / `span` / `independence` / `basis` / `coordinates` / `system` / `elimination` / `predicate_set` 各一支 `src/*.rs`,`error.rs` 放共用的橫切錯誤型別。`src/lib.rs` 只負責 `pub mod` 與 `pub use` re-export 公開 API。
- **測試與實作同檔**:inline `#[cfg(test)] mod tests`,white-box 測試(可存取 private 欄位)。
- **白箱建構 helper**:測試裡用 `matrix_from(data: Vec<Vec<f64>>)` 直接從字面值建出 `Matrix`,繞過正式建構子(對應 Go 的 `matrixFrom`)。新增型別時沿用這個模式。
- **example test 與 property test 分兩個模組**:具體案例放 `mod tests`;教材定理(「for all」形式的代數律)另立 `#[cfg(test)] mod laws`,用 proptest 隨機驗證。`mod laws` 的慣例:整數策略(如 `int_matrix`,元素為小整數 —— f64 下加減乘完全精確)配精確 `equals`;真實浮點策略(如 `real_matrix`)配 `approx_equals(_, 1e-9)`。新增定理對應的運算時,優先考慮能不能寫成 law。
- **`proptest-regressions/` 被 gitignore**:本專案的 property test 失敗代表「該當場修」,不長期保存反例檔。

### 設計慣例(跨檔案的「big picture」)

- **錯誤是值,不是 panic**:可能失敗的運算(如維度不合)回傳 `Result<_, LinAlgError>`,用型別把失敗可能性逼到呼叫端面前。不要在運算裡 panic。
- **單一手刷錯誤 enum**:`LinAlgError`(在 `error.rs`,作為跨概念共用的橫切型別)不依賴 `thiserror` / `anyhow`,自行 impl `Display` + `std::error::Error`。新錯誤種類加 variant 到這個 enum,讓呼叫端能用 `match` 精確區分。
- **維度從資料導出,不另存**:`Matrix` 內部是 private `data: Vec<Vec<f64>>`(row-major),`rows()` / `cols()` 從 `data` 算出來 —— `data` 是唯一真相來源,沒有「維度欄位與資料對不上」的不變式要維護。欄位一律 private,只透過方法存取。
- **浮點比較用 `approx_equals(other, epsilon)`**,不要在浮點運算後用精確的 `equals`;容差 `epsilon` 由呼叫端視運算數量級指定。

## 視覺化 / WASM 軌道

`web/` 是 core 的互動視覺化前端(React 19 + Vite + TanStack Router/Query + Tailwind v4),透過 WASM 呼叫 core 做「矩陣作為 2D 線性變換」等視覺化。幾條鐵律:

- **core 零改動**:WASM binding 全鎖在 `#[cfg(feature = "wasm")]`(`src/wasm.rs`,`lib.rs` 以 gated `pub mod wasm` 宣告)。`wasm-bindgen` 是 optional dep + `dep:` feature,沒開 `wasm` feature 時不進依賴樹 —— `cargo test` / `task check` 完全不受影響。新增視覺化功能只在 binding 層轉呼叫 core,**不為了前端去改 core**。
- **計算單一真相在 Rust**:JS 只負責 Canvas 繪圖與互動,每個結果都由 core 計算,不要在 JS 重寫任何線代。
- **最小依賴**:binding 只加 `wasm-bindgen`,前端不引繪圖庫(純 Canvas 2D)—— 延續 core 的精神。
- **建置**:`task wasm:build` 產物到 `web/src/lib/wasm`(已 gitignore,衍生物不入版控)。需 **wasm-pack 0.15+**(舊版內建的 wasm-opt 看不懂新 wasm-bindgen 產的多 table 區段)。
- **`web/` 一律用 `pnpm`,不要 `npm`**。

### Vercel 部署

`vercel.json` 把 install / build 階段指到 `scripts/vercel-install.sh` / `scripts/vercel-build.sh`(WASM 編譯 + Vite 打包)。這兩支 script 的寫法是多次部署失敗換來的,**改動前先讀 script 內的註解**,兩個坑:

- Vercel build image **已預裝 Rust 於 `/rust`**,不要用 rustup installer 重裝(container 的 `$HOME` 與 euid home 不一致會直接失敗),把 `/rust/bin` 掛上 PATH 沿用即可。
- image 內建的 pnpm 太舊,讀不懂 lockfileVersion 9;光靠 `corepack enable` + `packageManager` 欄位不夠(內建舊版在 PATH 更前面會蓋過 corepack shim),必須把 corepack shim 裝進自有目錄並 **prepend** 到 PATH,再 `corepack prepare pnpm@<版本> --activate` 釘住版本。

## 提交規範

- commit 與 PR 標題用 **Conventional Commits**,帶 scope:`feat(matrix): ...`、`refactor(matrix): ...`、`chore: ...`。
- 一律使用繁體中文溝通,英文技術名詞保留原文。
