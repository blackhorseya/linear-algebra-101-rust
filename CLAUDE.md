# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 這個 repo 是什麼

用 Rust 從零手刻 linear algebra 的**學習型 library**,目的是建立對線性代數的直覺,不是做高效能數值庫。

- **刻意不依賴** `nalgebra` / `ndarray` 之類的數值 crate,vector、matrix 及其運算全部親手寫。除非有極強理由,否則不要引入外部數值依賴。
- 純 **library crate**:沒有 `main.rs`、沒有 bin target。`Cargo.lock` 被 `.gitignore` 排除且未追蹤(library 慣例)—— 不要把它 commit 進去。
- edition 2024,需要 Rust 1.85 以上。

## 最重要的工作流程:逐 commit 移植 Go 版

本專案是 [linear-algebra-101](https://github.com/blackhorseya/linear-algebra-101)(Go 版)的 Rust 改寫,並**嚴格依照 Go 專案 git log 的正序逐步移植 —— 一個 Rust commit 對應一個 Go commit**。

實作下一個功能前,先確認對應的 Go commit 做了什麼,再用 Rust 的型別與慣例重寫(不是逐行翻譯,而是「同一個數學概念,換 Rust 的方式表達」)。原始碼註解常會標明「對應原始 Go 專案的哪個 commit」,延續這個習慣。

學習路徑與進度勾選見 `README.md`(Vector → Matrix → 線性方程組與分解 → 進階主題)。

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

跑單一測試:`cargo test add_sums_elementwise`;跑整個模組:`cargo test matrix::tests`。

提交前務必通過 `task check`。

## 程式碼結構與慣例

- **一個概念一個模組**:`src/matrix.rs`、未來的 `src/vector.rs` 等。`src/lib.rs` 只負責 `pub mod` 與 `pub use` re-export 公開 API。
- **測試與實作同檔**:inline `#[cfg(test)] mod tests`,white-box 測試(可存取 private 欄位)。
- **白箱建構 helper**:測試裡用 `matrix_from(data: Vec<Vec<f64>>)` 直接從字面值建出 `Matrix`,繞過正式建構子(對應 Go 的 `matrixFrom`)。新增型別時沿用這個模式。

### 設計慣例(跨檔案的「big picture」)

- **錯誤是值,不是 panic**:可能失敗的運算(如維度不合)回傳 `Result<_, LinAlgError>`,用型別把失敗可能性逼到呼叫端面前。不要在運算裡 panic。
- **單一手刷錯誤 enum**:`LinAlgError`(在 `error.rs`,作為跨概念共用的橫切型別)不依賴 `thiserror` / `anyhow`,自行 impl `Display` + `std::error::Error`。新錯誤種類加 variant 到這個 enum,讓呼叫端能用 `match` 精確區分。
- **維度從資料導出,不另存**:`Matrix` 內部是 private `data: Vec<Vec<f64>>`(row-major),`rows()` / `cols()` 從 `data` 算出來 —— `data` 是唯一真相來源,沒有「維度欄位與資料對不上」的不變式要維護。欄位一律 private,只透過方法存取。
- **浮點比較用 `approx_equals(other, epsilon)`**,不要在浮點運算後用精確的 `equals`;容差 `epsilon` 由呼叫端視運算數量級指定。

## 提交規範

- commit 與 PR 標題用 **Conventional Commits**,帶 scope:`feat(matrix): ...`、`refactor(matrix): ...`、`chore: ...`。
- 一律使用繁體中文溝通,英文技術名詞保留原文。
