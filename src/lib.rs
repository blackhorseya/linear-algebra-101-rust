//! linear-algebra-101 —— 用 Rust 從零實作線性代數,邊寫邊學。
//!
//! 依照原始 Go 專案的 git log 順序逐步移植,每一步對應一個 commit。
//! 目前進度:Matrix 基礎運算(equality / add / scalar multiply)。

pub mod matrix;

pub use matrix::{LinAlgError, Matrix};
