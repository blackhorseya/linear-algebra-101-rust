//! linear-algebra-101 —— 用 Rust 從零實作線性代數,邊寫邊學。
//!
//! 依照原始 Go 專案的 git log 順序逐步移植,每一步對應一個 commit。
//! 目前進度:Matrix(運算 + Theorem 1.1/1.2 + matrix-vector product)、
//! Vector(運算 + linear_combination + 標準基底 eᵢ)。

pub mod error;
pub mod matrix;
pub mod vector;

pub use error::LinAlgError;
pub use matrix::Matrix;
pub use vector::Vector;
