//! linear-algebra-101 —— 用 Rust 從零實作線性代數,邊寫邊學。
//!
//! 依照原始 Go 專案的 git log 順序逐步移植,每一步對應一個 commit。
//! 目前進度:Matrix(運算 + Theorem 1.1/1.2/1.3 + identity/stochastic/column)、
//! Vector(運算 + linear_combination + 標準基底 eᵢ)、System(線性方程組 Ax=b)。

pub mod basis;
pub mod composition;
pub mod coordinates;
pub mod determinant;
pub mod diagonal;
pub mod elimination;
pub mod error;
pub mod independence;
pub mod inverse;
pub mod matrix;
pub mod predicate_set;
pub mod range;
pub mod span;
pub mod system;
pub mod transformation;
pub mod vector;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use basis::{is_basis, is_standard_basis};
pub use composition::TransformationReport;
pub use coordinates::{coordinates, from_coordinates};
pub use diagonal::DiagonalMatrix;
pub use error::LinAlgError;
pub use independence::{
    is_linearly_dependent, is_linearly_independent, redundancy_count, removable_columns,
};
pub use matrix::Matrix;
pub use predicate_set::PredicateSet;
pub use span::{Span, affine_span, on_line, on_plane};
pub use system::{RowKind, Solution, System};
pub use transformation::{Transformation, standard_matrix, verify_linearity};
pub use vector::Vector;
