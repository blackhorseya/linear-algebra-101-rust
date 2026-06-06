//! WASM binding —— 把 core 的純數學接到瀏覽器,薄 adapter,core 零改動。
//!
//! 整個模組鎖在 `#[cfg(feature = "wasm")]` 後面(見 `lib.rs` 的 gated `pub mod wasm`),
//! 不開 feature 時等於不存在:`cargo test` / `task check` 看不到它,也不把
//! `wasm-bindgen` 拉進依賴樹。
//!
//! 原則:**計算只在 Rust 一份**。JS 只負責 Canvas 繪圖與滑鼠事件,每個變換後的點
//! 都是 core 的 [`multiply_vector`](crate::Matrix::multiply_vector) 算的、每個平行
//! 判定都是 [`is_parallel`](crate::Vector::is_parallel) 算的 —— JS 不重寫任何線代。
//!
//! 結構沿 core 的慣例「一個概念一個模組」:一個視覺化章一支子模組,本檔只負責
//! `mod` 宣告與 `pub use` re-export(鏡像 `lib.rs`);跨章共用的邊界工具收在私有的
//! `helpers`。`#[wasm_bindgen]` 的匯出一律攤平在套件根層、與 Rust 模組巢狀無關,
//! 所以拆檔對 JS 端 API 零影響。

mod helpers;

pub mod elimination;
pub mod inverse;
pub mod linearity;
pub mod multiply;
pub mod range;
pub mod standard_matrix;
pub mod transform;

pub use elimination::{EliminationTrace, eliminate};
pub use inverse::{InverseTrace, invert_trace};
pub use linearity::{add_vectors, check_linearity, scale_vector};
pub use multiply::{MultiplyExpansion, can_multiply, multiply_expand};
pub use range::{is_onto, range_basis, range_contains, solve_for_input, unreachable_vector};
pub use standard_matrix::{apply_rule, sample_standard_matrix};
pub use transform::{are_parallel, determinant, transform_point};
