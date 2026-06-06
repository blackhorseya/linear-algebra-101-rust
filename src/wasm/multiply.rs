//! 矩陣乘法 row × col 展開 —— 給前端「點 C 的任一格,看 A 第 i 列 · B 第 j 欄
//! 的 dot product 攤開」圖解用。
//!
//! 設計取向(與 core 的關係):core 的 `multiply` 只回最終的 C,把每格的展開項
//! a_ik·b_kj 吃掉了;且鐵律是 **core 零改動**。所以 C 本身仍由 core 計算(單一
//! 真相),展開項在 binding 層補算(沿 `determinant` 的「薄運算」慣例,計算仍只
//! 在 Rust 一份),並以下方測試對帳 Σₖ terms == c_ij,漂移會被 `cargo test` 抓到。

use super::helpers::flatten;
use crate::Matrix;
use wasm_bindgen::prelude::*;

/// `A(a_rows×a_cols)` 能否右乘 `B(b_rows×b_cols)`(內維相等)。
///
/// 維度規則的單一真相在 core:用兩個零矩陣把尺寸帶進 [`Matrix::can_multiply`]
/// 問答案,binding 不重寫 `a_cols == b_rows` 這條判斷。
#[wasm_bindgen]
pub fn can_multiply(a_rows: usize, a_cols: usize, b_rows: usize, b_cols: usize) -> bool {
    Matrix::new(a_rows, a_cols).can_multiply(&Matrix::new(b_rows, b_cols))
}

/// 一次矩陣乘法的結果與逐格展開,過 WASM 邊界的單一物件。
///
/// 過邊界策略同 [`EliminationTrace`](super::elimination::EliminationTrace):SoA,
/// 每個欄位一條 typed array,前端 wrapper 縫回 plain-JS 物件後 `free()`。
#[wasm_bindgen]
pub struct MultiplyExpansion {
    rows: usize,     // C 的列數 m(= A 的列數)
    cols: usize,     // C 的欄數 p(= B 的欄數)
    inner: usize,    // 內維 n(= A 的欄數 = B 的列數)
    c: Vec<f64>,     // C = A·B,row-major flatten(m×p),由 core 的 multiply 計算
    terms: Vec<f64>, // 展開項:terms[(i·p + j)·n + k] = a_ik·b_kj(m×p×n)
}

#[wasm_bindgen]
impl MultiplyExpansion {
    // --- 純量 getter(在 JS 端是 property)---
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[wasm_bindgen(getter)]
    pub fn inner(&self) -> usize {
        self.inner
    }

    // --- SoA 陣列 getter(各跨界一次)---
    pub fn c(&self) -> Vec<f64> {
        self.c.clone()
    }
    pub fn terms(&self) -> Vec<f64> {
        self.terms.clone()
    }
}

/// 矩陣乘法 `C = A·B` 與每格的 row × col 展開項。
///
/// - `a_data` / `b_data`:row-major flatten 的元素,長度須為 `rows * cols`
///   (前端保證,故用 reshape 不檢查;沿 `eliminate` 慣例)。
/// - 維度相容性由前端先以 [`can_multiply`] 確認後才呼叫 —— 此處的 `expect`
///   把「不會發生」寫成自證的不變式(沿 `transform_point` 慣例)。
#[wasm_bindgen]
pub fn multiply_expand(
    a_data: Vec<f64>,
    a_rows: usize,
    a_cols: usize,
    b_data: Vec<f64>,
    b_rows: usize,
    b_cols: usize,
) -> MultiplyExpansion {
    // b_rows 只用來自證前置條件(內維相等);reshape 本身由 chunks 依欄數完成。
    debug_assert_eq!(a_cols, b_rows, "前端先以 can_multiply 檢查過維度");
    let a = Matrix::from_rows(a_data.chunks(a_cols).map(<[f64]>::to_vec).collect());
    let b = Matrix::from_rows(b_data.chunks(b_cols).map(<[f64]>::to_vec).collect());

    // C 的單一真相:core 的 multiply。
    let c = a.multiply(&b).expect("前端先以 can_multiply 檢查過維度");

    // 展開項 a_ik·b_kj:把每格 c_ij 的 dot product 攤開,給前端帶實際數字顯示。
    let mut terms = Vec::with_capacity(a_rows * b_cols * a_cols);
    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                terms.push(a.row(i).unwrap()[k] * b.row(k).unwrap()[j]);
            }
        }
    }

    MultiplyExpansion {
        rows: a_rows,
        cols: b_cols,
        inner: a_cols,
        c: flatten(&c),
        terms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::helpers::input_matrix;

    #[test]
    fn can_multiply_requires_inner_dims_match() {
        assert!(can_multiply(2, 3, 3, 2)); // (2×3)·(3×2):內維 3 = 3 ✓
        assert!(!can_multiply(2, 3, 2, 3)); // (2×3)·(2×3):內維 3 ≠ 2 ✗
        assert!(can_multiply(1, 4, 4, 1)); // 列向量 · 欄向量 → 1×1
        assert!(can_multiply(2, 2, 2, 2)); // 同維方陣必可乘
    }

    /// **黃金迴歸**:`multiply_expand` 的 C 必須等於 core 的 `multiply`(整數在 f64
    /// 下精確,可用 `assert_eq!`),且每格 c_ij 必須等於其展開項之和、每項必須等於
    /// a_ik·b_kj —— binding 補算的展開項與 core 的乘法對帳,漂移即測試失敗。
    #[test]
    fn multiply_expand_terms_reconcile_with_core() {
        // 經典 (2×3)·(3×2) → 2×2:C = [[58, 64], [139, 154]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let exp = multiply_expand(a.clone(), 2, 3, b.clone(), 3, 2);

        assert_eq!((exp.rows, exp.cols, exp.inner), (2, 2, 3));

        // C == core 的 multiply
        let core_c = input_matrix(&a, 3).multiply(&input_matrix(&b, 2)).unwrap();
        assert_eq!(exp.c, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(exp.c, flatten(&core_c));

        // 每項 terms[(i·p+j)·n+k] == a_ik·b_kj,且 Σₖ == c_ij
        let (p, n) = (exp.cols, exp.inner);
        for i in 0..exp.rows {
            for j in 0..p {
                let base = (i * p + j) * n;
                for k in 0..n {
                    assert_eq!(exp.terms[base + k], a[i * 3 + k] * b[k * 2 + j]);
                }
                let sum: f64 = exp.terms[base..base + n].iter().sum();
                assert_eq!(sum, exp.c[i * p + j]);
            }
        }
    }
}
