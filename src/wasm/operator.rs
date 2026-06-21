//! 線性運算子的矩陣表示(單元 7-3)—— 給前端「相似不變量」圖解用:固定運算子 A,
//! 拖斜格基底 B「換尺」,看 [T]_B = B⁻¹AB 的四格即時變化,而 det([T]_B) 永遠釘在 det(A)
//! (相似不變量)。標準基底時 [T]_B = A。
//!
//! 與 core 的關係:委派 [`b_matrix`](crate::b_matrix)。眉角是 `b_matrix` 收的是 **closure**
//! `Fn(&Vector) -> Vector` —— **行為跨不了 WASM 邊界**(只有資料能過界)。故 binding 收
//! 運算子 A 的**四個純量**(過界的是資料),在 Rust 端用它造 closure `|v| A·v` 餵 `b_matrix`。
//! det 不在這裡算:前端對 `[T]_B` 與 A 各呼一次既有的 `determinant` binding 當場對帳,
//! 「相似 ⟹ 同 det」是兩條獨立計算的結果,不是 JS 寫死的假設(沿 determinant 章三燈精神)。

use super::helpers::{TRACE_EPSILON, flatten};
use crate::{Matrix, Vector, b_matrix};
use wasm_bindgen::prelude::*;

/// 運算子 A(row-major 2×2 flatten `[a11, a12, a21, a22]`)相對於基底 B = {b₁, b₂} 的矩陣
/// `[T]_B`(core 的 [`b_matrix`]):回 row-major `[t11, t12, t21, t22]`;當 b₁ ∥ b₂
/// (退化、不是 ℝ² 的基底)時回 `[]`(邊界編碼:空 = `[T]_B` 未定義,沿 `coordinates_2d`
/// 慣例)。前端據此切「畫斜格 [T]_B / 紅字非基底」。
///
/// A 以 `Vec<f64>` 傳(沿 `determinant` binding 的矩陣慣例),基底維持 4 純量(沿
/// `coordinates_2d`)。closure `|v| A·v` 在 Rust 端造好再餵 `b_matrix` —— A 的資料過界、
/// closure 不過界。
#[wasm_bindgen]
pub fn b_matrix_2d(a: Vec<f64>, b1x: f64, b1y: f64, b2x: f64, b2y: f64) -> Vec<f64> {
    debug_assert_eq!(a.len(), 4, "前端保證 A 為 2×2 的 flatten");
    let a = Matrix::from_rows(a.chunks(2).map(<[f64]>::to_vec).collect());
    let basis = [
        Vector::from_vec(vec![b1x, b1y]),
        Vector::from_vec(vec![b2x, b2y]),
    ];
    b_matrix(
        TRACE_EPSILON,
        |v| a.multiply_vector(v).expect("2×2 作用於 ℝ²"),
        &basis,
    )
    .map(|m| flatten(&m))
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 標準基底:換座標是 identity,故 [T]_E = A —— 直接讀回 A 的四格。
    /// A = x 軸反射 [[1,0],[0,−1]]。
    #[test]
    fn b_matrix_2d_standard_basis_is_the_operator() {
        let got = b_matrix_2d(vec![1.0, 0.0, 0.0, -1.0], 1.0, 0.0, 0.0, 1.0);
        assert_eq!(got, vec![1.0, 0.0, 0.0, -1.0]);
    }

    /// 同一個反射,換到傾斜基底 B = {(1,1),(1,−1)}:T(1,1)=(1,−1)=b₂、T(1,−1)=(1,1)=b₁,
    /// 故 [T]_B 是交換矩陣 [[0,1],[1,0]] —— 與 A 不相等,卻相似(det 同為 −1)。
    #[test]
    fn b_matrix_2d_tilted_basis_is_the_swap_matrix() {
        let got = b_matrix_2d(vec![1.0, 0.0, 0.0, -1.0], 1.0, 1.0, 1.0, -1.0);
        assert_eq!(got, vec![0.0, 1.0, 1.0, 0.0]);
    }

    /// 相似不變量(binding 層見證):旋轉 90° A=[[0,−1],[1,0]](det 1)換到傾斜基底,
    /// [T]_B 的四格變了,但有號面積 t11·t22 − t12·t21 仍 = det A = 1。
    #[test]
    fn b_matrix_2d_preserves_determinant() {
        let got = b_matrix_2d(vec![0.0, -1.0, 1.0, 0.0], 2.0, 1.0, -1.0, 1.0);
        assert_eq!(got.len(), 4);
        let det = got[0] * got[3] - got[1] * got[2];
        assert!((det - 1.0).abs() < 1e-9, "det[T]_B = {det} 應 = det A = 1");
    }

    /// b₁ ∥ b₂(退化)→ 不是基底 → 空陣列([T]_B 未定義的邊界編碼)。
    #[test]
    fn b_matrix_2d_degenerate_basis_is_empty() {
        assert!(b_matrix_2d(vec![1.0, 0.0, 0.0, 1.0], 1.0, 1.0, 2.0, 2.0).is_empty());
    }
}
