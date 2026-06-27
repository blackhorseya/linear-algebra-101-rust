//! 特徵值與特徵向量(單元 8-1)—— 給前端「塌縮」圖解用:拖 λ 看 A−λI 把單位方塊
//! 壓成的平行四邊形;det(A−λI) → 0(λ 是特徵值)時塌成一條線,而那條線 ——
//! A−λI 的零空間 —— 就是特徵向量。
//!
//! 三顆都是薄 adapter,委派 core 的 8-1 函式:
//! [`characteristic_matrix`](crate::characteristic_matrix)(A−λI)、
//! [`eigenspace_basis`](crate::eigenspace_basis)(Eλ = Null(A−λI),收斂到本章新積木
//! `null_space_basis`)、[`has_real_eigenvalues_2x2`](crate::has_real_eigenvalues_2x2)。
//! **det(A−λI) 不另開 binding** —— 前端對回傳的 M 呼叫既有的 `determinant`(沿 operator 章
//! 重用 determinant 的精神:相似 / 奇異是兩條獨立計算的結果,不是 JS 寫死的假設)。
//!
//! A 一律以 row-major `Vec<f64>`(2×2 flatten)過界(沿 determinant / operator 的矩陣慣例)。

use super::helpers::flatten;
use crate::Matrix;
use wasm_bindgen::prelude::*;

/// 2×2 的 row-major flatten → `Matrix`。前端保證長度 4。
fn matrix_2x2(a: &[f64]) -> Matrix {
    debug_assert_eq!(a.len(), 4, "前端保證 A 為 2×2 的 flatten");
    Matrix::from_rows(a.chunks(2).map(<[f64]>::to_vec).collect())
}

/// 特徵閘門矩陣 **M = A − λI**(core 的 [`characteristic_matrix`]):回 row-major
/// `[m11, m12, m21, m22]`。A 是 2×2(方陣)故恆有定義;理論上的非方陣失敗回 `[]`。
/// 前端用它畫「M 把單位方塊送到的平行四邊形」、並對它呼 `determinant` 看 det(A−λI)。
#[wasm_bindgen]
pub fn characteristic_matrix_2d(a: Vec<f64>, lambda: f64) -> Vec<f64> {
    crate::characteristic_matrix(&matrix_2x2(&a), lambda)
        .map(|m| flatten(&m))
        .unwrap_or_default()
}

/// 特徵空間 **Eλ = Null(A − λI)** 的基底(core 的 [`eigenspace_basis`] → `null_space_basis`):
/// 把每個基底向量攤平 —— 回 `[]`(λ 非特徵值)、`[vx, vy]`(一維特徵空間)或
/// `[v1x, v1y, v2x, v2y]`(二維 = 整個平面,純量矩陣 λI)。
///
/// `epsilon` 由前端傳:它是 RREF「算零」門檻,等同「λ 要多接近特徵值,特徵向量才浮現」
/// 的吸附範圍 —— 把這個 UX 參數交給頁面,調吸附手感不必重編 WASM(沿用核心 RREF 容差,
/// 計算仍全在 core)。
#[wasm_bindgen]
pub fn eigenspace_basis_2d(a: Vec<f64>, lambda: f64, epsilon: f64) -> Vec<f64> {
    crate::eigenspace_basis(epsilon, &matrix_2x2(&a), lambda)
        .map(|basis| {
            basis
                .iter()
                .flat_map(|v| v.entries().iter().copied())
                .collect()
        })
        .unwrap_or_default()
}

/// 此 2×2 是否有實特徵值(core 的 [`has_real_eigenvalues_2x2`]):`false` 時 det(A−λI) 對
/// 任何實 λ 都 > 0 —— 平行四邊形永遠不塌(純旋轉把每個方向都轉開)。
#[wasm_bindgen]
pub fn has_real_eigenvalues_2x2(a: Vec<f64>) -> bool {
    crate::has_real_eigenvalues_2x2(&matrix_2x2(&a))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M = A − 3I:A = diag(3, 1) → [[0,0],[0,−2]]。只動主對角線。
    #[test]
    fn characteristic_matrix_2d_subtracts_lambda_on_diagonal() {
        assert_eq!(
            characteristic_matrix_2d(vec![3.0, 0.0, 0.0, 1.0], 3.0),
            vec![0.0, 0.0, 0.0, -2.0]
        );
    }

    /// diag(2, 3) 在 λ = 2:Eλ = Null(diag(0,1)) = span{e₁} → 回 [1, 0]。
    #[test]
    fn eigenspace_basis_2d_returns_eigenvector_at_eigenvalue() {
        let basis = eigenspace_basis_2d(vec![2.0, 0.0, 0.0, 3.0], 2.0, 1e-9);
        assert_eq!(basis.len(), 2, "一維特徵空間");
        // 平行 (1, 0):cross = 1·0 − 0·basis[1] = −basis[1] ≈ 0
        assert!(basis[1].abs() < 1e-9, "特徵向量應沿 e₁");
        assert!(basis[0].abs() > 1e-9, "且非零");
    }

    /// λ 不是特徵值 → 空(A−λI 滿秩,Eλ = {0})。
    #[test]
    fn eigenspace_basis_2d_is_empty_off_eigenvalue() {
        assert!(eigenspace_basis_2d(vec![2.0, 0.0, 0.0, 3.0], 2.5, 1e-9).is_empty());
    }

    /// 純量矩陣 2I 在 λ = 2:M = 0 → 整個平面都是特徵空間 → 回兩個基底向量(len 4)。
    #[test]
    fn eigenspace_basis_2d_scalar_matrix_is_whole_plane() {
        let basis = eigenspace_basis_2d(vec![2.0, 0.0, 0.0, 2.0], 2.0, 1e-9);
        assert_eq!(basis.len(), 4, "二維特徵空間 = 整個 ℝ²");
    }

    /// 90° 旋轉沒有實特徵值;對稱矩陣有。
    #[test]
    fn has_real_eigenvalues_2x2_distinguishes_rotation_from_symmetric() {
        assert!(!has_real_eigenvalues_2x2(vec![0.0, -1.0, 1.0, 0.0]));
        assert!(has_real_eigenvalues_2x2(vec![2.0, 1.0, 1.0, 2.0]));
    }
}
