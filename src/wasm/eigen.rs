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

use super::helpers::{flatten, flatten_vectors, matrix_n};
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

/// [`characteristic_matrix_2d`] 的 n×n 通用版(計算機軌道):回 row-major flatten 的
/// M = A − λI。方陣恆有定義;理論上的失敗回 `[]`(沿 2d 版邊界編碼)。
#[wasm_bindgen]
pub fn characteristic_matrix_n(a: Vec<f64>, n: usize, lambda: f64) -> Vec<f64> {
    crate::characteristic_matrix(&matrix_n(&a, n), lambda)
        .map(|m| flatten(&m))
        .unwrap_or_default()
}

/// [`eigenspace_basis_2d`] 的 n×n 通用版(計算機軌道):Eλ 基底攤平串接,
/// 支數 = 總長 ÷ n = 幾何重數(0 = λ 非特徵值)。`epsilon` 語意同 2d 版。
#[wasm_bindgen]
pub fn eigenspace_basis_n(a: Vec<f64>, n: usize, lambda: f64, epsilon: f64) -> Vec<f64> {
    crate::eigenspace_basis(epsilon, &matrix_n(&a, n), lambda)
        .map(|basis| flatten_vectors(&basis))
        .unwrap_or_default()
}

/// 跡 tr(A) = Σ aᵢᵢ(core 的 [`Matrix::trace`],8-2 新積木;計算機軌道)。
/// `matrix_n` 產物恆為方陣,`trace` 不會失敗。
#[wasm_bindgen]
pub fn matrix_trace(a: Vec<f64>, n: usize) -> f64 {
    matrix_n(&a, n)
        .trace()
        .expect("matrix_n 恆造方陣,trace 不會失敗")
}

/// 三角矩陣特徵值直讀(core 的 [`triangular_eigenvalues`],8-2:上/下三角的特徵值
/// = 主對角線,保留重複 = 代數重數)。非三角回 `[]`(邊界編碼無歧義:三角必回 n 個)。
#[wasm_bindgen]
pub fn triangular_eigenvalues_n(a: Vec<f64>, n: usize, epsilon: f64) -> Vec<f64> {
    crate::triangular_eigenvalues(epsilon, &matrix_n(&a, n)).unwrap_or_default()
}

/// 特徵值重數報告(core 的 [`analyze_eigen_properties`],8-3):每個相異 λ 一列,
/// 攤平成 `[λ, 代數重數, 幾何重數, 虧損(0/1)]` × 列數(SoA 對小資料是殺雞用牛刀,
/// 定寬 4-tuple 列即可)。`eigenvalues` 為含重複的根列表(如 `triangular_eigenvalues_n`
/// 的輸出);Theorem 5.1(幾何 ≤ 代數)由 core 的 laws 證,前端只是把數字並排。
#[wasm_bindgen]
pub fn eigen_reports_n(a: Vec<f64>, n: usize, eigenvalues: Vec<f64>, epsilon: f64) -> Vec<f64> {
    crate::analyze_eigen_properties(epsilon, &matrix_n(&a, n), &eigenvalues)
        .map(|reports| {
            reports
                .iter()
                .flat_map(|r| {
                    [
                        r.value,
                        r.algebraic as f64,
                        r.geometric as f64,
                        f64::from(r.is_deficient),
                    ]
                })
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

    /// n×n 通用版:diag(2,3,2) 在 λ = 2 —— M = A−2I 只動主對角線,
    /// Eλ 幾何重數 2(基底 2 支、各長 3)。
    #[test]
    fn n_variants_handle_3x3_diagonal() {
        let a = vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0];
        assert_eq!(
            characteristic_matrix_n(a.clone(), 3, 2.0),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        let basis = eigenspace_basis_n(a, 3, 2.0, 1e-9);
        assert_eq!(basis.len() / 3, 2, "diag(2,3,2) 在 λ=2 幾何重數 2");
    }

    /// 8-2 積木:trace = 對角線和;三角直讀保留重複、非三角回空(邊界編碼)。
    #[test]
    fn trace_and_triangular_eigenvalues_read_the_diagonal() {
        let upper = vec![3.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, -1.0];
        assert_eq!(matrix_trace(upper.clone(), 3), 5.0);
        assert_eq!(
            triangular_eigenvalues_n(upper, 3, 1e-9),
            vec![3.0, 3.0, -1.0],
            "對角線直讀,重根保留"
        );
        // 非三角(兩個角都非零)→ 直讀失效 → 空。
        let full = vec![1.0, 2.0, 3.0, 4.0];
        assert!(triangular_eigenvalues_n(full, 2, 1e-9).is_empty());
    }

    /// 8-3 報告:剪切 [[3,1],[0,3]] 的 λ=3 —— 代數 2、幾何 1、虧損
    /// (Theorem 5.1 幾何 ≤ 代數的嚴格不等示例);相異 λ 恰一列。
    #[test]
    fn eigen_reports_n_flags_deficient_shear() {
        let shear = vec![3.0, 1.0, 0.0, 3.0];
        let evs = triangular_eigenvalues_n(shear.clone(), 2, 1e-9);
        let flat = eigen_reports_n(shear, 2, evs, 1e-9);
        assert_eq!(flat, vec![3.0, 2.0, 1.0, 1.0], "[λ, 代數, 幾何, 虧損]");
    }
}
