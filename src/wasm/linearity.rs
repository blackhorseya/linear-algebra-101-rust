//! 線性轉換守恆律(單元 5-1)—— 給前端「拖動向量看 shear / 投影的影像,並親眼
//! 驗證 T(u+v) = T(u)+T(v)、T(ku) = k·T(u)」圖解用。
//!
//! 與 core 的關係:T(x) 沿用既有的 `transform_point`(計算同源 multiply_vector);
//! 這裡補上前端需要、但 JS 不准重寫的兩個向量運算(add / scale),以及把 core
//! `transformation` 模組的 verify_linearity 原樣接出來 —— 前端顯示的 ✓/✗ 是
//! core 親自驗的,不是 JS 寫死的字。

use crate::{Matrix, Transformation, Vector, verify_linearity};
use wasm_bindgen::prelude::*;

/// 2D 向量加法 `u + v`,回傳 `[x, y]`。委派 core 的 [`Vector::add`];
/// 維度恆 2,`expect` 把「不會發生」寫成自證的不變式(沿 `transform_point` 慣例)。
#[wasm_bindgen]
pub fn add_vectors(ux: f64, uy: f64, vx: f64, vy: f64) -> Vec<f64> {
    let u = Vector::from_vec(vec![ux, uy]);
    let v = Vector::from_vec(vec![vx, vy]);
    u.add(&v)
        .expect("同為 2D 不會維度不匹配")
        .entries()
        .to_vec()
}

/// 2D 向量純量乘 `k·u`,回傳 `[x, y]`。委派 core 的 [`Vector::scale`](不會失敗)。
#[wasm_bindgen]
pub fn scale_vector(x: f64, y: f64, k: f64) -> Vec<f64> {
    Vector::from_vec(vec![x, y]).scale(k).entries().to_vec()
}

/// 2×2 矩陣 `[[a, b], [c, d]]` 誘導的轉換 T_A 在樣本 `(u, v, k)` 上的**線性檢查**:
/// T(u+v) = T(u)+T(v) 且 T(ku) = k·T(u)。
///
/// 直接委派 core 的 [`verify_linearity`] + [`Transformation::apply`] ——
/// Theorem 2.7 說矩陣誘導的轉換必過此檢查,所以前端看到的永遠是 ✓;這顆 binding
/// 的價值正是「✓ 由 core 當場驗出來」,不是前端寫死的裝飾。
/// epsilon 寫死 `1e-9`(沿 `are_parallel` 慣例:拖曳座標數量級穩定)。
// 9 個參數沿 `transform_point` 的「f64 過邊界零 marshalling」慣例:2×2·2D·純量
// 形狀固定,攤平比包陣列更不易錯 —— 故 allow 而不改簽名。
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn check_linearity(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    ux: f64,
    uy: f64,
    vx: f64,
    vy: f64,
    k: f64,
) -> bool {
    let t = Transformation::new(Matrix::from_rows(vec![vec![a, b], vec![c, d]]));
    let u = Vector::from_vec(vec![ux, uy]);
    let v = Vector::from_vec(vec![vx, vy]);
    verify_linearity(
        |x| t.apply(x).expect("2×2·2D 不會維度不匹配"),
        &u,
        &v,
        k,
        1e-9,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 向量加法 / 純量乘 binding:整數值在 f64 下精確,可用 `assert_eq!`。
    #[test]
    fn add_and_scale_vectors_delegate_to_core() {
        assert_eq!(add_vectors(1.0, 2.0, 3.0, -1.0), vec![4.0, 1.0]);
        assert_eq!(scale_vector(1.5, -2.0, 2.0), vec![3.0, -4.0]);
        assert_eq!(scale_vector(1.0, 2.0, 0.0), vec![0.0, 0.0]); // k=0 → 零向量
    }

    /// **Theorem 2.7 的 binding 對帳**:單元 5-1 頁面的三個 preset(shear、投影、
    /// 零轉換)都必須通過 core 的線性檢查 —— ✓ 是 core 驗出來的,不是前端寫死。
    /// 投影 det = 0(不可逆)仍線性,正是「線性 ≠ 可逆」的教學點。
    #[test]
    fn check_linearity_passes_matrix_transformations() {
        // shear [[1,1],[0,1]]
        assert!(check_linearity(
            1.0, 1.0, 0.0, 1.0, 1.0, -2.0, 3.0, 0.5, -1.5
        ));
        // 投影到 x 軸 [[1,0],[0,0]](det = 0,不可逆但線性)
        assert!(check_linearity(
            1.0, 0.0, 0.0, 0.0, 1.0, 2.0, -3.0, 4.0, 2.0
        ));
        // 零轉換 [[0,0],[0,0]]
        assert!(check_linearity(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0));
    }
}
