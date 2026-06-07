//! 「矩陣作為 2D 線性變換」的基礎 binding —— /transform、/span 頁面用:
//! 點經 2×2 矩陣變換、平行(共線)判定。
//!
//! 歷史註記:`determinant` 曾寄住本檔(當年 core 尚無行列式,在此手算封閉式
//! ad − bc);行列式章成章後已搬回自己的 `determinant` 模組、改為委派 core ——
//! `#[wasm_bindgen]` 匯出攤平在套件根層,搬家對 JS 端 API 零影響。

use crate::{Matrix, Vector};
use wasm_bindgen::prelude::*;

/// 2×2 變換矩陣 A 作用在點 `(x, y)` 上,回傳變換後的 `[x', y']`。
///
/// 這是「矩陣作為 2D 線性變換」的核心:row-major 傳 4 個純量(`a b` / `c d`)——
/// `f64` 過邊界零 marshalling,且 2×2·2D 維度固定,比傳陣列更不易出錯。回傳的
/// `Vec<f64>`(長度 2)在 JS 端是 `Float64Array`。
#[wasm_bindgen]
pub fn transform_point(a: f64, b: f64, c: f64, d: f64, x: f64, y: f64) -> Vec<f64> {
    // 1. Matrix::from_rows 組出 2×2:row0 = [a, b],row1 = [c, d]
    // 2. Vector::from_vec 組出輸入點向量 (x, y)
    // 3. 呼叫 core 的 multiply_vector(&v) 算 A·v —— 計算的單一真相就在這一行
    // 4. 維度恆 2×2·2,multiply_vector 不可能回 DimensionMismatch,故用 .expect
    //    把「不會發生」寫成自證的不變式;再 .entries().to_vec() 轉成 Vec<f64> 回傳
    let matrix = Matrix::from_rows(vec![vec![a, b], vec![c, d]]);
    let point = Vector::from_vec(vec![x, y]);
    matrix
        .multiply_vector(&point)
        .expect("2×2·2D 不會維度不匹配")
        .entries()
        .to_vec()
}

/// 兩個 2D 向量是否**平行**(共線 = 線性相依)。直接委派 core 的
/// [`Vector::is_parallel`]。
///
/// `epsilon` 寫死 `1e-9`(與 crate 內測試同量級):視覺化的拖曳座標數量級穩定,
/// 不需要把容差開放到 JS 端 —— binding 替呼叫端決定一個合理的預設。
#[wasm_bindgen]
pub fn are_parallel(ux: f64, uy: f64, wx: f64, wy: f64) -> bool {
    let u = Vector::from_vec(vec![ux, uy]);
    let w = Vector::from_vec(vec![wx, wy]);
    u.is_parallel(&w, 1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 90° 逆時針旋轉矩陣 `[[0, -1], [1, 0]]` 把 `(1, 0)` 送到 `(0, 1)`;
    /// 單位矩陣不動點。整數值在 f64 下精確,可用 `assert_eq!`。
    #[test]
    fn transform_point_applies_matrix() {
        assert_eq!(
            transform_point(0.0, -1.0, 1.0, 0.0, 1.0, 0.0),
            vec![0.0, 1.0]
        );
        assert_eq!(
            transform_point(1.0, 0.0, 0.0, 1.0, 7.0, 8.0),
            vec![7.0, 8.0]
        );
    }

    #[test]
    fn are_parallel_detects_collinearity() {
        assert!(are_parallel(1.0, 2.0, 2.0, 4.0)); // 純量倍數
        assert!(!are_parallel(1.0, 0.0, 0.0, 1.0)); // 垂直軸
        assert!(are_parallel(0.0, 0.0, 5.0, 7.0)); // 零向量與任意向量平行
    }
}
