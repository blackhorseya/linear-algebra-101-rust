//! 行列式視覺化(/determinant)的 binding —— det = 有號 n 維體積。
//!
//! `determinant` 從 `transform` 章搬家至此(一章一檔歸位):當年 core 尚無
//! 行列式、在 transform 手算 2×2 封閉式 ad − bc;行列式章為 core 補上正名
//! [`Matrix::determinant`](Gaussian 消去版)後改為**委派** —— 計算單一真相
//! 歸位。頁面升級 n×n 後,binding 跟著改收 row-major flatten + `n`(沿
//! [`invert_trace`](super::inverse::invert_trace) 的邊界慣例),封閉式只活在
//! 測試裡當 2×2 對照組,n×n 的對照組則是 core 的餘因子展開
//! [`Matrix::determinant_recursive`]。
//!
//! 章門面兩顆:體積與正負號讀 `determinant`(det 路),「塌縮判定」走
//! `is_invertible`(rank 路)—— 兩條獨立計算在頁面上對帳,
//! Theorem 3.4(a)(可逆 ⟺ det ≠ 0)每一幀上演。

use super::helpers::TRACE_EPSILON;
use crate::Matrix;
use wasm_bindgen::prelude::*;

/// n×n 矩陣(row-major flatten)的**行列式**(委派 core 的 Gaussian 版)。
///
/// 幾何意義:單位 n 維方體經此變換後的平行多面體之**有號體積**(n = 2 是
/// 有號面積)。`|det|` 是體積縮放倍率,正負號代表是否翻轉定向,`det == 0`
/// 表示空間被壓扁到更低維(不可逆 / 線性相依)。
///
/// `data` 長度須為 `n * n` 且 `n ≥ 1`(前端保證,沿 `multiply_expand` 的
/// debug_assert 慣例);依 `n` 切列必為方陣,`determinant` 不可能回
/// `NotSquare`,故 `expect` 把不變式寫明。
#[wasm_bindgen]
pub fn determinant(data: Vec<f64>, n: usize) -> f64 {
    debug_assert_eq!(data.len(), n * n, "前端保證 data 為 n×n 的 flatten");
    Matrix::from_rows(data.chunks(n).map(<[f64]>::to_vec).collect())
        .determinant(TRACE_EPSILON)
        .expect("依 n 切列必為方陣")
}

/// n×n 矩陣(row-major flatten)可逆嗎?委派 core 的
/// [`Matrix::is_invertible`](rank 滿不滿)。
///
/// 視覺化頁拿它判「塌縮」:不可逆 → 像被壓到更低維(體積 0)。刻意**不**在
/// JS 用 |det| < ε 判 —— rank 路與 det 路是兩條獨立計算,頁面同時顯示兩者,
/// 「可逆 ⟺ det ≠ 0」不是寫死的假設,是每一幀的對帳結果。
#[wasm_bindgen]
pub fn is_invertible(data: Vec<f64>, n: usize) -> bool {
    debug_assert_eq!(data.len(), n * n, "前端保證 data 為 n×n 的 flatten");
    Matrix::from_rows(data.chunks(n).map(<[f64]>::to_vec).collect()).is_invertible(TRACE_EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::helpers::input_matrix;

    /// 自 transform 章搬來的 2×2 案例 —— 委派 core 後同時是「Gaussian = ad − bc
    /// 封閉式」的對照組(整數案例零殘差,精確比較)。
    #[test]
    fn determinant_2x2_matches_closed_form() {
        assert_eq!(determinant(vec![0.0, -1.0, 1.0, 0.0], 2), 1.0); // 90° 旋轉:面積不變
        assert_eq!(determinant(vec![2.0, 0.0, 0.0, 3.0], 2), 6.0); // 縮放:面積 ×6
        assert_eq!(determinant(vec![1.0, 0.0, 0.0, -1.0], 2), -1.0); // 鏡射:翻面
        assert_eq!(determinant(vec![1.0, 2.0, 2.0, 4.0], 2), 0.0); // 退化:塌成線
    }

    /// n×n 的守備範圍:已知值案例覆蓋「翻號 / 精確零 / 對角線乘積」三條
    /// Gaussian 路徑。partial pivoting 帶除法的案例用容差;消去全程無除法
    /// 殘差的案例(三角、置換、奇異 early-return)維持精確比較。
    #[test]
    fn determinant_handles_n_by_n() {
        // 3×3 已知值:餘因子手算 = −3(pivoting 引入除法 → 容差比較)
        let m3 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        assert!((determinant(m3, 3) - (-3.0)).abs() < 1e-9);

        // 上三角:det = 對角線乘積 2·3·4 = 24(Theorem 3.2 的 binding 面)
        let upper = vec![2.0, 1.0, 3.0, 0.0, 3.0, 5.0, 0.0, 0.0, 4.0];
        assert_eq!(determinant(upper, 3), 24.0);

        // 4×4 單次列交換的置換矩陣:det = −1(定向翻轉,sign 翻號路徑)
        #[rustfmt::skip]
        let perm = vec![
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        assert_eq!(determinant(perm, 4), -1.0);

        // 奇異(兩列相同):core 找不到 pivot → early return **精確** 0.0
        let singular = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0];
        assert_eq!(determinant(singular, 3), 0.0);
    }

    /// n×n 的對照組:binding 委派的 Gaussian 路 vs core 的餘因子路
    /// (`determinant_recursive`)—— 2×2 用封閉式對帳的精神,推廣到 n×n。
    #[test]
    fn delegation_agrees_with_recursive_cofactor() {
        let cases: [(Vec<f64>, usize); 3] = [
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], 3),
            (vec![2.0, 1.0, 3.0, 0.0, 3.0, 5.0, 0.0, 0.0, 4.0], 3),
            (vec![3.0, -2.0, 1.0, 4.0], 2),
        ];
        for (data, n) in cases {
            let recursive = input_matrix(&data, n).determinant_recursive().unwrap();
            assert!(
                (determinant(data.clone(), n) - recursive).abs() < 1e-9,
                "Gaussian 路與餘因子路在 {data:?} (n={n}) 分歧"
            );
        }
    }

    /// Theorem 3.4(a) 的 binding 版對帳:rank 路(is_invertible)與 det 路
    /// (determinant ≠ 0)必同答案 —— 頁面三燈的理論根據,n×n 後守備不變。
    #[test]
    fn invertibility_agrees_with_nonzero_determinant() {
        let cases: [(Vec<f64>, usize); 7] = [
            (vec![0.0, -1.0, 1.0, 0.0], 2),                          // 旋轉:可逆
            (vec![1.0, 1.0, 0.0, 1.0], 2),                           // 剪切:可逆(det = 1,面積不變)
            (vec![1.0, 0.0, 0.0, 0.0], 2),                           // 投影:塌縮
            (vec![1.0, 2.0, 2.0, 4.0], 2),                           // 行向量共線:塌縮
            (vec![-3.0, 0.0, 0.0, 2.0], 2), // 翻面:可逆(det < 0 也算「非零」)
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], 3), // 3×3:可逆
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0], 3), // 3×3 兩列相同:塌縮
        ];
        for (data, n) in cases {
            assert_eq!(
                is_invertible(data.clone(), n),
                determinant(data.clone(), n).abs() > 1e-9,
                "rank 路與 det 路在 {data:?} (n={n}) 分歧"
            );
        }
    }
}
