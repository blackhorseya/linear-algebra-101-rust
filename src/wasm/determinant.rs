//! 行列式視覺化(/determinant)的 binding —— det = 有號面積。
//!
//! `determinant` 從 `transform` 章搬家至此(一章一檔歸位):當年 core 尚無
//! 行列式、在 transform 手算封閉式 ad − bc;行列式章為 core 補上正名
//! [`Matrix::determinant`](Gaussian 消去版)後改為**委派** —— 計算單一真相
//! 歸位,封閉式只活在測試裡當對照組。`#[wasm_bindgen]` 匯出攤平在套件根層,
//! 搬家對 JS 端 API 零影響。
//!
//! 章門面兩顆:面積與正負號讀 `determinant`(det 路),「塌縮判定」走
//! `is_invertible`(rank 路)—— 兩條獨立計算在頁面上對帳,
//! Theorem 3.4(a)(可逆 ⟺ det ≠ 0)每一幀上演。

use super::helpers::TRACE_EPSILON;
use crate::Matrix;
use wasm_bindgen::prelude::*;

/// 2×2 矩陣 `[[a, b], [c, d]]` 的**行列式**(委派 core 的 Gaussian 版)。
///
/// 幾何意義:單位正方形經此變換後的平行四邊形之**有號面積**。`|det|` 是面積
/// 縮放倍率,正負號代表是否翻面(定向),`det == 0` 表示平面被壓扁成一條線
/// (不可逆 / 線性相依)。
///
/// 2×2 必為方陣,`determinant` 不可能回 `NotSquare`,故 `expect` 把不變式寫明。
#[wasm_bindgen]
pub fn determinant(a: f64, b: f64, c: f64, d: f64) -> f64 {
    Matrix::from_rows(vec![vec![a, b], vec![c, d]])
        .determinant(TRACE_EPSILON)
        .expect("2×2 必為方陣")
}

/// 2×2 矩陣可逆嗎?委派 core 的 [`Matrix::is_invertible`](rank 滿不滿)。
///
/// 視覺化頁拿它判「塌縮」:不可逆 → 平行四邊形壓成線段(面積 0)。刻意
/// **不**在 JS 用 |det| < ε 判 —— rank 路與 det 路是兩條獨立計算,頁面同時
/// 顯示兩者,「可逆 ⟺ det ≠ 0」不是寫死的假設,是每一幀的對帳結果。
#[wasm_bindgen]
pub fn is_invertible(a: f64, b: f64, c: f64, d: f64) -> bool {
    Matrix::from_rows(vec![vec![a, b], vec![c, d]]).is_invertible(TRACE_EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 自 transform 章搬來的案例 —— 委派 core 後同時是「Gaussian = ad − bc
    /// 封閉式」的對照組(整數案例零殘差,精確比較)。
    #[test]
    fn determinant_matches_closed_form() {
        assert_eq!(determinant(0.0, -1.0, 1.0, 0.0), 1.0); // 90° 旋轉:面積不變
        assert_eq!(determinant(2.0, 0.0, 0.0, 3.0), 6.0); // 縮放:面積 ×6
        assert_eq!(determinant(1.0, 0.0, 0.0, -1.0), -1.0); // 鏡射:翻面
        assert_eq!(determinant(1.0, 2.0, 2.0, 4.0), 0.0); // 退化:塌成線
    }

    /// Theorem 3.4(a) 的 binding 版對帳:rank 路(is_invertible)與 det 路
    /// (determinant ≠ 0)必同答案 —— 頁面三燈的理論根據。
    #[test]
    fn invertibility_agrees_with_nonzero_determinant() {
        let cases = [
            (0.0, -1.0, 1.0, 0.0),  // 旋轉:可逆
            (1.0, 1.0, 0.0, 1.0),   // 剪切:可逆(det = 1,面積不變)
            (1.0, 0.0, 0.0, 0.0),   // 投影:塌縮
            (1.0, 2.0, 2.0, 4.0),   // 行向量共線:塌縮
            (-3.0, 0.0, 0.0, 2.0),  // 翻面:可逆(det < 0 也算「非零」)
        ];
        for (a, b, c, d) in cases {
            assert_eq!(
                is_invertible(a, b, c, d),
                determinant(a, b, c, d).abs() > 1e-9,
                "rank 路與 det 路在 [[{a},{b}],[{c},{d}]] 分歧"
            );
        }
    }
}
