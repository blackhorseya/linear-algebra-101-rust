//! 標準矩陣取樣(單元 5-2)—— 給前端「選幾何規則,看 e₁、e₂ 的影像被取樣、
//! 直放成矩陣的行」圖解用。
//!
//! 與 core 的關係:幾何規則(旋轉/反射/剪切/投影/縮放)以**座標規則**的形式
//! 住在這裡 —— 只有規則,沒有矩陣。矩陣由 core 的 [`standard_matrix`] 對規則
//! 取樣「發現」(Theorem 2.9 的工作流原樣上演,呼應 `transformation.rs` 的
//! x 軸反射測試:寫規則,讓構造器去發現矩陣)。`apply_rule` 是同一條規則的
//! 直接施作,給前端畫「規則路徑 vs 矩陣路徑」的兩路會合(對帳測試釘住)。

use super::helpers::flatten;
use crate::{Vector, standard_matrix};
use wasm_bindgen::prelude::*;

// 幾何規則編碼。用 `u8` 過邊界(沿 `PHASE_*` 慣例),前端的對照表順序必須一致。
const RULE_ROTATE: u8 = 0; // 旋轉 θ(param = 弧度)
const RULE_REFLECT_X: u8 = 1; // x 軸反射
const RULE_REFLECT_DIAG: u8 = 2; // 對 y = x 反射
const RULE_SHEAR_X: u8 = 3; // 水平剪切(param = k)
const RULE_PROJECT_X: u8 = 4; // 投影到 x 軸
const RULE_SCALE: u8 = 5; // 等比縮放(param = k)

/// 幾何規則本體:把 2D 向量依規則送到影像 —— **這裡只有規則,沒有矩陣字面值**。
/// 旋轉是 (x cosθ − y sinθ, x sinθ + y cosθ)、反射是「x 不動、y 翻號」⋯⋯
/// 全是課本上「幾何直觀」那一側的描述;矩陣那一側交給 `standard_matrix` 取樣。
fn rule_image(rule: u8, param: f64, v: &Vector) -> Vector {
    let (x, y) = (v.entries()[0], v.entries()[1]);
    let (ix, iy) = match rule {
        RULE_ROTATE => (
            x * param.cos() - y * param.sin(),
            x * param.sin() + y * param.cos(),
        ),
        RULE_REFLECT_X => (x, -y),
        RULE_REFLECT_DIAG => (y, x),
        RULE_SHEAR_X => (x + param * y, y),
        RULE_PROJECT_X => (x, 0.0),
        RULE_SCALE => (param * x, param * y),
        _ => unreachable!("前端只送上方定義的 rule 編碼"),
    };
    Vector::from_vec(vec![ix, iy])
}

/// 幾何規則的**標準矩陣**:core 的 [`standard_matrix`] 對規則做 e₁、e₂ 取樣,
/// 回傳 row-major `[a, b, c, d]`(Theorem 2.9:A 的第 j 行 = T(eⱼ))。
///
/// 頁面上的矩陣數字就是這裡「發現」出來的 —— 不是前端寫死的字面值,
/// 也不是 binding 抄好的公式;單元 5-2 練習 1 的構造器親自上場。
#[wasm_bindgen]
pub fn sample_standard_matrix(rule: u8, param: f64) -> Vec<f64> {
    let a = standard_matrix(2, |v| rule_image(rule, param, v))
        .expect("n = 2 ≥ 1 且規則恆回 2D —— 兩個 Err 都是死路");
    flatten(&a)
}

/// 幾何規則**直接施作**在點 `(x, y)`,回傳 `[x', y']` —— 「規則路徑」。
/// 前端用它畫 T(e₁)、T(e₂) 與 T(v);「矩陣路徑」則是 `transform_point` 左乘
/// 取樣矩陣 —— 兩條路會合即 Theorem 2.9(下方對帳測試釘住)。
#[wasm_bindgen]
pub fn apply_rule(rule: u8, param: f64, x: f64, y: f64) -> Vec<f64> {
    rule_image(rule, param, &Vector::from_vec(vec![x, y]))
        .entries()
        .to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify_linearity;
    use crate::wasm::transform::transform_point;

    /// 全部六條幾何規則,搭配各自有意義的 param(無 param 的規則給 0)。
    fn all_rules() -> Vec<(u8, f64)> {
        vec![
            (RULE_ROTATE, 0.7),
            (RULE_REFLECT_X, 0.0),
            (RULE_REFLECT_DIAG, 0.0),
            (RULE_SHEAR_X, -1.5),
            (RULE_PROJECT_X, 0.0),
            (RULE_SCALE, -0.5),
        ]
    }

    /// 取樣出的標準矩陣對帳幾何規則的課本矩陣 —— Theorem 2.9 的黃金案例。
    /// 旋轉 90° 的 cos(π/2) 帶 ~6e-17 浮點殘差,容差比對;其餘規則整數精確。
    #[test]
    fn sample_standard_matrix_discovers_textbook_matrices() {
        let rot = sample_standard_matrix(RULE_ROTATE, std::f64::consts::FRAC_PI_2);
        for (got, want) in rot.iter().zip([0.0, -1.0, 1.0, 0.0]) {
            assert!(
                (got - want).abs() < 1e-12,
                "90° 旋轉應取樣出 [[0,−1],[1,0]],got={rot:?}"
            );
        }
        assert_eq!(
            sample_standard_matrix(RULE_REFLECT_X, 0.0),
            vec![1.0, 0.0, 0.0, -1.0]
        );
        assert_eq!(
            sample_standard_matrix(RULE_REFLECT_DIAG, 0.0),
            vec![0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            sample_standard_matrix(RULE_SHEAR_X, 2.0),
            vec![1.0, 2.0, 0.0, 1.0]
        );
        assert_eq!(
            sample_standard_matrix(RULE_PROJECT_X, 0.0),
            vec![1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            sample_standard_matrix(RULE_SCALE, 1.5),
            vec![1.5, 0.0, 0.0, 1.5]
        );
    }

    /// **Theorem 2.9 的 binding 對帳**:每條規則、同一測試點,「規則直接算」
    /// (apply_rule)與「左乘取樣矩陣」(transform_point)兩條路必須會合 ——
    /// 頁面上綠箭頭與白圓環重合的數學保證。
    #[test]
    fn apply_rule_agrees_with_sampled_matrix() {
        for (rule, param) in all_rules() {
            let m = sample_standard_matrix(rule, param);
            let (x, y) = (2.5, -1.25);
            let via_rule = apply_rule(rule, param, x, y);
            let via_matrix = transform_point(m[0], m[1], m[2], m[3], x, y);
            for (r, a) in via_rule.iter().zip(via_matrix.iter()) {
                assert!((r - a).abs() < 1e-12, "rule={rule}:兩條路應會合");
            }
        }
    }

    /// 每條幾何規則都通過 core 的線性檢查 —— 它們才有資格談「標準矩陣」
    /// (Theorem 2.9 的「若 T 線性」前提;非線性規則取樣出的矩陣重現不了規則)。
    #[test]
    fn geometry_rules_are_linear() {
        let u = Vector::from_vec(vec![1.0, -2.0]);
        let w = Vector::from_vec(vec![3.0, 0.5]);
        for (rule, param) in all_rules() {
            assert!(
                verify_linearity(|v| rule_image(rule, param, v), &u, &w, -2.5, 1e-9),
                "rule={rule} 應為線性"
            );
        }
    }
}
