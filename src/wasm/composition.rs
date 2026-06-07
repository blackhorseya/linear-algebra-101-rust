//! 合成與可逆性(單元 5-4)—— 給前端「兩路會合」圖解用:拖 x 看「先 T 再 U」
//! 兩步路徑與「一步 BA」直達在同一點會合(T_B ∘ T_A = T_BA);逆轉換模式把
//! U 鎖成 T⁻¹,看「變形 → 復原」回到原地;Summary Table 三燈由 core 的
//! report 一次點亮。
//!
//! 與 core 的關係:全部直接委派 `composition` 模組(compose / inverse /
//! is_one_to_one / report)—— binding 只做 2×2 形狀的攤平與 Result 的邊界編碼
//! (空陣列 = 無逆轉換),零演算法。epsilon 一律寫死 TRACE_EPSILON。

use super::helpers::{TRACE_EPSILON, flatten, transformation_2x2};
use wasm_bindgen::prelude::*;

/// U ∘ T 的標準矩陣 = B·A(core 的 `compose`),row-major 4 元素攤平。
/// 參數序與 `u.compose(&t)` 的讀序同向:先外層 U、再內層 T。
/// 2×2 ∘ 2×2 中間空間必接得上,expect 安全。
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)] // 兩個 2×2 攤平本來就是 8 個純量
pub fn compose_matrix(
    ua: f64,
    ub: f64,
    uc: f64,
    ud: f64,
    ta: f64,
    tb: f64,
    tc: f64,
    td: f64,
) -> Vec<f64> {
    let u = transformation_2x2(ua, ub, uc, ud);
    let t = transformation_2x2(ta, tb, tc, td);
    flatten(u.compose(&t).expect("2×2 ∘ 2×2 維度必合").matrix())
}

/// T⁻¹ 的標準矩陣 = A⁻¹(core 的 `inverse`,Theorem 2.13):
/// 可逆 → 4 元素攤平;不可逆 → `[]`(Result 的邊界編碼:空陣列 = 無逆轉換,
/// 沿 `unreachable_vector` 的 Option 慣例 —— 2×2 必方陣,失敗只剩 NotInvertible)。
#[wasm_bindgen]
pub fn inverse_matrix(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    transformation_2x2(a, b, c, d)
        .inverse(TRACE_EPSILON)
        .map(|t_inv| flatten(t_inv.matrix()))
        .unwrap_or_default()
}

/// T 一對一嗎?直接委派 core 的 `is_one_to_one`(Theorem 2.11:rank = n)——
/// 與 5-3 的 [`is_onto`](super::range::is_onto) 成對,前端對偶面板各問一次。
#[wasm_bindgen]
pub fn is_one_to_one(a: f64, b: f64, c: f64, d: f64) -> bool {
    transformation_2x2(a, b, c, d).is_one_to_one(TRACE_EPSILON)
}

/// Summary Table 三燈:`[1-1, onto, invertible]` 的 0 / 1 —— core 的 `report`
/// 一次算好,前端純讀三盞燈,連 Theorem 2.12 的合取都不留給 JS(計算單一真相)。
#[wasm_bindgen]
pub fn transformation_report(a: f64, b: f64, c: f64, d: f64) -> Vec<u8> {
    let r = transformation_2x2(a, b, c, d).report(TRACE_EPSILON);
    vec![
        u8::from(r.is_one_to_one),
        u8::from(r.is_onto),
        u8::from(r.is_invertible),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::range::is_onto;
    use crate::wasm::transform::transform_point;

    /// 三種秩的代表(沿 range binding 的測試常數):可逆、行成比例、零矩陣。
    const FULL_RANK: [f64; 4] = [2.0, 1.0, 1.0, 1.0];
    const RANK_ONE: [f64; 4] = [1.0, 2.0, 2.0, 4.0];
    const RANK_ZERO: [f64; 4] = [0.0, 0.0, 0.0, 0.0];

    /// 兩路會合(T_B ∘ T_A = T_BA 的 binding 重述):x 走「先 T 再 U」兩步,
    /// 與走 compose_matrix 的一步 BA,落在同一點 —— 前端動畫的數學保證。
    #[test]
    fn compose_matrix_agrees_with_two_step_path() {
        let (u, t) = ([0.0, -1.0, 1.0, 0.0], [1.0, 0.0, 0.0, -1.0]); // 旋轉 ∘ 反射
        let ba = compose_matrix(u[0], u[1], u[2], u[3], t[0], t[1], t[2], t[3]);
        for (x, y) in [(3.0, 1.0), (-2.0, 5.0), (0.0, 0.0)] {
            let mid = transform_point(t[0], t[1], t[2], t[3], x, y);
            let two_steps = transform_point(u[0], u[1], u[2], u[3], mid[0], mid[1]);
            let one_step = transform_point(ba[0], ba[1], ba[2], ba[3], x, y);
            assert_eq!(one_step, two_steps, "兩路不會合:x=({x}, {y})");
        }
    }

    /// 逆轉換的邊界編碼:可逆 → 4 元素且「變形 → 復原」回到原地;
    /// 奇異 / 零矩陣 → 空陣列(前端據此切換「無逆轉換」提示)。
    #[test]
    fn inverse_matrix_round_trips_or_is_empty() {
        let [a, b, c, d] = FULL_RANK;
        let inv = inverse_matrix(a, b, c, d);
        assert_eq!(inv.len(), 4, "可逆:回 A⁻¹ 攤平");
        let (x, y) = (3.0, -2.0);
        let mid = transform_point(a, b, c, d, x, y);
        let back = transform_point(inv[0], inv[1], inv[2], inv[3], mid[0], mid[1]);
        assert!(
            (back[0] - x).abs() < 1e-9 && (back[1] - y).abs() < 1e-9,
            "T⁻¹(T(x)) ≠ x"
        );

        for m in [RANK_ONE, RANK_ZERO] {
            let [a, b, c, d] = m;
            assert!(inverse_matrix(a, b, c, d).is_empty(), "不可逆:空陣列");
        }
    }

    /// 三燈與獨立述詞逐欄對帳,且 2×2 是方陣 —— IMT 三位一體:三燈必同步
    /// (全亮或全滅),可逆 ⟺ inverse_matrix 非空(Theorem 2.13 的邊界版)。
    #[test]
    fn report_lights_agree_with_predicates() {
        for m in [FULL_RANK, RANK_ONE, RANK_ZERO] {
            let [a, b, c, d] = m;
            let lights = transformation_report(a, b, c, d);
            assert_eq!(lights[0] == 1, is_one_to_one(a, b, c, d), "燈 0 = 1-1");
            assert_eq!(lights[1] == 1, is_onto(a, b, c, d), "燈 1 = onto");
            assert_eq!(
                lights[2] == 1,
                !inverse_matrix(a, b, c, d).is_empty(),
                "燈 2 ⟺ 逆轉換存在(Theorem 2.13)"
            );
            assert!(
                lights.iter().all(|&l| l == lights[0]),
                "方陣三位一體:三燈同步"
            );
        }
    }

    /// 三種秩的 Summary Table:滿秩全亮、塌縮全滅 —— 前端 preset 的預期長相。
    #[test]
    fn report_classifies_three_ranks() {
        let [a, b, c, d] = FULL_RANK;
        assert_eq!(transformation_report(a, b, c, d), vec![1, 1, 1]);
        let [a, b, c, d] = RANK_ONE;
        assert_eq!(transformation_report(a, b, c, d), vec![0, 0, 0]);
        let [a, b, c, d] = RANK_ZERO;
        assert_eq!(transformation_report(a, b, c, d), vec![0, 0, 0]);
    }
}
