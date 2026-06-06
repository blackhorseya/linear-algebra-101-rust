//! 值域與映成(單元 5-3)—— 給前端「值域覆蓋」圖解用:拖動 A 的行向量看
//! Range(T) = Col(A) 從整個平面塌成直線、再塌到原點;拖 w 看可達性即時判定,
//! 不映成時把 unreachable_vector 的見證標在圖上。
//!
//! 與 core 的關係:全部直接委派 `range` 模組(range_basis / range_contains /
//! is_onto / unreachable_vector)與 `system` 模組(solve)—— binding 只做
//! 2×2 形狀的攤平與 Option / enum 的邊界編碼,零演算法(本章 core 模組的
//! 「只有積木接線」精神原樣延伸到邊界層)。
//! epsilon 一律寫死 TRACE_EPSILON(沿 eliminate 慣例:拖曳座標數量級穩定)。

use super::helpers::TRACE_EPSILON;
use crate::{Matrix, Solution, System, Transformation, Vector};
use wasm_bindgen::prelude::*;

/// 把 row-major 的 2×2 純量升格為轉換 T_A: ℝ² → ℝ²(本章五顆 binding 共用)。
fn transformation_2x2(a: f64, b: f64, c: f64, d: f64) -> Transformation {
    Transformation::new(Matrix::from_rows(vec![vec![a, b], vec![c, d]]))
}

/// Range(T) 的基底(core 的 `range_basis`),行向量攤平串接回傳:
/// `[]`(rank 0:值域 = {0})、`[x, y]`(rank 1:值域塌成直線)、
/// `[x₁, y₁, x₂, y₂]`(rank 2:值域 = ℝ²)—— 支數 = rank,長度就把維度說完了。
#[wasm_bindgen]
pub fn range_basis(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    transformation_2x2(a, b, c, d)
        .range_basis(TRACE_EPSILON)
        .iter()
        .flat_map(|v| v.entries().iter().copied())
        .collect()
}

/// `w ∈ Range(T)`?直接委派 core 的 `range_contains`(w 可達 ⟺ Ax = w 相容)——
/// 前端 w 箭頭的綠 / 紅是 core 當場判的,不是 JS 寫死的條件。
#[wasm_bindgen]
pub fn range_contains(a: f64, b: f64, c: f64, d: f64, wx: f64, wy: f64) -> bool {
    transformation_2x2(a, b, c, d).range_contains(&Vector::from_vec(vec![wx, wy]), TRACE_EPSILON)
}

/// T 映成嗎?直接委派 core 的 `is_onto`(Theorem 2.10:rank = m)。
#[wasm_bindgen]
pub fn is_onto(a: f64, b: f64, c: f64, d: f64) -> bool {
    transformation_2x2(a, b, c, d).is_onto(TRACE_EPSILON)
}

/// 不可達向量的見證(core 的 `unreachable_vector`,標準基底掃描):
/// 不映成 → `[x, y]`(某支 eᵢ);映成 → `[]`(`Option` 的邊界編碼:空陣列 = None)。
#[wasm_bindgen]
pub fn unreachable_vector(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    transformation_2x2(a, b, c, d)
        .unreachable_vector(TRACE_EPSILON)
        .map(|v| v.entries().to_vec())
        .unwrap_or_default()
}

/// 解 `Ax = w`「哪個輸入到得了 w」,回傳 `[kind, x, y]` ——
/// kind 沿 [`EliminationTrace`](super::elimination::EliminationTrace) 的
/// solution_kind 編碼(1 = Unique、2 = Infinite、3 = Inconsistent;前端對照表
/// 共用),x、y 只在 Unique 時有意義(其餘補 0)。
///
/// 與 `range_contains` 是同一個問題的兩種問法(可達 ⟺ kind ≠ Inconsistent,
/// 對帳測試釘住),但 solve 多給了「存在性的見證」:那個輸入 x 本身。
#[wasm_bindgen]
pub fn solve_for_input(a: f64, b: f64, c: f64, d: f64, wx: f64, wy: f64) -> Vec<f64> {
    let system = System::new(
        Matrix::from_rows(vec![vec![a, b], vec![c, d]]),
        Vector::from_vec(vec![wx, wy]),
    )
    .expect("2×2 配 2D 常數向量,維度必合");
    match system.solve(TRACE_EPSILON) {
        Solution::Unique(x) => vec![1.0, x.entries()[0], x.entries()[1]],
        Solution::Infinite => vec![2.0, 0.0, 0.0],
        Solution::Inconsistent => vec![3.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::transform::transform_point;

    /// 三種秩的代表矩陣:rank 2(可逆)、rank 1(行成比例)、rank 0(零矩陣),
    /// 以 row-major `[a, b, c, d]` 表示 —— 值域分別是 ℝ²、直線、{0}。
    const FULL_RANK: [f64; 4] = [2.0, 1.0, 1.0, 1.0];
    const RANK_ONE: [f64; 4] = [1.0, 2.0, 2.0, 4.0]; // 行 (1,2) 與 (2,4) 共線
    const RANK_ZERO: [f64; 4] = [0.0, 0.0, 0.0, 0.0];

    /// 基底的攤平長度把 Range 的維度說完:4 個數(平面)、2 個數(直線)、
    /// 空(原點)。rank 1 的基底是 pivot 行對應的**原始行** (1, 2)。
    #[test]
    fn range_basis_length_encodes_collapse() {
        let [a, b, c, d] = FULL_RANK;
        assert_eq!(range_basis(a, b, c, d).len(), 4, "rank 2:兩支基底");
        let [a, b, c, d] = RANK_ONE;
        assert_eq!(range_basis(a, b, c, d), vec![1.0, 2.0], "rank 1:原始行 0");
        let [a, b, c, d] = RANK_ZERO;
        assert!(range_basis(a, b, c, d).is_empty(), "rank 0:空基底");
    }

    /// 可達性判定:rank 1 的值域是直線 span{(1,2)},線上的 (3,6) 可達、
    /// 線外的 (1,1) 不可達;滿秩則整個 ℝ² 都可達。
    #[test]
    fn range_contains_judges_reachability() {
        let [a, b, c, d] = RANK_ONE;
        assert!(range_contains(a, b, c, d, 3.0, 6.0));
        assert!(!range_contains(a, b, c, d, 1.0, 1.0));
        let [a, b, c, d] = FULL_RANK;
        assert!(range_contains(a, b, c, d, -7.5, 4.25), "滿秩:處處可達");
    }

    /// 映成判定與見證的對偶(core 對偶律的 binding 重述):onto ⟺ 見證為空;
    /// 不映成時見證必須真的不可達 —— 前端紅色標記的數學保證。
    #[test]
    fn unreachable_witness_dual_to_is_onto() {
        for m in [FULL_RANK, RANK_ONE, RANK_ZERO] {
            let [a, b, c, d] = m;
            let witness = unreachable_vector(a, b, c, d);
            assert_eq!(witness.is_empty(), is_onto(a, b, c, d), "對偶律");
            if let [wx, wy] = witness[..] {
                assert!(!range_contains(a, b, c, d, wx, wy), "見證居然可達");
            }
        }
    }

    /// solve_for_input 的三種結局,與 range_contains 對帳(可達 ⟺ kind ≠ 3):
    /// 滿秩 → Unique 且回傳的 x 經 transform_point 左乘必須回到 w(兩路會合);
    /// rank 1 線上 → Infinite(一整條輸入);線外 → Inconsistent。
    #[test]
    fn solve_for_input_classifies_and_returns_witness() {
        // 滿秩:唯一輸入,且 A·x = w(存在性的見證拿去矩陣路徑驗收)
        let [a, b, c, d] = FULL_RANK;
        let out = solve_for_input(a, b, c, d, 5.0, 3.0);
        assert_eq!(out[0], 1.0, "可逆 → Unique");
        let back = transform_point(a, b, c, d, out[1], out[2]);
        assert!((back[0] - 5.0).abs() < 1e-9 && (back[1] - 3.0).abs() < 1e-9);

        // rank 1:線上 → Infinite、線外 → Inconsistent;與 range_contains 一致
        let [a, b, c, d] = RANK_ONE;
        assert_eq!(solve_for_input(a, b, c, d, 3.0, 6.0)[0], 2.0);
        assert_eq!(solve_for_input(a, b, c, d, 1.0, 1.0)[0], 3.0);
        for (wx, wy) in [(3.0, 6.0), (1.0, 1.0), (0.0, 0.0)] {
            let kind = solve_for_input(a, b, c, d, wx, wy)[0];
            assert_eq!(
                range_contains(a, b, c, d, wx, wy),
                kind != 3.0,
                "可達 ⟺ 有解"
            );
        }
    }
}
