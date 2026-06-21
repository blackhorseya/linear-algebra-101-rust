//! 子空間:零空間(Null space)圖解(單元 6-2)—— 給前端 `/nullspace` 用,
//! 與 `/range` 對偶:Range = Col A 是「**輸出端**蓋住多少」,Null A 是「**輸入端**
//! 壓扁多少」。同一個 T 的兩端各切出一個子空間。
//!
//! 全部直接委派 core:`null_space_contains`(`subspace` 模組)與 `Matrix` 的
//! `rank` / `nullity`。binding 只做 2×2 攤平,零演算法(沿 range 章「只有積木
//! 接線」的精神)。前端 rank-nullity 定理(rank + nullity = 2)的兩個數**各自
//! 由 core 獨立算**,相加 = 2 是當場對帳,不是前端推導。
//! epsilon 一律寫死 `TRACE_EPSILON`(沿 range 章:拖曳座標數量級穩定)。

use super::helpers::{TRACE_EPSILON, transformation_2x2};
use crate::Vector;
use wasm_bindgen::prelude::*;

/// `v ∈ Null A`?直接委派 core 的 `null_space_contains`(v ∈ Null A ⟺ Av ≈ 0)——
/// 前端 v 箭頭「在核裡」的綠色由 core 當場判,不是 JS 寫死的條件。
#[wasm_bindgen]
pub fn null_space_contains(a: f64, b: f64, c: f64, d: f64, vx: f64, vy: f64) -> bool {
    transformation_2x2(a, b, c, d)
        .null_space_contains(&Vector::from_vec(vec![vx, vy]), TRACE_EPSILON)
}

/// Null A 的維度(nullity):被壓到原點的**獨立輸入方向數**。
/// 0 → 核 = {0}(可逆,沒有非零輸入被壓扁)、1 → 核是一條過原點的線、
/// 2 → 整個 domain 都被壓扁(A = 0)。委派 `Matrix::nullity`。
#[wasm_bindgen]
pub fn nullity(a: f64, b: f64, c: f64, d: f64) -> usize {
    transformation_2x2(a, b, c, d)
        .matrix()
        .nullity(TRACE_EPSILON)
}

/// Col A 的維度(rank)。與 [`nullity`] 滿足 rank-nullity 定理:**rank + nullity
/// = 2**(domain 維度)—— 前端把兩個獨立算出的數相加,當場驗證這個守恆。
/// 委派 `Matrix::rank`。
#[wasm_bindgen]
pub fn rank(a: f64, b: f64, c: f64, d: f64) -> usize {
    transformation_2x2(a, b, c, d).matrix().rank(TRACE_EPSILON)
}

/// Row A 的基底(core 的 `row_space_basis`,Theorem 4.8:**RREF 的非零列**),
/// 列向量攤平串接:`[]`(rank 0:Row A = {0})、`[x, y]`(rank 1:Row A 是直線)、
/// `[x₁, y₁, x₂, y₂]`(rank 2:Row A = ℝ²)—— 支數 = dim Row A。與 range 章的
/// `range_basis`(Col A 基底)**對偶**:給 `/rank` 頁並排畫「domain 的 Row A」
/// 與「codomain 的 Col A」。回的是 **RREF 列**(canonical),非原始列。
#[wasm_bindgen]
pub fn row_space_basis(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    transformation_2x2(a, b, c, d)
        .row_space_basis(TRACE_EPSILON)
        .iter()
        .flat_map(|v| v.entries().iter().copied())
        .collect()
}

/// dim Row A,經 **rank(Aᵀ)** 獨立算出(轉置後數 pivot)。與 [`rank`](rank)
/// (= rank(A) = dim Col A)**恆相等** —— 這就是 `rank(A) = rank(Aᵀ)`
/// (= dim Row A = dim Col A,整章最深的定理)。前端把這兩個**獨立計算**的數
/// 並列,當場對帳(不是前端湊的)。
#[wasm_bindgen]
pub fn rank_transpose(a: f64, b: f64, c: f64, d: f64) -> usize {
    transformation_2x2(a, b, c, d)
        .matrix()
        .transpose()
        .rank(TRACE_EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::transform::transform_point;

    /// 四種秩的代表(row-major `[a, b, c, d]`):
    const FULL_RANK: [f64; 4] = [2.0, 1.0, 1.0, 1.0]; // 可逆 → nullity 0
    const RANK_ONE: [f64; 4] = [1.0, 2.0, 2.0, 4.0]; // 行共線 → nullity 1,核方向 (2,-1)
    const PROJECTION: [f64; 4] = [1.0, 0.0, 0.0, 0.0]; // 投影到 x 軸 → 核 = y 軸
    const ZERO: [f64; 4] = [0.0, 0.0, 0.0, 0.0]; // 零矩陣 → nullity 2

    /// rank-nullity 定理:rank + nullity = 2(domain 維度),四種秩全中 ——
    /// 前端面板「兩數相加 = 2」的數學保證。
    #[test]
    fn rank_plus_nullity_is_domain_dim() {
        for m in [FULL_RANK, RANK_ONE, PROJECTION, ZERO] {
            let [a, b, c, d] = m;
            assert_eq!(
                rank(a, b, c, d) + nullity(a, b, c, d),
                2,
                "rank-nullity 破裂"
            );
        }
    }

    /// 投影到 x 軸:核 = y 軸 —— (0,1) 被壓扁(在核裡)、(1,0) 保留(不在核裡)。
    #[test]
    fn null_space_contains_classifies_projection_kernel() {
        let [a, b, c, d] = PROJECTION;
        assert!(null_space_contains(a, b, c, d, 0.0, 1.0), "y 軸被壓到原點");
        assert!(
            !null_space_contains(a, b, c, d, 1.0, 0.0),
            "x 軸保留,不在核裡"
        );
        assert_eq!(nullity(a, b, c, d), 1, "投影核是一條線");
    }

    /// 核成員 ⟺ 像為零(與 transform_point 對帳):v ∈ Null A **就是** Av ≈ 0,
    /// 兩條路徑(成員判定 vs 直接左乘)必須給同一個答案 —— 前端綠色高亮的依據。
    #[test]
    fn null_member_iff_image_is_zero() {
        let [a, b, c, d] = RANK_ONE; // 核方向 (2,-1):A·(2,-1) = (0,0)
        for (vx, vy) in [(2.0, -1.0), (4.0, -2.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)] {
            let img = transform_point(a, b, c, d, vx, vy);
            let image_is_zero = img[0].abs() < 1e-9 && img[1].abs() < 1e-9;
            assert_eq!(
                null_space_contains(a, b, c, d, vx, vy),
                image_is_zero,
                "({vx},{vy}) 的成員判定與像是否為零不一致"
            );
        }
    }

    /// 可逆 → 核 = {0}:除了原點,沒有非零輸入被壓扁(nullity 0,與 /range 的
    /// 「滿秩處處可達」對偶 —— 同一個可逆性,兩端各說一次)。
    #[test]
    fn invertible_has_trivial_kernel() {
        let [a, b, c, d] = FULL_RANK;
        assert_eq!(nullity(a, b, c, d), 0);
        assert!(null_space_contains(a, b, c, d, 0.0, 0.0), "0 永遠在核裡");
        assert!(
            !null_space_contains(a, b, c, d, 1.0, 1.0),
            "可逆不壓扁非零向量"
        );
    }

    /// 零矩陣 → 整個平面都是核(nullity 2):任何 v 都被壓到原點。
    #[test]
    fn zero_matrix_kernel_is_everything() {
        let [a, b, c, d] = ZERO;
        assert_eq!(nullity(a, b, c, d), 2);
        for (vx, vy) in [(1.0, 0.0), (0.0, 1.0), (3.0, -5.0)] {
            assert!(null_space_contains(a, b, c, d, vx, vy), "零矩陣壓扁一切");
        }
    }

    /// row_space_basis 的攤平長度說完 dim Row A:4(ℝ²)、2(直線)、0(原點);
    /// rank 1 取的是 **RREF 非零列**(canonical),非原始列。
    #[test]
    fn row_space_basis_length_encodes_row_dimension() {
        let [a, b, c, d] = FULL_RANK;
        assert_eq!(row_space_basis(a, b, c, d).len(), 4, "rank 2:Row A = ℝ²");
        let [a, b, c, d] = RANK_ONE; // [1,2,2,4] → RREF 非零列 (1,2)
        assert_eq!(
            row_space_basis(a, b, c, d),
            vec![1.0, 2.0],
            "rank 1:RREF 列"
        );
        let [a, b, c, d] = ZERO;
        assert!(row_space_basis(a, b, c, d).is_empty(), "rank 0:空基底");
    }

    /// rank(A) = rank(Aᵀ) = dim Row A = dim Col A:四種秩全中 —— 前端「兩個獨立
    /// 計算對帳」的數學保證(整章最深的定理,binding 端再釘一次)。
    #[test]
    fn rank_transpose_always_equals_rank() {
        for m in [FULL_RANK, RANK_ONE, PROJECTION, ZERO] {
            let [a, b, c, d] = m;
            assert_eq!(
                rank_transpose(a, b, c, d),
                rank(a, b, c, d),
                "rank(A) ≠ rank(Aᵀ)"
            );
        }
    }
}
