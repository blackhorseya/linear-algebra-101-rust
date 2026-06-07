//! 合成(Composition)與可逆性(Invertibility)—— 線性轉換的「函數操作」層級。
//!
//! 筆記「線性轉換的合成與可逆性 2/2」章(單元 5-4,講義 2.8 後段)。
//! 5-3(`range` 模組)用 **Range(T) = Col(A)** 翻譯了「集合的等號」;
//! 本章翻譯「**操作**的等號」—— 函數世界對函數做的每件事,矩陣世界都有對應的運算:
//!
//! | 函數視角(本模組) | 矩陣視角(既有積木) | 定理 |
//! |---|---|---|
//! | 一對一(one-to-one) | rank(A) = n([`Matrix::rank`]) | Theorem 2.11 |
//! | 合成 U ∘ T | 乘積 BA([`Matrix::multiply`]) | T_B ∘ T_A = T_BA |
//! | 逆轉換 T⁻¹ | 反矩陣 A⁻¹([`Matrix::inverse`]) | Theorem 2.13 |
//! | 可逆 ⟺ 雙射(1-1 且 onto) | A 可逆(rank = n = m) | Theorem 2.12 |
//!
//! 與 `range` 同款:**零新演算法、零新錯誤 variant,只有積木接線** ——
//! 乘法章、可逆矩陣章、`elimination` 的 rank / nullity 全部現成。
//! 這一章把 Chapter 2 的「函數 ↔ 矩陣」字典編完:Theorem 2.9 翻譯了**物件**
//! (T ↔ A)、5-3 翻譯了**集合**(Range ↔ Col)、本章翻譯**操作**(∘ ↔ ×、⁻¹ ↔ ⁻¹)。
//!
//! [`Matrix::rank`]: crate::Matrix::rank
//! [`Matrix::multiply`]: crate::Matrix::multiply
//! [`Matrix::inverse`]: crate::Matrix::inverse

use crate::Transformation;

impl Transformation {
    /// 一對一(one-to-one)判定:**T 一對一 ⟺ rank(A) = n**(Theorem 2.11)。
    ///
    /// 「一對一」問的是輸入端:不同輸入必到不同輸出(T(u) = T(v) ⟹ u = v)。
    /// 線性世界裡這句話可以收攏到原點:T(u) = T(v) ⟺ A(u − v) = 0,所以
    /// 「不撞」⟺「Ax = 0 只有零解」⟺ nullity = 0 ⟺ rank = n(rank–nullity)。
    ///
    /// 與 5-3 的 [`is_onto`](Transformation::is_onto) 是完美對偶:
    /// **onto 問 rank 搆不搆得到 m(輸出端蓋滿),1-1 問 rank 搆不搆得到 n
    /// (輸入端不浪費)** —— 同一個數字,兩端各問一次。
    ///
    /// 題目驗收的「n > m 必非一對一」**不需特判**:rank ≤ min(m, n) < n,
    /// 數學自己排掉 —— ℝ³ 塞進 ℝ² 必有不同輸入被擠到同一點(鴿籠)。
    ///
    /// 實作提示:一行,與 `is_onto` 只差「對上哪個維度」—— 5-1 的老陷阱
    /// 第三次上演:n 是行數(domain),用轉換自身的詞彙講。
    pub fn is_one_to_one(&self, epsilon: f64) -> bool {
        self.matrix().rank(epsilon) == self.domain_dim()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Matrix, Transformation};

    /// 題目原例(一):2×3 矩陣(ℝ³ → ℝ²)—— n = 3 > m = 2,鴿籠直接判死:
    /// rank ≤ 2 < 3,三維輸入擠進二維輸出必有相撞。
    #[test]
    fn is_one_to_one_rejects_map_from_higher_dimension() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![3.0, -4.0, 0.0],
            vec![2.0, 0.0, 1.0],
        ]));
        assert!(!t.is_one_to_one(1e-9));
    }

    /// 題目原例(二):3×2 且 rank = 2(嵌入 ℝ² → ℝ³)—— 行向量獨立,
    /// 不同輸入到不同輸出。注意它**不映成**(值域只是 ℝ³ 裡的 xy 平面):
    /// 1-1 與 onto 是兩個獨立的性質 —— 與 5-3 的投影測試恰成對偶。
    #[test]
    fn is_one_to_one_accepts_full_column_rank_embedding() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ]));
        assert!(t.is_one_to_one(1e-9));
        assert!(!t.is_onto(1e-9), "嵌入 1-1 但不映成 —— 兩性質獨立");
    }

    /// 5-3 的投影(ℝ³ → ℝ²)反向對照:它映成(x、y 全保留)卻**不**一對一
    /// —— z 軸整條被吸到原點,(0,0,1) 與 (0,0,2) 撞在同一個輸出。
    #[test]
    fn is_one_to_one_rejects_onto_projection() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ]));
        assert!(!t.is_one_to_one(1e-9));
        assert!(t.is_onto(1e-9), "投影映成但不 1-1 —— 與嵌入恰成對偶");
    }

    /// 高瘦**不保證** 1-1,還得看 rank:3×2 但兩行共線(rank 1 < 2)——
    /// 第二行是冗餘的,x = (2, −1) 與 x = (0, 0) 都被送到原點。
    #[test]
    fn is_one_to_one_rejects_tall_with_dependent_columns() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
        ]));
        assert!(!t.is_one_to_one(1e-9));
    }

    /// 可逆方陣:rank = n = m,一對一(也映成 —— 5-4 的合流預告)。
    #[test]
    fn is_one_to_one_accepts_invertible_square() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 1.0]]));
        assert!(t.is_one_to_one(1e-9));
    }

    /// 行成比例的方陣:rank 1 < 2 —— 整條直線 span{(2, −1)} 被吸到原點,
    /// 「撞在一起」的輸入不只一對,而是一整條。
    #[test]
    fn is_one_to_one_rejects_rank_deficient_square() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]));
        assert!(!t.is_one_to_one(1e-9));
    }
}

/// 合成與可逆性的 property test —— 沿 5-3 的傳統:**跨練習交叉對帳**,
/// 隨練習推進逐條累積(策略沿「先抽形狀、再抽內容」的依賴式兩階段抽樣)。
#[cfg(test)]
mod laws {
    use crate::{Matrix, Transformation};
    use proptest::prelude::*;

    /// 消去法判零門檻(整數輸入的殘差遠低於此,沿 range laws 的 EPS)。
    const EPS: f64 = 1e-9;

    /// 固定 `rows×cols`、元素為 [-10, 10] 整數的矩陣(f64 下加減乘完全精確)。
    fn int_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-10i64..=10, cols), rows).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            )
        })
    }

    /// 隨機形狀(1..=4 × 1..=4)的整數矩陣 —— 涵蓋 ℝⁿ → ℝᵐ 各種組合。
    fn int_matrix_any_shape() -> impl Strategy<Value = Matrix> {
        (1usize..=4, 1usize..=4).prop_flat_map(|(rows, cols)| int_matrix(rows, cols))
    }

    /// 寬矮矩陣(cols > rows):cols = rows + extra,**依建構**保證比高還寬
    /// —— 5-3 `tall_int_matrix` 的鏡像(那邊測「高瘦必不映成」,這邊測
    /// 「寬矮必非 1-1」:同一個 rank ≤ min(m, n),兩端各擠死一次)。
    fn wide_int_matrix() -> impl Strategy<Value = Matrix> {
        (1usize..=3, 1usize..=3).prop_flat_map(|(rows, extra)| int_matrix(rows, rows + extra))
    }

    proptest! {
        // 題目提示的交叉驗證:1-1 ⟺ nullity = 0 —— Theorem 2.11 的另一半臉。
        // rank–nullity(rank + nullity = n)保證兩判準等價,但 is_one_to_one
        // 數的是 pivot 行、nullity 數的是 free 行,兩條獨立路徑必須同答案。
        #[test]
        fn one_to_one_iff_nullity_zero(a in int_matrix_any_shape()) {
            let t = Transformation::new(a.clone());
            prop_assert_eq!(t.is_one_to_one(EPS), a.nullity(EPS) == 0);
        }

        // 鴿籠(Theorem 2.11 的維度限制半邊):cols > rows ⟹ rank ≤ m < n,
        // 高維塞進低維必撞 —— 5-3「高瘦必不映成」的鏡像律。
        #[test]
        fn wider_than_tall_is_never_one_to_one(a in wide_int_matrix()) {
            let t = Transformation::new(a);
            prop_assert!(!t.is_one_to_one(EPS));
        }

        // 轉置對偶(5-3 ↔ 5-4 的橋):T_A 一對一 ⟺ T_{Aᵀ} 映成 ——
        // rank(A) = rank(Aᵀ)(row rank = column rank),而 Aᵀ 的 m 是 A 的 n。
        // 「1-1 與 onto 是同一個數字從兩端讀」用 transpose 寫成可跑的定理。
        #[test]
        fn one_to_one_iff_transpose_is_onto(a in int_matrix_any_shape()) {
            let t = Transformation::new(a.clone());
            let t_transpose = Transformation::new(a.transpose());
            prop_assert_eq!(t.is_one_to_one(EPS), t_transpose.is_onto(EPS));
        }

        // IMT 接線(方陣):1-1 ⟺ 可逆 —— 與 5-3 的 onto ⟺ 可逆合起來,
        // 方陣的「1-1 / onto / 可逆」三位一體(練習 4 的綜合判定表預告)。
        #[test]
        fn square_transformation_one_to_one_iff_invertible(a in int_matrix(3, 3)) {
            let t = Transformation::new(a.clone());
            prop_assert_eq!(t.is_one_to_one(EPS), a.is_invertible(EPS));
        }
    }
}
