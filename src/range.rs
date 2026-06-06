//! 值域(Range)與映成(Onto)—— 線性轉換的「輸出端」幾何。
//!
//! 筆記「線性轉換的合成與可逆性 1/2」章(單元 5-3,講義 2.8 前段)。
//! 上一章(Theorem 2.9)建立了 T ↔ A 的一一對應;這一章追問輸出端:
//! **T 的影像到底蓋住 codomain ℝᵐ 的多少?** 全章五個概念都是同一個等號的變奏:
//!
//! > **Range(T) = Col(A)** —— 值域 = 標準矩陣行向量張成的空間(column space)。
//!
//! 函數論的詞彙經這個等號逐一翻譯成矩陣語彙,而右邊的工具第三、四單元都刻好了:
//!
//! | 函數視角(本模組) | 矩陣視角(既有積木) |
//! |---|---|
//! | 值域的生成集合 | A 的所有行([`Matrix::column`]) |
//! | w 可達(w ∈ Range) | Ax = w 相容([`System::is_consistent`]) |
//! | T 映成(onto) | rank(A) = m(Theorem 2.10) |
//! | 值域的基底 | pivot 行對應的原始行([`Matrix::pivot_columns`]) |
//!
//! 本模組**沒有任何新演算法,只有積木接線** —— 把等號寫進程式的依賴關係裡。
//!
//! 與 `elimination` / `inverse` 同款佈局:跨在 `transformation` 模組外、
//! 碰不到 private 欄位,一律走 public API([`Transformation::matrix`] getter
//! 是 Theorem 2.9 給的讀取通道:T ↦ A 唯一,「取出轉換的標準矩陣」才良定義)。
//!
//! [`System::is_consistent`]: crate::System::is_consistent

use crate::{Transformation, Vector};

impl Transformation {
    /// 值域的生成集合(generating set for the range):**Range(T) = Col(A) =
    /// Span{A 的各行}**,故 A 的 n 支行向量整組就是值域的生成集合。
    ///
    /// 為什麼?T(x) = Ax = x₁·a₁ + ⋯ + xₙ·aₙ(矩陣–向量乘法的 column view,
    /// 第二單元的核心觀念)—— 每個輸出都是行向量的線性組合,反之每個線性組合
    /// 都取得到(係數就是輸入 x)。「所有輸出」與「行的張成」是同一個集合。
    ///
    /// 不會失敗、也不收 epsilon:純資料提取,無消去、無判零(0 行矩陣回空 Vec)。
    /// 集合**允許冗餘**(零行、相依行照收)—— 剔除冗餘是 `range_basis`(練習 5)
    /// 的工作,生成與基底是兩個不同的概念。
    ///
    /// 實作提示:行抽取第二單元就刻好了([`Matrix::column`],經
    /// [`matrix()`](Transformation::matrix) 取 A)—— 沿「已存在就不重刻」,
    /// 這題是 `(0..n).map(…).collect()` 的迭代器慣用法;j < n 是迴圈不變式,
    /// `column(j)` 的 Err 是被證明的死路(先守衛、再 unwrap)。
    ///
    /// [`Matrix::column`]: crate::Matrix::column
    pub fn range_generating_set(&self) -> Vec<Vector> {
        let a = self.matrix();
        (0..self.domain_dim())
            .map(|j| a.column(j).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Matrix, Transformation, Vector};

    /// 題目原例:A = [[3,-4,0],[2,0,1]](ℝ³ → ℝ²)的值域生成集合是
    /// A 的三支行向量 (3,2)、(-4,0)、(0,1) —— 順序沿行索引,各自住在 codomain ℝ²。
    /// 整數元素在 f64 下精確,用精確 equals。
    #[test]
    fn range_generating_set_of_formula_example() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![3.0, -4.0, 0.0],
            vec![2.0, 0.0, 1.0],
        ]));
        let gens = t.range_generating_set();
        assert_eq!(gens.len(), 3, "一行一支生成元素,共 n = domain_dim 支");
        assert!(gens[0].equals(&Vector::from_vec(vec![3.0, 2.0])));
        assert!(gens[1].equals(&Vector::from_vec(vec![-4.0, 0.0])));
        assert!(gens[2].equals(&Vector::from_vec(vec![0.0, 1.0])));
    }

    /// 驗收條件:生成元素的維度必須是**列數 m(codomain)**,不是行數 n ——
    /// 「行向量有 m 個分量」,與 5-1 的「m×n 唸法 vs ℝⁿ → ℝᵐ 方向」同一個陷阱。
    #[test]
    fn range_generating_set_lives_in_codomain() {
        // 4×2:ℝ² → ℝ⁴,兩支生成元素、每支 4 個分量
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]));
        let gens = t.range_generating_set();
        assert_eq!(gens.len(), t.domain_dim());
        for g in &gens {
            assert_eq!(g.rows(), t.codomain_dim(), "生成元素住在 codomain ℝᵐ");
        }
    }

    /// 驗收條件:含零行的矩陣 —— 零向量是**合法的生成元素**(對張成毫無貢獻,
    /// 但生成集合的定義照收;剔除冗餘是 range_basis 的事,概念不混)。
    #[test]
    fn range_generating_set_keeps_zero_columns() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 0.0]]));
        let gens = t.range_generating_set();
        assert_eq!(gens.len(), 2);
        assert!(gens[0].equals(&Vector::from_vec(vec![1.0, 0.0])));
        assert!(gens[1].is_zero(), "零行照收,不擅自剔除");
    }
}

/// 值域與映成的 property test —— 本章的 laws 幾乎都是**跨練習交叉對帳**,
/// 隨練習推進逐條累積(策略沿 transformation laws 的「先抽形狀、再抽內容」)。
#[cfg(test)]
mod laws {
    use crate::{Matrix, Transformation};
    use proptest::prelude::*;

    /// 隨機形狀(1..=4 × 1..=4)、元素為 [-10, 10] 整數的矩陣
    /// (f64 下加減乘完全精確;形狀也隨機,涵蓋 ℝⁿ → ℝᵐ 各種組合)。
    fn int_matrix_any_shape() -> impl Strategy<Value = Matrix> {
        (1usize..=4, 1usize..=4).prop_flat_map(|(rows, cols)| {
            prop::collection::vec(prop::collection::vec(-10i64..=10, cols), rows).prop_map(|grid| {
                Matrix::from_rows(
                    grid.into_iter()
                        .map(|row| row.into_iter().map(|v| v as f64).collect())
                        .collect(),
                )
            })
        })
    }

    proptest! {
        // 形狀律:生成集合恆有 n(domain_dim)支、每支住在 codomain ℝᵐ ——
        // 「n 行、每行 m 個分量」用轉換自身的詞彙重讀一遍。
        #[test]
        fn generating_set_shape_matches_dimensions(a in int_matrix_any_shape()) {
            let t = Transformation::new(a);
            let gens = t.range_generating_set();
            prop_assert_eq!(gens.len(), t.domain_dim(), "一行一支生成元素");
            for g in &gens {
                prop_assert_eq!(g.rows(), t.codomain_dim(), "生成元素住在 ℝᵐ");
            }
        }
    }
}
