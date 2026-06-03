//! basis —— `ℝ^dim` 的一組**基底**:既張滿空間、又線性獨立的最小生成集。
//!
//! 對應原始 Go 專案 commit `e3589e6`
//! (`feat(basis): verify a minimal spanning set with IsBasis (Theorems 1.6 and 1.7)`)。
//!
//! `ℝ^dim` 的一組基底是一份向量清單,同時是:
//! - **生成的(spanning)** —— 它們的 span 是整個 `ℝ^dim`(Theorem 1.6,onto)、
//! - **獨立的(independent)** —— 沒有一個多餘(Theorem 1.7,一對一)。
//!
//! 「最小」是雙向的:抽掉任一個就不再張滿(獨立性不容鬆弛),加入任一個就不再獨立。
//! 所以基底是「仍能張滿的最小集合」,等價地也是「仍保持獨立的最大集合」。
//!
//! 這個檔案做的是**初步**工作:**驗證**一份清單是不是基底;還沒做從任意生成集**萃取**
//! 一組基底 —— [`removable_columns`](crate::removable_columns) 是下一步的引擎。

use crate::{Span, Vector, is_linearly_independent};

/// 這些向量是否構成 `ℝ^dim` 的一組基底:既張滿整個空間、又線性獨立。這兩句恰好就是
/// Theorem 1.6 與 1.7,所以基底正是「兩個定理同時成立」之處。
///
/// 一個微妙的推論**免費掉出來、不必顯式檢查**:spanning 逼出 rank=dim(=m),independent
/// 逼出 rank=向量個數(=n),於是 `ℝ^dim` 的任何基底**恰好有 dim 個向量** —— 矩陣是方的。
/// 這就是定理「`ℝ^dim` 的每組基底都有 dim 個元素」,它從兩者的**合取**浮現,而非被假設。
pub fn is_basis(epsilon: f64, dim: usize, vectors: &[Vector]) -> bool {
    let spanning = Span::new(epsilon, vectors.to_vec()).spans_all(dim);
    let independent = is_linearly_independent(epsilon, vectors);
    spanning && independent
}

/// 這些向量是否**依序**為 `ℝ^dim` 的**標準**基底:e₀, e₁, …, e_{dim−1}(單位矩陣的各行)。
/// 這是基底的典範例子,也比 [`is_basis`] 更嚴格 —— 它在意順序與精確的向量,而不只是
/// 「它們獨立地張滿空間」。
pub fn is_standard_basis(epsilon: f64, dim: usize, vectors: &[Vector]) -> bool {
    if vectors.len() != dim {
        return false;
    }
    vectors.iter().enumerate().all(|(i, v)| {
        let e_i = Vector::standard(dim, i).unwrap();
        v.approx_equals(&e_i, epsilon)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    const BASIS_EPS: f64 = 1e-9;

    /// 從字面值建向量的測試輔助。
    fn v(data: Vec<f64>) -> Vector {
        Vector::from_vec(data)
    }

    #[test]
    fn is_basis_known_cases() {
        // 手算對照,涵蓋兩種失敗模式:向量太少(張不滿)與向量相依(數量對也不獨立)。
        let cases: Vec<(usize, Vec<Vector>, bool)> = vec![
            // ℝ² 的標準基底
            (2, vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0])], true),
            // 歪斜但仍獨立地張滿 ℝ² → 仍是基底
            (2, vec![v(vec![1.0, 1.0]), v(vec![1.0, -1.0])], true),
            // 獨立但只有一個:張成一條線,非 ℝ² → 不 spanning
            (2, vec![v(vec![1.0, 0.0])], false),
            // 數量對(ℝ² 要 2 個)但相依:同一條線上 → 不 independent
            (2, vec![v(vec![1.0, 2.0]), v(vec![2.0, 4.0])], false),
            // ℝ² 裡 3 個向量:張得滿,但不可能獨立 → 非基底
            (
                2,
                vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0]), v(vec![1.0, 1.0])],
                false,
            ),
            // ℝ³ 的標準基底
            (
                3,
                vec![
                    v(vec![1.0, 0.0, 0.0]),
                    v(vec![0.0, 1.0, 0.0]),
                    v(vec![0.0, 0.0, 1.0]),
                ],
                true,
            ),
        ];
        for (dim, vectors, want) in cases {
            assert_eq!(
                is_basis(BASIS_EPS, dim, &vectors),
                want,
                "is_basis 不符 (dim={dim}): {vectors:?}"
            );
        }
    }

    #[test]
    fn is_standard_basis_is_order_sensitive() {
        let e0 = v(vec![1.0, 0.0, 0.0]);
        let e1 = v(vec![0.0, 1.0, 0.0]);
        let e2 = v(vec![0.0, 0.0, 1.0]);

        assert!(
            is_standard_basis(BASIS_EPS, 3, &[e0.clone(), e1.clone(), e2.clone()]),
            "e0,e1,e2 是 ℝ³ 的標準基底"
        );
        // 同樣的向量、錯誤的順序:是基底,但不是「標準」基底。
        assert!(
            !is_standard_basis(BASIS_EPS, 3, &[e1.clone(), e0.clone(), e2.clone()]),
            "亂序的標準向量不是依序的標準基底"
        );
        // 數量不對。
        assert!(
            !is_standard_basis(BASIS_EPS, 3, &[e0.clone(), e1.clone()]),
            "兩個向量不可能是 ℝ³ 的標準基底"
        );
        // 非標準(但合法)的基底也不是標準基底。
        assert!(
            !is_standard_basis(BASIS_EPS, 2, &[v(vec![1.0, 1.0]), v(vec![1.0, -1.0])]),
            "歪斜的基底不是標準基底"
        );
    }

    #[test]
    fn identity_columns_are_basis() {
        // 典範事實:I_n 的各行(標準向量 e₀…e_{n−1})構成 ℝⁿ 的基底,n=1..=5。
        // 這也為 laws 的 basis_forces_square_count 提供非 vacuous 保證:每個 n 都產生
        // 一個 dim×dim 的方陣基底。
        for n in 1..=5 {
            let id = Matrix::identity(n);
            let columns: Vec<Vector> = (0..n).map(|j| id.column(j).unwrap()).collect();
            assert!(
                is_basis(BASIS_EPS, n, &columns),
                "I_{n} 的行應是 ℝ^{n} 的基底"
            );
        }
    }
}

/// Theorem 1.6 與 1.7 合流的 property test —— 把「基底 = 生成 ∧ 獨立」變成可執行的等價,
/// 並驗證從中浮現的結構定理「基底必為方陣(向量數 = dim)」。
#[cfg(test)]
mod laws {
    use super::*;
    use crate::Matrix;
    use proptest::prelude::*;

    /// 產生 `rows×cols`、元素為 [-10, 10] 整數的矩陣(f64 下精確)。
    fn int_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-10i64..=10, cols), rows).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|x| x as f64).collect())
                    .collect(),
            )
        })
    }

    proptest! {
        /// `is_basis` 必須等於分別算出的「columns 張滿 ℝ^dim 且 columns 獨立」 —— 防止
        /// 兩個 clause 各自漂移。
        #[test]
        fn basis_is_spanning_and_independent(
            a in (1usize..=5, 1usize..=5).prop_flat_map(|(r, c)| int_matrix(r, c)),
        ) {
            const EPS: f64 = 1e-9;
            let rows = a.rows();
            let cols = a.cols();
            let columns: Vec<Vector> = (0..cols).map(|j| a.column(j).unwrap()).collect();

            let spanning = Span::new(EPS, columns.clone()).spans_all(rows);
            let independent = is_linearly_independent(EPS, &columns);
            prop_assert_eq!(
                is_basis(EPS, rows, &columns),
                spanning && independent,
                "is_basis 與 (spanning ∧ independent) 不一致\n a={:?}", a
            );
        }

        /// 從合取浮現的結構定理:`is_basis` 為真時,向量數**必然**等於 dim。我們從不在
        /// `is_basis` 裡檢查數量 —— spanning 要 rank=dim、independent 要 rank=count,兩者
        /// 逼出 count=dim。非方陣的 A 必然 `is_basis` 為假(vacuously 通過);非 vacuous 由
        /// `identity_columns_are_basis` 確定性涵蓋(每個 n 都產生一個方陣基底)。
        #[test]
        fn basis_forces_square_count(
            a in (1usize..=4, 1usize..=4).prop_flat_map(|(r, c)| int_matrix(r, c)),
        ) {
            const EPS: f64 = 1e-9;
            let rows = a.rows();
            let cols = a.cols();
            let columns: Vec<Vector> = (0..cols).map(|j| a.column(j).unwrap()).collect();

            if is_basis(EPS, rows, &columns) {
                prop_assert_eq!(
                    columns.len(), rows,
                    "ℝ^dim 的基底必須恰有 dim 個向量,卻有 {} 個 (dim={})", columns.len(), rows
                );
            }
        }
    }
}
