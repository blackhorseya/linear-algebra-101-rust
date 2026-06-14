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

/// 縮減定理(Reduction Theorem,**Theorem 4.3**):把一個**生成集** S 蒸餾成
/// 它所張成空間 Span(S) 的一組**基底** —— 丟掉冗餘(線性相依)的向量,留下一個
/// 仍張出同一空間的極大獨立子集。
///
/// 這正是 [`is_basis`] 模組開頭預告的「下一步」:`is_basis` 只**驗證**一份清單
/// 是不是基底,這支才真的從任意生成集**萃取**出基底。
///
/// **怎麼挑該留誰?** 把 S 的向量當矩陣 A 的**行**,化 RREF,留下 **pivot 行對應
/// 的原始向量**。根據是行對應定理(Column Correspondence Theorem):列運算不改變
/// **行之間**的線性關係 —— pivot 落在哪幾行,原矩陣的那幾支行就是獨立的,其餘行
/// 都是它們的組合(可丟)。
///
/// **經典陷阱(同 [`range_basis`])**:回的是**原始 S 的向量**,不是 RREF 的行 ——
/// RREF 的 pivot 行長得像 eᵢ,通常根本不在 Span(S) 裡。「保留原始向量」是測試釘死的。
///
/// 與既有積木的三角關係:
/// - 與 [`removable_columns`](crate::removable_columns) **互補**:它回「該丟的」
///   (自由行),這支回「該留的」(pivot 行)—— 同一刀的兩面,合起來是全部。
/// - 與 [`range_basis`](crate::Transformation::range_basis) 是**同一操作的一般版**:
///   `reduce_to_basis(A 的各行)` 把那些行排回矩陣恰好重建 A,於是與 `range_basis(A)`
///   逐向量相等 —— 行空間基底只是「對 A 的行做縮減」這個特例(下方 law 對帳)。
///
/// 邊界:
/// - **空集 → 空基底**:Span(∅) = {0},零子空間的基底是 ∅,維度 0(不需特判,
///   `column_matrix` 對空集回 `None`)。
/// - **含零向量 → 自動丟掉**:零行不貢獻 pivot,不會被選進來(不需特判)。
///
/// 結果**不唯一**(哪些算冗餘不唯一,見 [`redundancy_count`](crate::redundancy_count)
/// 的例子),回消去法做的特定選擇;但其**大小恆等於** `Span::dimension()` = rank ——
/// 這就是維度良定(**Theorem 4.5**:任兩基底等勢),由下方 law 釘住。
///
/// 實作提示:三個積木接線 ——
/// 1. [`Span::new`](crate::Span)`(epsilon, s.clone())` 把向量擺成矩陣的行;
/// 2. [`Span::column_matrix`](crate::Span::column_matrix) 取出那個矩陣
///    (空集回 `None` → 回空 `Vec`),對它呼叫
///    [`pivot_columns`](crate::Matrix::pivot_columns) 拿 pivot 索引;
/// 3. pivot 索引 < `s.len()`(依建構),`s[j].clone()` 取**原始**向量 ——
///    與 `range_basis` 同款的 map-collect。
pub fn reduce_to_basis(epsilon: f64, s: Vec<Vector>) -> Vec<Vector> {
    let span = Span::new(epsilon, s.clone());
    let pivot_columns = span
        .column_matrix()
        .map_or(vec![], |m| m.pivot_columns(epsilon));
    pivot_columns.into_iter().map(|j| s[j].clone()).collect()
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

    // ---- reduce_to_basis(Theorem 4.3 縮減定理)----

    #[test]
    fn reduce_to_basis_textbook_example() {
        // 題目原例:三向量,中間那支是第一支的 2 倍(冗餘)→ 縮減成第一、三支。
        // u₁=(1,2,-1)、u₂=2u₁=(2,4,-2)、u₃=(1,0,1)。pivot 落在第 0、2 行。
        let u1 = v(vec![1.0, 2.0, -1.0]);
        let u2 = v(vec![2.0, 4.0, -2.0]);
        let u3 = v(vec![1.0, 0.0, 1.0]);
        let basis = reduce_to_basis(BASIS_EPS, vec![u1.clone(), u2, u3.clone()]);
        assert_eq!(basis.len(), 2, "rank 2 → 兩支基底");
        assert!(basis[0].equals(&u1), "保留第一支 (1,2,-1)");
        assert!(basis[1].equals(&u3), "保留第三支 (1,0,1)");
    }

    #[test]
    fn reduce_to_basis_keeps_original_not_rref() {
        // 經典陷阱:回的是原始向量,不是 RREF 的行。(1,2) 與 (3,6) 共線,rank 1,
        // 縮減保留**原始的** (1,2),而非 RREF 化簡出的 e₀=(1,0)。
        let basis = reduce_to_basis(BASIS_EPS, vec![v(vec![1.0, 2.0]), v(vec![3.0, 6.0])]);
        assert_eq!(basis.len(), 1);
        assert!(
            basis[0].equals(&v(vec![1.0, 2.0])),
            "保留原始 (1,2),不是 RREF 的 (1,0)"
        );
    }

    #[test]
    fn reduce_to_basis_independent_set_unchanged() {
        // 已獨立 → 沒有冗餘可丟,原樣返回(題 1 的洞察:獨立集對自家 span 已是基底)。
        let vectors = vec![v(vec![1.0, 0.0]), v(vec![1.0, 1.0])];
        let basis = reduce_to_basis(BASIS_EPS, vectors.clone());
        assert_eq!(basis.len(), 2, "獨立集一個都不丟");
        assert!(basis[0].equals(&vectors[0]) && basis[1].equals(&vectors[1]));
    }

    #[test]
    fn reduce_to_basis_empty_is_empty() {
        // 邊界:Span(∅) = {0},零子空間的基底是 ∅(維度 0)。不 panic、不越界。
        // (Vector 無 PartialEq —— 專案刻意只給 equals/approx_equals,故用 is_empty 斷言。)
        assert!(reduce_to_basis(BASIS_EPS, vec![]).is_empty());
    }

    #[test]
    fn reduce_to_basis_drops_zero_vector() {
        // 零向量是冗餘的(零行不貢獻 pivot)→ 自動被丟掉,只留下非零的那支。
        let basis = reduce_to_basis(BASIS_EPS, vec![v(vec![0.0, 0.0]), v(vec![1.0, 1.0])]);
        assert_eq!(basis.len(), 1);
        assert!(basis[0].equals(&v(vec![1.0, 1.0])));
    }

    #[test]
    fn reduce_to_basis_all_parallel_keeps_one() {
        // 全是 (1,1) 的倍數:rank 1 → 縮減成一支(消去法挑第一支 pivot)。
        let basis = reduce_to_basis(
            BASIS_EPS,
            vec![v(vec![1.0, 1.0]), v(vec![2.0, 2.0]), v(vec![3.0, 3.0])],
        );
        assert_eq!(basis.len(), 1, "一條線 → 維度 1");
        assert!(basis[0].equals(&v(vec![1.0, 1.0])));
    }
}

/// Theorem 1.6 與 1.7 合流的 property test —— 把「基底 = 生成 ∧ 獨立」變成可執行的等價,
/// 並驗證從中浮現的結構定理「基底必為方陣(向量數 = dim)」。
#[cfg(test)]
mod laws {
    use super::*;
    use crate::{Matrix, Transformation};
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

    /// 產生長度 `n`、元素為 [-10, 10] 整數的向量(f64 下加減乘完全精確)。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|x| x as f64).collect()))
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

        /// Theorem 4.3(縮減定理):`reduce_to_basis(S)` 從任意生成集萃取出基底 ——
        /// (1) 結果線性獨立、(2) span 與原集合相同、(3) 是原集合的子集(原始向量,非
        /// RREF 行)。三條合起來就是「基底」的定義加上「萃取自 S」。與 independence 的
        /// `removable_columns_leave_independent_spanning_set` 互為表裡:那條從「丟自由行」
        /// 證,這條從「留 pivot 行」證,同一刀的兩面。
        #[test]
        fn reduce_to_basis_is_independent_subset_spanning_same(
            s in (1usize..=4, 1usize..=5)
                .prop_flat_map(|(rows, count)| prop::collection::vec(int_vector(rows), count)),
        ) {
            const EPS: f64 = 1e-9;
            let basis = reduce_to_basis(EPS, s.clone());
            // (1) 獨立
            prop_assert!(is_linearly_independent(EPS, &basis), "萃取的基底不獨立");
            // (2) span 不變
            prop_assert!(
                Span::new(EPS, s.clone()).equals(&Span::new(EPS, basis.clone())),
                "縮減改變了 span"
            );
            // (3) 每支基底都是 S 的某個原始向量
            for b in &basis {
                prop_assert!(s.iter().any(|u| u.equals(b)), "基底含非原始向量");
            }
        }

        /// Theorem 4.5(維度良定)+ 題 4:萃取出的基底大小**恆等於** rank ——
        /// 不管生成集多冗餘、消去法挑哪幾支,基底大小都釘死在 `Span::dimension()`(= rank)。
        /// 這就是「維度與基底選擇無關」的可執行版本:維度權威是 rank,不必另立型別。
        #[test]
        fn reduce_to_basis_size_equals_dimension(
            s in (1usize..=4, 1usize..=5)
                .prop_flat_map(|(rows, count)| prop::collection::vec(int_vector(rows), count)),
        ) {
            const EPS: f64 = 1e-9;
            prop_assert_eq!(
                reduce_to_basis(EPS, s.clone()).len(),
                Span::new(EPS, s).dimension(),
                "基底大小 ≠ rank"
            );
        }

        /// 題 1(基底 ⟺ 線性獨立):對自家 span,「S 是 Span(S) 的基底」⟺「S 獨立」——
        /// 因為 spanning 對自家 span 恆真。可執行版本:縮減**一個都不丟**(len 不變)
        /// ⟺ S 本來就線性獨立。把題 1 的洞察接到 reduce_to_basis 上,零新 API。
        #[test]
        fn reduce_to_basis_unchanged_iff_independent(
            s in (1usize..=4, 1usize..=5)
                .prop_flat_map(|(rows, count)| prop::collection::vec(int_vector(rows), count)),
        ) {
            const EPS: f64 = 1e-9;
            let unchanged = reduce_to_basis(EPS, s.clone()).len() == s.len();
            prop_assert_eq!(unchanged, is_linearly_independent(EPS, &s), "不丟 ⟺ 獨立 破裂");
        }

        /// 題 2(Col A 基底,三路對帳):`reduce_to_basis(A 的各行)` 必須
        /// (a) 與 [`range_basis`](Transformation::range_basis)`(A)` 逐向量相等(跑同一個
        /// 重建出的 A,取同一組 pivot)、(b) 每支都是 A 的某**原始**行、(c) 大小 = rank。
        /// 行空間基底只是「對 A 的行做縮減」的特例。
        #[test]
        fn col_space_basis_matches_range_basis(
            a in (1usize..=4, 1usize..=4).prop_flat_map(|(r, c)| int_matrix(r, c)),
        ) {
            const EPS: f64 = 1e-9;
            let columns: Vec<Vector> = (0..a.cols()).map(|j| a.column(j).unwrap()).collect();
            let reduced = reduce_to_basis(EPS, columns.clone());
            let range_basis = Transformation::new(a.clone()).range_basis(EPS);

            // (a) 與 range_basis 逐向量相等
            prop_assert_eq!(reduced.len(), range_basis.len(), "基底大小不一致");
            for (r, rb) in reduced.iter().zip(&range_basis) {
                prop_assert!(r.equals(rb), "reduce_to_basis(行) ≠ range_basis");
            }
            // (b) 每支都是 A 的原始行
            for r in &reduced {
                prop_assert!(columns.iter().any(|col| col.equals(r)), "基底含非原始行");
            }
            // (c) 大小 = rank
            prop_assert_eq!(reduced.len(), a.rank(EPS), "Col A 基底大小 ≠ rank");
        }

        /// Theorem 4.5 強版:**同一個** span 的不同生成集,萃取出的基底**等勢**。
        /// 構造法:S 與 S' = S ++ (S 的一個線性組合) span 相同(加進去的向量已在 Span(S)
        /// 裡),故 reduce 後大小必相等 —— 把「任兩基底等勢」直接做成可跑的命題。
        #[test]
        fn dimension_invariant_under_adding_redundant_generator(
            (s, weights) in (1usize..=4, 1usize..=4).prop_flat_map(|(rows, count)| {
                (
                    prop::collection::vec(int_vector(rows), count),
                    prop::collection::vec(-3i64..=3, count),
                )
            }),
        ) {
            const EPS: f64 = 1e-9;
            // redundant = Σ wᵢ·sᵢ,依建構落在 Span(S) 裡。
            let weights_f: Vec<f64> = weights.iter().map(|&w| w as f64).collect();
            let redundant = Vector::linear_combination(&weights_f, &s).unwrap();
            let mut s_plus = s.clone();
            s_plus.push(redundant);

            // S 與 S' span 相同 → 維度相同。
            prop_assert!(Span::new(EPS, s.clone()).equals(&Span::new(EPS, s_plus.clone())));
            prop_assert_eq!(
                reduce_to_basis(EPS, s).len(),
                reduce_to_basis(EPS, s_plus).len(),
                "加冗餘生成元素改變了維度"
            );
        }
    }
}
