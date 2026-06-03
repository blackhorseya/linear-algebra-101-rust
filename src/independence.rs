//! independence —— 向量之間的「冗餘」:線性獨立與它的失敗。這是 Theorem 1.6「onto」
//! 問題的一對一對偶。
//!
//! 對應原始 Go 專案 commit `5615951`
//! (`feat(independence): detect redundant vectors via Theorem 1.7`)。
//!
//! Theorem 1.7 —— 以向量為矩陣 A 的**行**,下列等價:
//! - (a) 這些行線性獨立、
//! - (b) 齊次系統 Ax=0 只有平凡解 x=0、
//! - (c) rank(A)=n(**行**數)、
//! - (d) 每一行都是 pivot 行(無自由行;nullity=0)、
//! - (e) 沒有行是其他行的線性組合。
//!
//! Theorem 1.6 把 rank 比 m(列,「填滿目標空間嗎」),Theorem 1.7 比 n(行,「有向量
//! 多餘嗎」)。引擎是 (b):x=0 永遠是解,所以獨立性就是問「它是不是**唯一**的解」。
//! 一個非零解 c 恰好是一組不全為零的權重,使 c₁v₁+…+cₙvₙ=0 —— 相依的定義。

use crate::{Span, Vector};

/// 這些向量是否線性獨立:把它們組合成零向量的唯一方式,是全零權重。依 Theorem 1.7
/// 這等於 rank(A)=n(A 以向量為行,n 是向量個數)。
///
/// 空集依慣例**獨立**,且**不需特判**:空 Span 的 dimension 是 0,恰好等於 0 個向量,
/// 公式 `0 == 0` 自然回 true。含 (近)零向量的清單**必相依**,因為 1·0=0 是非平凡組合 ——
/// 下面的快路徑用一次便宜掃描解決它,rank 判準則作為 backstop 同樣會抓到(零行不貢獻 rank)。
pub fn is_linearly_independent(epsilon: f64, vectors: &[Vector]) -> bool {
    // 快路徑:任何 (近)零向量都單獨逼出相依 —— 對 vₖ≈0,組合 0·v₁+…+1·vₖ+…+0·vₙ=0 是
    // 非平凡的(vₖ 的權重 1≠0)。一次 O(mn) 掃描就解決這些 case,省下後面 Span/dimension
    // 裡 O(mn²) 的消去。安全:rank 判準對它們會給同樣答案,這只是讓簡單情形更快。
    if vectors.iter().any(|v| v.is_approx_zero(epsilon)) {
        return false;
    }
    // 獨立 ⟺ rank 等於數量 —— 每個向量都帶來新方向,無一多餘。`dimension()` 即 rank(A)。
    // 對照上一課的 `spans_all`:那裡 dimension 比**列數**(onto),這裡比**向量個數**。
    Span::new(epsilon, vectors.to_vec()).dimension() == vectors.len()
}

/// `is_linearly_independent` 的否定:至少有一個向量多餘(是其他向量的線性組合)。
pub fn is_linearly_dependent(epsilon: f64, vectors: &[Vector]) -> bool {
    !is_linearly_independent(epsilon, vectors)
}

/// 有多少向量是多餘的:n − rank,即 A 的 nullity。這個數**良定**(要丟掉幾個才能達到
/// 獨立)—— 即使**哪些**向量算多餘並不唯一(在 {e₀,e₁,e₀+e₁} 裡任一個都是另兩個的
/// 組合)。獨立恰好就是 `redundancy_count == 0`。
///
/// 空集自然回 0(0 個向量 − 維度 0),且因 rank ≤ n,減法永不 underflow —— 皆不需特判。
pub fn redundancy_count(epsilon: f64, vectors: &[Vector]) -> usize {
    vectors.len() - Span::new(epsilon, vectors.to_vec()).dimension()
}

/// 回傳一組可以**移除而不縮小 span** 的向量索引:消去法判定的自由行。移除這些剩下
/// pivot 行 —— 一組仍張出相同空間的極大獨立子集(span 的基底)。
///
/// 對應數學的注意事項:哪些向量「可移除」**不唯一**(見 [`redundancy_count`] 的例子),
/// 所以回傳消去法做的特定選擇,不是「那個」冗餘集合;但它的長度恆等於 `redundancy_count`。
pub fn removable_columns(epsilon: f64, vectors: &[Vector]) -> Vec<usize> {
    // 空集自然回空 vec(空 Span 的 free_columns 即 []),不需特判。
    Span::new(epsilon, vectors.to_vec()).free_columns()
}

/// A 的行(向量排成行)化成 **RREF** 後,各行是否為 ℝᵐ 中**相異的標準向量** —— Theorem 1.8
/// 的條件 (e),也是與 [`is_linearly_independent`] 殊途同歸的**獨立路徑**。它不數 rank,而是
/// 讀階梯:當每行都是 pivot 行,完全化簡會把第 j 行變成 pivot 列為 1、其餘為 0 —— 恰是那一
/// 列的標準向量 —— 而相異的 pivot 列使各行相異。自由(冗餘)行則化簡成非標準的行(pivot 列上
/// 的分量是組合權重),破壞此性質。空集 vacuously true(沒有行能違反它)。
pub fn rref_columns_are_distinct_standard_vectors(epsilon: f64, vectors: &[Vector]) -> bool {
    // 先把臨時 Span 綁進變數:column_matrix() 借的是它。若寫成 `Span::new(...).column_matrix()`
    // 一行,臨時 Span 會在該行結束就 drop,回傳的借用懸空。
    let span = Span::new(epsilon, vectors.to_vec());
    let Some(matrix) = span.column_matrix() else {
        return true; // 空集 vacuously true
    };
    let rref = matrix.reduced_row_echelon_form(epsilon);
    let mut seen = vec![false; rref.rows()];
    for j in 0..rref.cols() {
        // 找這一行的 pivot 列:唯一一個 ≈1 的位置,其餘分量都得 ≈0。
        let col = rref.column(j).unwrap();
        let mut pivot_row = None;
        for (i, &entry) in col.entries().iter().enumerate() {
            if entry.abs() <= epsilon {
                continue; // ≈ 0
            } else if (entry - 1.0).abs() <= epsilon {
                if pivot_row.is_some() {
                    return false; // 同一行有兩個 ≈1 → 非標準向量
                }
                pivot_row = Some(i);
            } else {
                return false; // 非 ≈0 也非 ≈1 → 非標準向量
            }
        }
        // 整行掃完才定奪:沒有 ≈1 是零行/非標準;pivot 列重複則不相異。三種結局就地處理,
        // 不必第二趟掃 seen(原本那段之所以寫不對,正是因為把這步拆出去了)。
        match pivot_row {
            None => return false,
            Some(idx) if seen[idx] => return false,
            Some(idx) => seen[idx] = true,
        }
    }
    true
}

/// 回傳**第一個**是其前面各向量線性組合的向量 index —— Theorem 1.9 的建構式見證。清單
/// 線性獨立時(沒有這種向量,含空集)回 `None`(對應 Go 的 `-1`,但用 `Option` 表達)。
///
/// Theorem 1.9 把相依拆成兩種情形 —— u₀=0,或某個 uᵢ(i≥1)是前驅 u₀…uᵢ₋₁ 的組合 ——
/// 但**單一測試**就統一了它們:「uᵢ 在它前面那些向量的 span 裡嗎?」i=0 時那個 span 是空
/// span {0},測試化為「u₀=0」,恰是 Theorem 1.9 的第一種情形,無需特判。這是冗餘的**由左
/// 至右 / 序列**觀點,與 [`removable_columns`] 的全域 RREF 觀點不同 —— 但這裡回傳的 index
/// 恰好是最小的自由行(見 law test)。
pub fn first_dependent_index(epsilon: f64, vectors: &[Vector]) -> Option<usize> {
    for (i, v) in vectors.iter().enumerate() {
        // span{vectors[..i]} 是前驅張出的平直集;i=0 時是空 span {0},contains 自動化為
        // 「v 是零向量」(u₀=0 情形),不必特判。contains 即「v 是前驅的線性組合」。
        if Span::new(epsilon, vectors[..i].to_vec()).contains(v) {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const INDEP_EPS: f64 = 1e-9;

    /// 從字面值建向量的測試輔助。
    fn v(data: Vec<f64>) -> Vector {
        Vector::from_vec(data)
    }

    #[test]
    fn is_linearly_independent_known_cases() {
        // 手算對照,涵蓋 rank 判準必須答對的邊界慣例(空集、零向量)。
        let cases: Vec<(Vec<Vector>, bool)> = vec![
            (vec![], true),                                      // 空集 vacuously 獨立
            (vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0])], true),  // 兩軸獨立
            (vec![v(vec![1.0, 2.0]), v(vec![2.0, 4.0])], false), // v₂=2v₁ 相依
            // e₀+e₁ 是前兩個的和 → 相依
            (
                vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0]), v(vec![1.0, 1.0])],
                false,
            ),
            // ℝ² 裡塞 3 個向量,維度不夠 → 必相依
            (
                vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0]), v(vec![2.0, 3.0])],
                false,
            ),
            (vec![v(vec![0.0, 0.0])], false), // 單獨一個零向量相依(1·0=0)
            // 數值零:每個分量遠低於 INDEP_EPS(1e-9),容差版快路徑視為零向量 → 相依。
            // 精確 is_zero 會漏掉它;此 case 釘住容差行為,防退回精確比較。
            (vec![v(vec![1e-12, 1e-12])], false),
            // 門檻另一側:1e-6 遠高於 INDEP_EPS,是真正的非零向量,快路徑不吞 → 獨立。
            (vec![v(vec![1e-6, 0.0])], true),
            (vec![v(vec![3.0, 4.0])], true), // 單一非零向量獨立
        ];
        for (vectors, want) in cases {
            assert_eq!(
                is_linearly_independent(INDEP_EPS, &vectors),
                want,
                "獨立判定不符: {vectors:?}"
            );
            // dependent 必為 independent 的嚴格否定。
            assert_eq!(
                is_linearly_dependent(INDEP_EPS, &vectors),
                !want,
                "dependent 應為 independent 的否定: {vectors:?}"
            );
        }
    }

    #[test]
    fn redundancy_count_known_cases() {
        // 良定的冗餘數(= nullity),與「怪罪哪個向量」無關。
        let cases: Vec<(Vec<Vector>, usize)> = vec![
            (vec![], 0),
            (vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0])], 0), // 獨立對:無冗餘
            // 3 個共平面向量,rank 2 → 不管怪哪個,就是 1 個多餘
            (
                vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0]), v(vec![1.0, 1.0])],
                1,
            ),
            // 全是 (1,1) 的倍數:rank 1 → 2 個多餘
            (
                vec![v(vec![1.0, 1.0]), v(vec![2.0, 2.0]), v(vec![3.0, 3.0])],
                2,
            ),
        ];
        for (vectors, want) in cases {
            assert_eq!(
                redundancy_count(INDEP_EPS, &vectors),
                want,
                "冗餘數不符: {vectors:?}"
            );
        }
    }

    #[test]
    fn removable_columns_known_choice() {
        // {e₀, e₁, e₀+e₁}:第 0、1 行是 pivot,第 2 行(和)是自由/可移除的那一個。
        let vectors = vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0]), v(vec![1.0, 1.0])];
        let mut got = removable_columns(INDEP_EPS, &vectors);
        got.sort_unstable();
        assert_eq!(got, vec![2], "消去法應挑第 2 行為可移除");
    }

    #[test]
    fn removable_columns_empty() {
        // 邊界:沒有向量就沒得移除,回空 vec(不 panic、不索引越界)。
        assert_eq!(removable_columns(INDEP_EPS, &[]), Vec::<usize>::new());
    }

    #[test]
    fn rref_columns_are_distinct_standard_vectors_known_cases() {
        // 性質是關於 A 的 RREF,不是原向量。獨立輸入化簡成相異標準向量;冗餘行化簡成非標準。
        let cases: Vec<(Vec<Vector>, bool)> = vec![
            (vec![], true),                                     // 空集 vacuously true
            (vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0])], true), // 標準軸化簡為自己
            // 獨立但非軸對齊:RREF 是 identity,各行變 e₀,e₁(讀的是化簡後,不是輸入)
            (vec![v(vec![1.0, 1.0]), v(vec![1.0, -1.0])], true),
            // v₂=2v₁:自由行化簡成 (2,0)ᵀ —— pivot 列是 2 非 ≈1 → 非標準
            (vec![v(vec![1.0, 2.0]), v(vec![2.0, 4.0])], false),
            // 三共平面 → false
            (
                vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0]), v(vec![1.0, 1.0])],
                false,
            ),
            // 兩行都化簡為 e₀:各自是標準向量,但**相同** —— 只有相異性檢查能擋,這個 case
            // 讓那個檢查是承重的而非防禦性的。
            (vec![v(vec![1.0, 0.0]), v(vec![1.0, 0.0])], false),
            (vec![v(vec![0.0, 0.0])], false), // 零向量(整行為零)
            (vec![v(vec![3.0, 4.0])], true),  // 單一非零 → 化簡為 e₀
        ];
        for (vectors, want) in cases {
            assert_eq!(
                rref_columns_are_distinct_standard_vectors(INDEP_EPS, &vectors),
                want,
                "RREF 標準向量簽名不符: {vectors:?}"
            );
        }
    }

    #[test]
    fn rref_signature_matches_independence() {
        // Theorem 1.8 (a)⟺(e):RREF「相異標準向量」簽名必須與基於 rank 的判定一致。
        // 兩種無關的計算 —— 數 pivot vs 讀化簡後各行 —— 必須回報同一個真值。
        let cases: Vec<Vec<Vector>> = vec![
            vec![],
            vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0])],
            vec![v(vec![1.0, 2.0]), v(vec![2.0, 4.0])],
            vec![v(vec![1.0, 0.0]), v(vec![1.0, 0.0])],
            vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0]), v(vec![1.0, 1.0])],
            vec![v(vec![0.0, 0.0])],
            vec![v(vec![3.0, 4.0])],
            vec![
                v(vec![1.0, 1.0, 0.0]),
                v(vec![0.0, 1.0, 1.0]),
                v(vec![1.0, 0.0, 1.0]),
            ],
            vec![v(vec![1.0, 2.0, 3.0]), v(vec![2.0, 4.0, 6.0])],
        ];
        for vectors in cases {
            assert_eq!(
                is_linearly_independent(INDEP_EPS, &vectors),
                rref_columns_are_distinct_standard_vectors(INDEP_EPS, &vectors),
                "Theorem 1.8 (a)⟺(e) 破裂: {vectors:?}"
            );
        }
    }

    #[test]
    fn first_dependent_index_known_cases() {
        // 釘住 Theorem 1.9 的建構式見證,含**由左至右**語意:回傳第一個落在前驅 span 裡的
        // 向量,且 u₀=0 在 index 0 回報、無需特判。None 表獨立。
        let cases: Vec<(Vec<Vector>, Option<usize>)> = vec![
            (vec![], None),                                     // 空集獨立
            (vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0])], None), // 獨立軸:無相依 index
            // u₀=0 是 Theorem 1.9 的第一種情形 —— 在 index 0 回報
            (vec![v(vec![0.0, 0.0]), v(vec![1.0, 0.0])], Some(0)),
            // u₁=2·u₀ 是第一個落在前驅 span 裡的:index 1,不是 2
            (
                vec![v(vec![1.0, 0.0]), v(vec![2.0, 0.0]), v(vec![0.0, 1.0])],
                Some(1),
            ),
            // (1,1) 需要**兩個**前驅;較早的都不冗餘,故見證是最後一個 → index 2
            (
                vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0]), v(vec![1.0, 1.0])],
                Some(2),
            ),
            // 重複向量在它第二次出現處相依 → index 1
            (vec![v(vec![1.0, 0.0]), v(vec![1.0, 0.0])], Some(1)),
            (vec![v(vec![3.0, 4.0])], None), // 單一非零向量獨立
        ];
        for (vectors, want) in cases {
            assert_eq!(
                first_dependent_index(INDEP_EPS, &vectors),
                want,
                "first_dependent_index 不符: {vectors:?}"
            );
        }
    }
}

/// Theorem 1.7 的 property test —— 把定理變成跨隨機矩陣的可執行斷言。四個等價條件各自
/// 走**不同的程式路徑**(rank 計數、Solve 的 RREF+解分類、自由行掃描),它們一致才是
/// 真正的交叉驗證,而非套套邏輯。
#[cfg(test)]
mod laws {
    use super::*;
    use crate::{Matrix, Solution, System};
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

    /// 產生長度 `n`、元素為 [-10, 10] 整數的向量。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|x| x as f64).collect()))
    }

    proptest! {
        /// (a) 行獨立、(b) Ax=0 唯一解、(c) rank=n、(d) nullity=0,在每種形狀的隨機 A 上
        /// 必須回報**同一**判定。
        #[test]
        fn theorem_1_7_conditions_agree(
            a in (1usize..=5, 1usize..=5).prop_flat_map(|(r, c)| int_matrix(r, c)),
        ) {
            const EPS: f64 = 1e-9;
            let rows = a.rows();
            let cols = a.cols();
            let columns: Vec<Vector> = (0..cols).map(|j| a.column(j).unwrap()).collect();

            let cond_a = is_linearly_independent(EPS, &columns); // (a)
            // (b) Ax=0:b=0 必相容,結局只會是 Unique(獨立)或 Infinite(相依),不會 Inconsistent。
            let system = System::new(a.clone(), Vector::new(rows)).unwrap();
            let cond_b = matches!(system.solve(EPS), Solution::Unique(_)); // (b)
            let cond_c = a.rank(EPS) == cols; // (c)
            let cond_d = a.nullity(EPS) == 0; // (d)

            prop_assert!(
                cond_a == cond_b && cond_b == cond_c && cond_c == cond_d,
                "Theorem 1.7 判準不一致: (a)={cond_a} (b)={cond_b} (c)={cond_c} (d)={cond_d}\n a={a:?}"
            );
        }

        /// 冗餘的操作意義:丟掉可移除(自由)行後,(1) span 不變、(2) 剩下的線性獨立。
        /// 合起來就是「從生成集萃取出一組基底」的定義 —— 並重用上一課的 `Span::equals`
        /// 精確比較兩個無限集合。
        #[test]
        fn removable_columns_leave_independent_spanning_set(
            vectors in (1usize..=4, 1usize..=5)
                .prop_flat_map(|(rows, count)| prop::collection::vec(int_vector(rows), count)),
        ) {
            const EPS: f64 = 1e-9;
            let removable = removable_columns(EPS, &vectors);
            let kept: Vec<Vector> = vectors
                .iter()
                .enumerate()
                .filter(|(j, _)| !removable.contains(j))
                .map(|(_, v)| v.clone())
                .collect();

            // (1) 丟掉可移除向量後,span 不變。
            let full = Span::new(EPS, vectors.clone());
            let reduced = Span::new(EPS, kept.clone());
            prop_assert!(full.equals(&reduced), "丟掉可移除行改變了 span");

            // (2) 倖存者線性獨立 —— span 的一組基底。
            prop_assert!(is_linearly_independent(EPS, &kept), "保留的行不獨立");

            // 計數對得上:kept + removable = 原數量。
            prop_assert_eq!(kept.len() + removable.len(), vectors.len());
        }

        /// Theorem 1.8 的七條件等價:(a) 獨立、(b) Ax=0 至多一解、(c) nullity=0、(d) rank=n、
        /// (e) RREF 各行為相異標準向量、(f) Ax=0 唯一解、(g) 每行有 pivot —— 跨隨機矩陣同判定。
        /// 只有 (e) 走與 rank 無關的獨立路徑,故一致才是真正的交叉驗證。
        #[test]
        fn theorem_1_8_conditions_agree(
            a in (1usize..=5, 1usize..=5).prop_flat_map(|(r, c)| int_matrix(r, c)),
        ) {
            const EPS: f64 = 1e-9;
            let rows = a.rows();
            let cols = a.cols();
            let columns: Vec<Vector> = (0..cols).map(|j| a.column(j).unwrap()).collect();
            let sys0 = System::new(a.clone(), Vector::new(rows)).unwrap();

            let cond_a = is_linearly_independent(EPS, &columns);
            let cond_b = sys0.has_at_most_one_solution(EPS);
            let cond_c = a.nullity(EPS) == 0;
            let cond_d = a.rank(EPS) == cols;
            let cond_e = rref_columns_are_distinct_standard_vectors(EPS, &columns);
            let cond_f = matches!(sys0.solve(EPS), Solution::Unique(_));
            let cond_g = a.pivot_columns(EPS).len() == cols;

            prop_assert!(
                cond_a == cond_b && cond_a == cond_c && cond_a == cond_d
                    && cond_a == cond_e && cond_a == cond_f && cond_a == cond_g,
                "Theorem 1.8 條件不一致: (a)={cond_a} (b)={cond_b} (c)={cond_c} (d)={cond_d} (e)={cond_e} (f)={cond_f} (g)={cond_g}\n a={a:?}"
            );
        }

        /// (b) 的「**對每個** b」內容(只看 b=0 的 agree-test 摸不到):獨立時每個 b 至多一解
        /// (nullity 0 不留自由變數);相依時 b=0 已有無限多解,故「對每個 b 至多一解」在 b=0
        /// 就被見證失敗 —— 這個逆向是**確定性**的,故兩邊都斷言。
        #[test]
        fn theorem_1_8_at_most_one_for_every_b(
            (a, bs) in (1usize..=5, 1usize..=5).prop_flat_map(|(r, c)| {
                (int_matrix(r, c), prop::collection::vec(int_vector(r), 1..=5))
            }),
        ) {
            const EPS: f64 = 1e-9;
            let rows = a.rows();
            let cols = a.cols();
            let columns: Vec<Vector> = (0..cols).map(|j| a.column(j).unwrap()).collect();

            if is_linearly_independent(EPS, &columns) {
                // 保證方向:任何 b 都不可能有超過一解。
                for b in bs {
                    let system = System::new(a.clone(), b.clone()).unwrap();
                    prop_assert!(
                        system.has_at_most_one_solution(EPS),
                        "獨立卻有某個 b 有 >1 解\n a={a:?} b={b:?}"
                    );
                }
            } else {
                // 相依:b=0 見證「對每個 b 至多一解」失敗。
                let sys0 = System::new(a.clone(), Vector::new(rows)).unwrap();
                prop_assert!(
                    !sys0.has_at_most_one_solution(EPS),
                    "相依卻在 b=0 報 at most one\n a={a:?}"
                );
            }
        }

        /// Theorem 1.9 兩面:(1)「前驅 span 裡的第一個向量」存在 ⟺ 相依;(2) 該 index =
        /// **最小自由行**。序列式 span-membership 走訪(`first_dependent_index`,建在 A·c=v
        /// 一致性上)與一次性 RREF 自由行掃描(`removable_columns`)是兩條獨立路徑,而某行為
        /// 自由 ⟺ 它落在前面各行的 span 裡 —— 故兩者的「第一個」相符。
        #[test]
        fn theorem_1_9_first_dependent_index(
            vectors in (1usize..=4, 1usize..=5)
                .prop_flat_map(|(rows, count)| prop::collection::vec(int_vector(rows), count)),
        ) {
            const EPS: f64 = 1e-9;
            let first = first_dependent_index(EPS, &vectors);
            // (1) 見證存在 ⟺ 相依
            prop_assert_eq!(first.is_some(), is_linearly_dependent(EPS, &vectors));
            // (2) 見證 = 最小自由行(都沒有時兩邊皆 None)
            let smallest_free = removable_columns(EPS, &vectors).into_iter().min();
            prop_assert_eq!(first, smallest_free);
        }
    }
}
