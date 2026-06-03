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
    }
}
