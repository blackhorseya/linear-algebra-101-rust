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

use crate::{System, Transformation, Vector};

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

    /// 值域成員判定:**w ∈ Range(T) ⟺ Ax = w 相容(consistent)**。
    ///
    /// 「w 可達」問的是「找得到輸入 x 使 T(x) = w」—— 而 T(x) = Ax,
    /// 這句話與「方程組 Ax = w 有解」一字不差。於是成員判定**就是**
    /// 一致性判定,委派 [`System::is_consistent`](crate::System::is_consistent)
    /// (已存在就不重刻)。
    ///
    /// 題目要求「用 RREF 檢查增廣矩陣的矛盾列 [0…0 | d]」—— `is_consistent`
    /// 用的是 Theorem 1.5 條件 (d)(rank(A) == rank([A|w])),第四單元的 law
    /// `consistency_theorem_conditions_agree` 已證明兩判準等價,單一真相。
    ///
    /// `w.rows() != m`(w 不住在 codomain ℝᵐ)→ **false**:談不上可達
    /// (沿 [`verify_linearity`](crate::verify_linearity) 的 bool 述詞慣例)。
    /// 判零門檻 `epsilon` 由呼叫端指定(消去法引入捨入,浮點慣例)。
    ///
    /// 實作提示:[`System::new`](crate::System::new) **擁有**它的 A 與 b
    /// (move 進去),所以要付兩次 clone —— 學習庫不追效能,語意正確優先。
    /// 它在維度不合時回 `Err`,恰好就是你要折成 `false` 的那個情況:
    /// `Result → bool` 的收法(`match` / `map_or` / `is_ok_and`)由你選。
    pub fn range_contains(&self, w: &Vector, epsilon: f64) -> bool {
        let a = self.matrix();
        System::new(a.clone(), w.clone()).is_ok_and(|sys| sys.is_consistent(epsilon))
    }

    /// 映成(onto)判定:**T 映成 ⟺ rank(A) = m**(Theorem 2.10)。
    ///
    /// 「映成」是把成員判定全稱化:不只某個 w 可達,而是 **codomain 的每個
    /// w 都可達**(Range(T) = ℝᵐ)。逐 w 檢查不可能(ℝᵐ 不可數),Theorem 2.10
    /// 把它壓縮成一個數字的比較:Col(A) 是 ℝᵐ 的子空間,「子空間 = 全空間」
    /// ⟺ 「維度拉滿」—— 而 Col(A) 的維度就是 rank(第四單元的定義)。
    ///
    /// 題目驗收的「n < m 必不映成」**不需特判**:rank ≤ min(m, n) < m,
    /// 數學自己把這個 case 排掉了 —— ℝ² 的兩支行向量怎麼張也張不滿 ℝ³。
    ///
    /// 實作提示:一行 —— [`Matrix::rank`](crate::Matrix::rank) 對上哪個維度?
    /// 5-1 的老陷阱:m 是 `rows`(codomain),用轉換自身的詞彙講。
    pub fn is_onto(&self, epsilon: f64) -> bool {
        let a = self.matrix();
        a.rank(epsilon) == self.codomain_dim()
    }

    /// 不可達向量(unreachable vector):T 不映成時,找一支 **b ∉ Range(T)**
    /// (即 Ax = b 無解的 b)作為「值域沒蓋滿」的具體見證;映成則回 `None`。
    ///
    /// 策略:**標準基底掃描** —— 逐一檢查 e₁ … e_m,回第一支不可達的。
    /// 正確性三行證明:T 不映成 ⟹ Range(T) 是 ℝᵐ 的 **proper** subspace ⟹
    /// 必有某 eᵢ ∉ Range(否則 Range ⊇ Span{e₁…e_m} = ℝᵐ,矛盾)——
    /// proper subspace 裝不下整組 spanning set。法 `onto_iff_every_standard_
    /// basis_vector_reachable` 已把這個根據存證成可跑的定理。
    ///
    /// **不需要先問 `is_onto` 短路**:映成時所有 eᵢ 可達,掃描自然空手而回
    /// —— `Option` 的兩個 variant 與「映成 / 不映成」嚴絲合縫,零分支。
    ///
    /// 實作提示:`(0..m).map(…).find(…)` —— [`Vector::standard`] 取 eᵢ
    /// (i < m 是迴圈不變式,unwrap 安全),`find` 收「**不**可達」述詞
    /// (注意否定:找的是 `!range_contains` 的那支)。
    pub fn unreachable_vector(&self, epsilon: f64) -> Option<Vector> {
        let m = self.codomain_dim();
        (0..m)
            .map(|i| Vector::standard(m, i).unwrap())
            .find(|e_i| !self.range_contains(e_i, epsilon))
    }

    /// 值域的基底(basis for the range):從生成集合(練習 1,可能冗餘)
    /// 蒸餾出**獨立**的子集 —— 取 **pivot 行對應的原始行向量**。
    ///
    /// 根據是行對應定理(Column Correspondence Theorem):列運算不改變
    /// **行之間**的線性關係(Ax = 0 與 RREF(A)·x = 0 同解,而「第 j 行
    /// 是其他行的組合」正是一支特定的 x)—— 所以 RREF 裡 pivot 落在哪幾行,
    /// 原矩陣的**那幾支行**就是獨立的,且其餘行都是它們的組合。
    ///
    /// **經典陷阱**:回的是**原始 A 的行**,不是 RREF 的行 —— RREF 的
    /// pivot 行長得像 eᵢ,通常**根本不在 Col(A) 裡**(列運算保持的是
    /// 行之間的「關係」,不是行本身)。題目原例的第一支基底是 (1,2,0)
    /// 而非 e₁,測試就釘在這裡。
    ///
    /// 實作提示:又是兩個積木接線 ——
    /// [`Matrix::pivot_columns`](crate::Matrix::pivot_columns) 給索引、
    /// [`Matrix::column`](crate::Matrix::column) 給內容(pivot 索引
    /// 依建構 < cols,unwrap 安全)。與練習 1 比較:同樣的 map-collect,
    /// 只是索引來源從「全部 0..n」換成「pivot 那幾個」。
    pub fn range_basis(&self, epsilon: f64) -> Vec<Vector> {
        let a = self.matrix();
        let pivot_cols = a.pivot_columns(epsilon);
        pivot_cols.iter().map(|&j| a.column(j).unwrap()).collect()
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

    /// 練習 2 題目場景:投影到 xy 平面(ℝ² → ℝ³ 的嵌入)—— 值域是 z = 0 平面。
    /// (1,2,0) 落在平面上 → 可達;(1,2,3) 的 z ≠ 0 → 驗收點名的
    /// 「w 不在行生成空間」案例。
    #[test]
    fn range_contains_classifies_xy_plane_image() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ]));
        assert!(t.range_contains(&Vector::from_vec(vec![1.0, 2.0, 0.0]), 1e-9));
        assert!(!t.range_contains(&Vector::from_vec(vec![1.0, 2.0, 3.0]), 1e-9));
    }

    /// 行成比例(行空間塌成直線 span{(1,2)})—— 可達與否要消去後才看得出來,
    /// 不是逐格比對:w = (3,6) = 3·(1,2) 在線上 → true;(1,1) 不在 → false。
    #[test]
    fn range_contains_needs_elimination_to_decide() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]));
        assert!(t.range_contains(&Vector::from_vec(vec![3.0, 6.0]), 1e-9));
        assert!(!t.range_contains(&Vector::from_vec(vec![1.0, 1.0]), 1e-9));
    }

    /// 零向量永遠可達:T(0) = A·0 = 0,任何轉換的值域都含原點 ——
    /// 這正是「值域是子空間」最起碼的一條(子空間必過原點)。
    #[test]
    fn range_contains_zero_vector_for_any_transformation() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]));
        assert!(t.range_contains(&Vector::new(2), 1e-9));
    }

    /// w 不住在 codomain ℝᵐ → 談不上可達,回 false(bool 述詞慣例,
    /// 沿 verify_linearity / is_parallel:不在同一空間就不比)。
    #[test]
    fn range_contains_rejects_dimension_mismatch() {
        let t = Transformation::new(Matrix::new(3, 2)); // codomain ℝ³
        assert!(!t.range_contains(&Vector::from_vec(vec![1.0, 2.0]), 1e-9)); // w ∈ ℝ²
    }

    /// 練習 3 題目原例(一):RREF 為 I₃ 的方陣 —— rank = 3 = m,映成。
    #[test]
    fn is_onto_accepts_full_rank_square() {
        let t = Transformation::new(Matrix::identity(3));
        assert!(t.is_onto(1e-9));
    }

    /// 練習 3 題目原例(二):3×2(ℝ² → ℝ³)必不映成 —— 兩支行向量
    /// 怎麼張也張不滿 ℝ³(rank ≤ 2 < 3),不需特判、數學自己排掉。
    #[test]
    fn is_onto_rejects_map_into_higher_dimension() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ]));
        assert!(!t.is_onto(1e-9));
    }

    /// 壓縮方向(ℝ³ → ℝ²)可以映成:投影丟掉 z,但 x、y 全保留 ——
    /// rank = 2 = m。注意它**不是**一對一(z 軸整條被吸到原點),
    /// onto 與 one-to-one 是兩個獨立的性質,5-4 會合流到可逆性。
    #[test]
    fn is_onto_accepts_projection_onto_smaller_codomain() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ]));
        assert!(t.is_onto(1e-9));
    }

    /// 方陣也可能不映成:行成比例 → 值域塌成直線(rank 1 < 2),
    /// ℝ² 裡直線之外的點全都不可達。
    #[test]
    fn is_onto_rejects_rank_deficient_square() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]));
        assert!(!t.is_onto(1e-9));
    }

    /// 練習 4 題目原例:ℝ² 嵌入 ℝ³(值域 = xy 平面)—— e₁、e₂ 都在平面上,
    /// 掃描走到 e₃ = (0,0,1) 才出界 → 它就是見證;並用 range_contains
    /// 反驗這支見證確實不可達(Ax = b 無解)。
    #[test]
    fn unreachable_vector_finds_witness_off_xy_plane() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ]));
        let b = t.unreachable_vector(1e-9).expect("不映成必有見證");
        assert!(
            b.equals(&Vector::standard(3, 2).unwrap()),
            "掃描序:e₃ 是第一支出界的"
        );
        assert!(!t.range_contains(&b, 1e-9), "見證必須真的不可達");
    }

    /// 行成比例的方陣:值域塌成直線 span{(1,2)},e₁ = (1,0) 就不在線上 ——
    /// 掃描第一步即命中。
    #[test]
    fn unreachable_vector_exits_early_on_first_witness() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]));
        let b = t.unreachable_vector(1e-9).expect("rank 1 < 2,必有見證");
        assert!(b.equals(&Vector::standard(2, 0).unwrap()));
    }

    /// 映成的轉換沒有不可達向量:所有 eᵢ 可達,掃描空手而回 → None ——
    /// Option 的兩個 variant 與「映成 / 不映成」嚴絲合縫。
    #[test]
    fn unreachable_vector_is_none_for_onto() {
        let identity = Transformation::new(Matrix::identity(2));
        assert!(identity.unreachable_vector(1e-9).is_none());
        // 壓縮方向的投影(ℝ³ → ℝ²)也映成 —— None 不是方陣專利
        let projection = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ]));
        assert!(projection.unreachable_vector(1e-9).is_none());
    }

    /// 練習 5 題目原例:A = [[1,2,1],[2,4,0],[0,0,2]],第二行 = 2×第一行 ——
    /// pivot 落在行 0 與行 2,基底取**原始矩陣**的那兩支行。
    /// 陷阱在 basis[0]:行對應定理取的是 (1,2,0),錯拿 RREF 的 pivot 行
    /// 會得到 e₁ = (1,0,0) —— 那支根本不在 Col(A) 裡。
    #[test]
    fn range_basis_of_textbook_example_takes_original_columns() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 2.0, 1.0],
            vec![2.0, 4.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ]));
        let basis = t.range_basis(1e-9);
        assert_eq!(basis.len(), 2, "rank 2:行 1 是行 0 的兩倍,被剔除");
        assert!(
            basis[0].equals(&Vector::from_vec(vec![1.0, 2.0, 0.0])),
            "要的是原始 A 的行 0,不是 RREF 的 e₁"
        );
        assert!(basis[1].equals(&Vector::from_vec(vec![1.0, 0.0, 2.0])));
    }

    /// 零轉換:值域 = {0},基底是**空集合**(空集合張成 {0} 是慣例,
    /// 沿 linear_combination 的空集合語意)—— rank 0,一支都不取。
    #[test]
    fn range_basis_of_zero_transformation_is_empty() {
        let t = Transformation::zero(2, 3);
        assert!(t.range_basis(1e-9).is_empty());
    }

    /// 滿秩(可逆)方陣:沒有冗餘行,基底 = 整組生成集合 —— 蒸餾無事可做。
    #[test]
    fn range_basis_of_invertible_keeps_every_column() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 1.0]]));
        let basis = t.range_basis(1e-9);
        assert_eq!(basis.len(), 2);
        assert!(basis[0].equals(&Vector::from_vec(vec![2.0, 1.0])));
        assert!(basis[1].equals(&Vector::from_vec(vec![1.0, 1.0])));
    }
}

/// 值域與映成的 property test —— 本章的 laws 幾乎都是**跨練習交叉對帳**,
/// 隨練習推進逐條累積(策略沿 transformation laws 的「先抽形狀、再抽內容」)。
#[cfg(test)]
mod laws {
    use crate::{Matrix, Transformation, Vector};
    use proptest::prelude::*;

    /// 消去法判零門檻(整數輸入的殘差遠低於此,沿 elimination 的 REF_EPSILON)。
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

    /// 長度 `n`、元素為 [-10, 10] 整數的向量。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()))
    }

    /// 隨機形狀(1..=4 × 1..=4)的整數矩陣 —— 涵蓋 ℝⁿ → ℝᵐ 各種組合。
    fn int_matrix_any_shape() -> impl Strategy<Value = Matrix> {
        (1usize..=4, 1usize..=4).prop_flat_map(|(rows, cols)| int_matrix(rows, cols))
    }

    /// 隨機形狀的整數矩陣,連同一支**住在定義域 ℝⁿ** 的輸入向量
    /// (n 在同一次 flat_map 裡共用,沿 transformation laws 的依賴式兩階段抽樣)。
    fn int_matrix_with_input_vector() -> impl Strategy<Value = (Matrix, Vector)> {
        (1usize..=4, 1usize..=4)
            .prop_flat_map(|(rows, cols)| (int_matrix(rows, cols), int_vector(cols)))
    }

    /// 高瘦矩陣(rows > cols):rows = cols + extra,**依建構**保證比寬還高 ——
    /// 與 ero 策略的「dst ≠ src 依建構成立」同一招,免去 prop_assume 丟樣本。
    fn tall_int_matrix() -> impl Strategy<Value = Matrix> {
        (1usize..=3, 1usize..=3).prop_flat_map(|(cols, extra)| int_matrix(cols + extra, cols))
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

        // 影像必可達(練習 2 的定義律):∀ x ∈ ℝⁿ,T(x) ∈ Range(T) ——
        // 植入法:w := T(x) 依建構即在值域,range_contains 必須看得出來。
        // (值域是「所有輸出」,輸出當然在值域 —— 但 range_contains 走的是
        //  消去法的獨立路徑,兩條路會合才是測試的意義。)
        #[test]
        fn image_is_always_reachable((a, x) in int_matrix_with_input_vector()) {
            let t = Transformation::new(a);
            let w = t.apply(&x).unwrap(); // x 依建構住在定義域,unwrap 安全
            prop_assert!(t.range_contains(&w, EPS), "T(x) 不在值域?x={x:?}");
        }

        // 生成集合住在值域(練習 1 ↔ 2 交叉對帳):range_generating_set 的
        // 每支行向量都通過 range_contains —— aⱼ = T(eⱼ) 本來就是一個輸出。
        #[test]
        fn generating_set_lives_in_range(a in int_matrix_any_shape()) {
            let t = Transformation::new(a);
            for g in t.range_generating_set() {
                prop_assert!(t.range_contains(&g, EPS), "行向量不在自己張成的空間?");
            }
        }

        // onto ⟺ 全標準基底可達(練習 2 ↔ 3 交叉對帳):Range 蓋滿 ℝᵐ ⟺
        // 連 {e₁…e_m} 都收得進來(⟸ 因為子空間對張成封閉:裝得下整組
        // spanning set 就裝得下整個 ℝᵐ)。這條 law 正是練習 4 掃描策略的根據。
        #[test]
        fn onto_iff_every_standard_basis_vector_reachable(a in int_matrix_any_shape()) {
            let t = Transformation::new(a);
            let m = t.codomain_dim();
            let all_reachable = (0..m)
                .all(|i| t.range_contains(&Vector::standard(m, i).unwrap(), EPS));
            prop_assert_eq!(t.is_onto(EPS), all_reachable);
        }

        // 高瘦必不映成(Theorem 2.10 的維度限制半邊):rows > cols ⟹
        // rank ≤ cols < rows = m,n 支行向量張不滿更高維的 ℝᵐ。
        #[test]
        fn taller_than_wide_is_never_onto(a in tall_int_matrix()) {
            let t = Transformation::new(a);
            prop_assert!(!t.is_onto(EPS));
        }

        // IMT 接線(方陣):onto ⟺ 可逆 —— rank = n ⟺ RREF = Iₙ,
        // 兩個述詞走兩條獨立路徑(rank 計數 vs RREF 比對)必須給同一個答案。
        // 隨機方陣大多可逆、偶有奇異,兩個方向都會被踩到。
        #[test]
        fn square_transformation_onto_iff_invertible(a in int_matrix(3, 3)) {
            let t = Transformation::new(a.clone());
            prop_assert_eq!(t.is_onto(EPS), a.is_invertible(EPS));
        }

        // 掃描與判定對偶(練習 3 ↔ 4 交叉對帳):「找不到見證」與「映成」
        // 是同一件事的兩種問法 —— is_onto 數 rank、unreachable_vector 掃基底,
        // 兩條獨立路徑必須同答案。
        #[test]
        fn no_witness_iff_onto(a in int_matrix_any_shape()) {
            let t = Transformation::new(a);
            prop_assert_eq!(t.unreachable_vector(EPS).is_none(), t.is_onto(EPS));
        }

        // 見證的品質(題目驗收):回 Some(b) 的那支 b 必須真的不可達 ——
        // 用 System::solve 走第三條獨立路徑驗證 Ax = b 確實無解。
        #[test]
        fn witness_is_truly_unreachable(a in int_matrix_any_shape()) {
            let t = Transformation::new(a.clone());
            if let Some(b) = t.unreachable_vector(EPS) {
                prop_assert!(!t.range_contains(&b, EPS), "見證居然可達");
                let s = crate::System::new(a, b).unwrap(); // b ∈ ℝᵐ 依建構,維度必合
                prop_assert!(
                    matches!(s.solve(EPS), crate::Solution::Inconsistent),
                    "Ax = b 應無解"
                );
            }
        }

        // ---- 練習 5 的三條合起來是「真的是基底」的完整證明:住在 Range 的
        // 獨立集合、大小又恰為 dim(Range) = rank ⟹ 必為 Range 的基底
        // (張成自動跟上 —— 維度論證,不必另驗)。----

        // 基底大小 = rank(題目驗收):Col(A) 的維度就是 pivot 數。
        #[test]
        fn basis_size_equals_rank(a in int_matrix_any_shape()) {
            let t = Transformation::new(a.clone());
            prop_assert_eq!(t.range_basis(EPS).len(), a.rank(EPS));
        }

        // 基底必獨立(題目驗收):交給 independence 模組走獨立路徑驗證
        // (它數的是 rank,但作用在「抽出來的行」重排成的矩陣上)。
        #[test]
        fn basis_is_linearly_independent(a in int_matrix_any_shape()) {
            let t = Transformation::new(a);
            prop_assert!(crate::is_linearly_independent(EPS, &t.range_basis(EPS)));
        }

        // 基底住在值域(練習 2 ↔ 5 交叉對帳):每支基底向量都可達。
        #[test]
        fn basis_lives_in_range(a in int_matrix_any_shape()) {
            let t = Transformation::new(a);
            for b in t.range_basis(EPS) {
                prop_assert!(t.range_contains(&b, EPS), "基底向量不在值域?");
            }
        }

        // 基底 ⊆ 生成集合(練習 1 ↔ 5 交叉對帳):行對應定理取的是
        // **原始矩陣的行**,每支基底向量必須一字不差出現在生成集合裡 ——
        // 錯拿 RREF 的行(根本不在 Col(A))會在這裡穿幫。
        #[test]
        fn basis_is_subset_of_generating_set(a in int_matrix_any_shape()) {
            let t = Transformation::new(a);
            let gens = t.range_generating_set();
            for b in t.range_basis(EPS) {
                prop_assert!(
                    gens.iter().any(|g| g.equals(&b)),
                    "基底向量不是原始行?b={b:?}"
                );
            }
        }
    }
}
