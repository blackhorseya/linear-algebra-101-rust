//! 子空間(Subspace)及其性質 —— 三公理,與矩陣誘導的三個子空間。
//!
//! 筆記「子空間及其性質」章(單元 6-2,講義 4.1)。前五單元一路在操作
//! **整個** ℝⁿ;這一章開始問:ℝⁿ 裡哪些**子集合** W 自己就構成一個向量空間?
//! 答案是三條公理(子空間判準):
//!
//! 1. **0 ∈ W**(非空,且錨在原點 —— 仿射平移立刻出局);
//! 2. **加法封閉**:u, v ∈ W ⟹ u + v ∈ W;
//! 3. **純量乘法封閉**:u ∈ W, c ∈ ℝ ⟹ cu ∈ W。
//!
//! 而矩陣 A(m×n)天生誘導三個子空間,本章把它們逐一接上既有積木:
//!
//! | 子空間 | 定義 | 程式落點 |
//! |---|---|---|
//! | Col A ⊆ ℝᵐ | Span{A 的各行} | **已存在**:[`range_generating_set`](Transformation::range_generating_set)(5-3,Range(T) = Col A) |
//! | Null A ⊆ ℝⁿ | { v : Av = 0 } | 本模組 [`null_space_contains`](Transformation::null_space_contains)(Theorem 4.2) |
//! | Row A ⊆ ℝⁿ | Span{A 的各列} = Col Aᵀ | 本模組 [`row_space_generators`](Transformation::row_space_generators) |
//!
//! 「Span 必為子空間」(Theorem 4.1)與「Range(T) = Col A」這兩條,5-3 章
//! 已分別寫進依賴關係與 law(`image_is_always_reachable`)—— 本章**不重刻**,
//! 只補上 6-2 真正的新東西:公理本身的機器(掛在 [`PredicateSet<Vector>`]),
//! 與輸入端的 Null A。
//!
//! 「集合」沿用 [`PredicateSet`]:子空間幾乎都是無限集,列舉不可能,
//! 成員規則(述詞)是唯一能裝下它的容器 —— 筆記題目簽名裡的
//! `F: Fn(&Vector) -> bool` 正是 `PredicateSet::new` 收掉的東西。
//!
//! **隨機抽樣放哪裡?** 公開 API 一律是**確定性**的「單點見證」檢查
//! (給定 u, v, c 驗一次公理);「隨機掃一百組」是全稱命題的抽樣驗證,
//! 那是 proptest 的本職 —— 隨機性留在 `mod laws`,不進 library
//! (專案無 `rand`,慣例:定理寫成 laws)。抽樣驗證**只能反證、不能證明**:
//! 通過一百組樣本不代表是子空間,但一組反例就足以判死(第一象限對 c < 0
//! 不封閉,proptest 自己找得到)。

use crate::{PredicateSet, Transformation, Vector};

impl PredicateSet<Vector> {
    /// 公理 1:**0 ∈ W?** —— 檢查 ℝ^dim 的零向量是否通過成員規則。
    ///
    /// 為什麼要收 `dim`?`PredicateSet` 只是一條規則,自己不知道「住在哪個
    /// ℝⁿ」—— 零向量長什麼樣(幾個 0)得由呼叫端指定。這也是子空間定義的
    /// 第一句話:「W 是 **ℝⁿ 的**子集合」—— 母空間是判準的一部分。
    ///
    /// 這條公理單獨就能殺掉一大類集合:任何不過原點的直線 / 平面(仿射集)
    /// 在這裡出局,根本輪不到封閉性上場。
    ///
    /// 實作提示:零向量第一單元就刻好了([`Vector::new`] 即全 0),
    /// 剩下就是問一次 [`contains`](PredicateSet::contains) —— 一行。
    pub fn contains_zero(&self, dim: usize) -> bool {
        let zero = Vector::new(dim);
        self.contains(&zero)
    }

    /// 公理 2 + 3 的**單點見證**:在這一組 (u, v, c) 上,封閉性成立嗎?
    ///
    /// 邏輯形式是蘊涵(implication):
    ///
    /// > (u ∈ W 且 v ∈ W) ⟹ (u + v ∈ W 且 cu ∈ W)
    ///
    /// 三個語意決定,都寫進下方測試釘死:
    /// - **前提不成立 → 空虛真(vacuously true)**:u 或 v 根本不是成員,
    ///   這組樣本對封閉性**無話可說**,回 `true`(蘊涵的標準語意 ——
    ///   laws 拿任意隨機向量轟它時,非成員樣本不該誤報)。
    /// - **u + v 加不起來(維度不合)→ `false`**:兩個成員相加都不封閉
    ///   (連加法都沒定義),這集合不可能是任何單一 ℝⁿ 的子空間。
    /// - 純量封閉只驗 **cu** 一支:公理是 ∀u 形式,單點見證一個 u 就夠;
    ///   v 在這裡只服務加法公理。
    ///
    /// 實作提示:[`Vector::add`] 回 `Result`(維度檢查在它身上),
    /// `Err → false` 的收法與 [`range_contains`](Transformation::range_contains)
    /// 同款;[`Vector::scale`] 不會失敗。先寫前提的早退,再驗兩個結論。
    pub fn closed_at(&self, u: &Vector, v: &Vector, c: f64) -> bool {
        if !(self.contains(u) && self.contains(v)) {
            return true; // 前提不成立,空虛真
        }
        match u.add(v) {
            Ok(sum) => self.contains(&sum) && self.contains(&u.scale(c)),
            Err(_) => false, // 維度不合,加法沒定義 → 不封閉
        }
    }
}

impl Transformation {
    /// Null space 成員判定:**v ∈ Null A ⟺ Av ≈ 0**(Theorem 4.2 的集合)。
    ///
    /// Null A = { v ∈ ℝⁿ : Av = 0 } 是 ℝⁿ 的子空間 —— 證明就是三公理:
    /// A·0 = 0(公理 1)、Au = Av = 0 ⟹ A(u+v) = Au + Av = 0(公理 2)、
    /// A(cu) = c(Au) = 0(公理 3),全靠矩陣乘法的線性。下方 laws 把這段
    /// 紙上證明轉成可跑的隨機驗證。
    ///
    /// 與 [`range_contains`](Transformation::range_contains) 成對 ——
    /// 同一個 T 的兩端各切出一個子空間:
    ///
    /// | | 住在哪 | 成員判定 | 成本 |
    /// |---|---|---|---|
    /// | Range(T) = Col A | ℝᵐ(輸出端) | Ax = w **有沒有解**(消去法) | O(n³) |
    /// | Null(T) = Null A | ℝⁿ(輸入端) | Av **是不是** 0(代入驗算) | O(mn) |
    ///
    /// 值域成員要「解方程」,零空間成員只要「驗算」—— 因為 Null A 是用
    /// **等式**直接切出來的集合,Col A 是用**存在量詞**描述的集合。
    ///
    /// `v.rows() != n`(v 不住在 domain ℝⁿ)→ `false`:談不上成員
    /// (沿 `range_contains` 的 bool 述詞慣例)。判零門檻 `epsilon`
    /// 由呼叫端指定(浮點慣例)。
    ///
    /// 實作提示:[`apply`](Transformation::apply) 算 Av(維度檢查隨之繼承,
    /// `Err` 恰好就是要折成 `false` 的 case),判零是第一單元的
    /// [`Vector::is_approx_zero`] —— 兩支積木接起來一行收工。
    pub fn null_space_contains(&self, v: &Vector, epsilon: f64) -> bool {
        match self.apply(v) {
            Ok(image) => image.is_approx_zero(epsilon),
            Err(_) => false, // 維度不合,v 進不了 domain → 不成員
        }
    }

    /// Row space 的生成集合:**Row A = Span{A 的各列} = Col Aᵀ**,
    /// 故生成集 = A 的 m 支列向量(轉成 [`Vector`],各自住在 ℝⁿ)。
    ///
    /// 等式 Row A = Col Aᵀ 不是定理、是**換句話說**:Aᵀ 的第 i 行就是
    /// A 的第 i 列(轉置的定義)。這讓列空間零成本繼承行空間的全部機器 ——
    /// 之後維度章的大定理 dim(Row A) = dim(Col A) = rank 會用上這支積木。
    ///
    /// 與 [`range_generating_set`](Transformation::range_generating_set)
    /// 同款的純資料提取:不會失敗、不收 epsilon、**允許冗餘**
    /// (零列、相依列照收 —— 生成與基底是兩個概念)。
    ///
    /// 實作提示:兩條等價路線自選 ——
    /// (a) 字面翻譯定義:[`Matrix::transpose`](crate::Matrix::transpose) 後
    /// 抽行([`Matrix::column`](crate::Matrix::column)),「Row A = Col Aᵀ」
    /// 直接寫進程式;(b) 直接抽列:[`Matrix::row`](crate::Matrix::row) 回
    /// `&[f64]`,`Vector::from_vec(slice.to_vec())` 升格成向量,省一次轉置。
    /// 兩條都對 —— (a) 讓等式自我documenting,(b) 少拷貝一輪;
    /// 迭代器寫法鏡像 `range_generating_set`(迴圈不變式保證 unwrap 安全)。
    pub fn row_space_generators(&self) -> Vec<Vector> {
        (0..self.matrix().rows())
            .map(|i| Vector::from_vec(self.matrix().row(i).unwrap().to_vec()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Matrix, PredicateSet, Transformation, Vector};

    const EPS: f64 = 1e-9;

    /// xy 平面 { (x, y, 0) } ⊆ ℝ³ —— 教科書第一個「真」子空間範例。
    fn xy_plane() -> PredicateSet<Vector> {
        PredicateSet::new(|v: &Vector| v.rows() == 3 && v.entries()[2] == 0.0)
    }

    /// 第一象限 { (x, y) : x ≥ 0, y ≥ 0 } ⊆ ℝ² —— 教科書第一個反例:
    /// 過原點、加法也封閉,**只有**負純量殺得死它。
    fn first_quadrant() -> PredicateSet<Vector> {
        PredicateSet::new(|v: &Vector| v.rows() == 2 && v.entries().iter().all(|&x| x >= 0.0))
    }

    // ---- 公理 1:contains_zero ----

    #[test]
    fn contains_zero_accepts_xy_plane() {
        assert!(
            xy_plane().contains_zero(3),
            "(0,0,0) 的第三分量是 0,在平面上"
        );
    }

    #[test]
    fn contains_zero_rejects_shifted_line() {
        // y = 1 的水平線:仿射、不過原點 —— 公理 1 單獨判死,輪不到封閉性。
        let line = PredicateSet::new(|v: &Vector| v.rows() == 2 && v.entries()[1] == 1.0);
        assert!(!line.contains_zero(2), "(0,0) 不在 y = 1 上");
    }

    #[test]
    fn contains_zero_accepts_first_quadrant() {
        // 反例集合也通過公理 1 —— 三公理是合取,殺死它的是別條(見 closed_at)。
        assert!(first_quadrant().contains_zero(2), "0 ≥ 0,原點在象限裡");
    }

    // ---- 公理 2 + 3:closed_at ----

    #[test]
    fn closed_at_first_quadrant_fails_on_negative_scalar() {
        let u = Vector::from_vec(vec![1.0, 2.0]);
        let v = Vector::from_vec(vec![2.0, 1.0]);
        assert!(
            !first_quadrant().closed_at(&u, &v, -1.0),
            "-1·(1,2) = (-1,-2) 衝出象限 —— 純量封閉破功"
        );
    }

    #[test]
    fn closed_at_first_quadrant_passes_on_positive_scalar() {
        // 單點通過 ≠ 是子空間 —— 抽樣只能反證。這組 (u, v, c) 恰好沒踩到痛點。
        let u = Vector::from_vec(vec![1.0, 2.0]);
        let v = Vector::from_vec(vec![2.0, 1.0]);
        assert!(first_quadrant().closed_at(&u, &v, 2.0));
    }

    #[test]
    fn closed_at_is_vacuously_true_when_premise_fails() {
        // u 不是成員 → 這組樣本對封閉性無話可說(蘊涵前提為假)。
        let u = Vector::from_vec(vec![-1.0, 0.0]);
        let v = Vector::from_vec(vec![1.0, 1.0]);
        assert!(
            first_quadrant().closed_at(&u, &v, -5.0),
            "前提不成立,空虛真 —— 不該誤報封閉性破功"
        );
    }

    #[test]
    fn closed_at_rejects_dimension_mismatch() {
        // 全集裝得下任何維度的向量,但 ℝ² + ℝ³ 連加法都沒定義 —— 不封閉。
        let anything = PredicateSet::<Vector>::universal();
        let u = Vector::from_vec(vec![1.0, 2.0]);
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(!anything.closed_at(&u, &v, 1.0));
    }

    // ---- Null space ----

    #[test]
    fn null_space_contains_accepts_kernel_vector() {
        // A 的兩列都與 (1, -2, 1) 正交:1-4+3 = 0、4-10+6 = 0。
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]));
        let v = Vector::from_vec(vec![1.0, -2.0, 1.0]);
        assert!(t.null_space_contains(&v, EPS));
    }

    #[test]
    fn null_space_contains_rejects_non_kernel_vector() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]));
        let v = Vector::from_vec(vec![1.0, 0.0, 0.0]);
        assert!(!t.null_space_contains(&v, EPS), "A·e₁ = (1,4) ≠ 0");
    }

    #[test]
    fn null_space_contains_zero_vector() {
        // A·0 = 0 恆成立 —— 公理 1 在 Null A 上的具體化(題目驗收條件)。
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]));
        assert!(t.null_space_contains(&Vector::new(3), EPS));
    }

    #[test]
    fn null_space_contains_rejects_dimension_mismatch() {
        // v ∈ ℝ² 進不了 domain ℝ³ —— 談不上成員,false(沿 range_contains 慣例)。
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]));
        assert!(!t.null_space_contains(&Vector::from_vec(vec![1.0, 2.0]), EPS));
    }

    // ---- Row space ----

    #[test]
    fn row_space_generators_of_textbook_example() {
        // 題目原例:3×2 矩陣 → 3 支生成元素、各住在 ℝ²;
        // 第二列是第一列的 2 倍、第三列是零列 —— 冗餘照收(生成 ≠ 基底)。
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![0.0, 0.0],
        ]));
        let gens = t.row_space_generators();
        assert_eq!(gens.len(), 3);
        assert!(gens[0].equals(&Vector::from_vec(vec![1.0, 2.0])));
        assert!(gens[1].equals(&Vector::from_vec(vec![2.0, 4.0])));
        assert!(gens[2].equals(&Vector::from_vec(vec![0.0, 0.0])));
    }
}

/// 教材定理的隨機驗證(「for all」形式的代數律)—— 題目 1 要的「隨機抽樣
/// 驗證器」本體在這裡:proptest 產樣本,公開 API 只做單點檢查。
#[cfg(test)]
mod laws {
    use crate::{Matrix, PredicateSet, Transformation, Vector};
    use proptest::prelude::*;

    /// 判零門檻(整數輸入的運算完全精確,殘差為 0,遠低於此)。
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

    /// 隨機形狀(1..=4 × 1..=4)的整數矩陣。
    fn int_matrix_any_shape() -> impl Strategy<Value = Matrix> {
        (1usize..=4, 1usize..=4).prop_flat_map(|(rows, cols)| int_matrix(rows, cols))
    }

    /// xy 平面的隨機成員:(x, y, 0),x、y 為小整數 —— **依建構**就在平面上
    /// (與 tall_int_matrix 的「依建構成立」同一招,免 prop_assume 丟樣本)。
    fn xy_plane_member() -> impl Strategy<Value = Vector> {
        (-10i64..=10, -10i64..=10)
            .prop_map(|(x, y)| Vector::from_vec(vec![x as f64, y as f64, 0.0]))
    }

    /// 第一象限的**嚴格正**成員:分量取 [1, 10] —— 排除 0 向量,
    /// 保證任何負純量都把它推出象限(0 向量是唯一推不出去的點)。
    fn strictly_positive_vector() -> impl Strategy<Value = Vector> {
        prop::collection::vec(1i64..=10, 2)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()))
    }

    /// 核(kernel)植入法:建構 (A, v) 使 **Av = 0 精確成立**,全程整數。
    ///
    /// 隨機找 Null A 的非零成員是大海撈針,反過來「先射箭再畫靶」:
    /// 1. 先抽 v = (v₁, …, vₙ₋₁, **1**) —— 末分量釘 1,保證 v ≠ 0;
    /// 2. A 的每一列抽前 n-1 個分量 r₁…rₙ₋₁,**末分量算出來**:
    ///    rₙ := -(r₁v₁ + ⋯ + rₙ₋₁vₙ₋₁),於是 r·v = 0 依建構成立。
    ///
    /// 末分量釘 1 是讓 rₙ 保持整數的關鍵(除法會出有理數)—— 整數策略
    /// 配精確判零,殘差恰為 0.0。
    fn kernel_pair() -> impl Strategy<Value = (Matrix, Vector)> {
        (2usize..=4, 1usize..=3).prop_flat_map(|(n, m)| {
            let v = prop::collection::vec(-5i64..=5, n - 1);
            let rows = prop::collection::vec(prop::collection::vec(-5i64..=5, n - 1), m);
            (v, rows).prop_map(|(mut v, rows)| {
                v.push(1);
                let a = Matrix::from_rows(
                    rows.into_iter()
                        .map(|r| {
                            let last = -r.iter().zip(&v).map(|(ri, vi)| ri * vi).sum::<i64>();
                            r.into_iter().chain([last]).map(|x| x as f64).collect()
                        })
                        .collect(),
                );
                let v = Vector::from_vec(v.into_iter().map(|x| x as f64).collect());
                (a, v)
            })
        })
    }

    proptest! {
        // 子空間範例律:xy 平面對**任意**成員 u, v 與任意純量 c 通過三公理
        // (Theorem 4.1 的特例:xy 平面 = Span{e₁, e₂})。整數成員的
        // u+v、cu 第三分量精確為 0.0 —— 整數策略配精確比較。
        #[test]
        fn xy_plane_passes_subspace_axioms(
            u in xy_plane_member(), v in xy_plane_member(), c in -10i64..=10,
        ) {
            let plane = PredicateSet::new(|w: &Vector| w.rows() == 3 && w.entries()[2] == 0.0);
            prop_assert!(plane.contains_zero(3), "公理 1");
            prop_assert!(plane.closed_at(&u, &v, c as f64), "公理 2+3 在 ({u:?}, {v:?}, {c}) 破功");
        }

        // 反例律:第一象限對**每一個**嚴格負純量、**每一個**非零成員都破功 ——
        // 題目的期望輸出 false 不是僥倖踩中,是整族樣本全滅。
        #[test]
        fn first_quadrant_fails_every_negative_scaling(
            u in strictly_positive_vector(), c in -10i64..=-1,
        ) {
            let quadrant = PredicateSet::new(|w: &Vector| {
                w.rows() == 2 && w.entries().iter().all(|&x| x >= 0.0)
            });
            prop_assert!(
                !quadrant.closed_at(&u, &u, c as f64),
                "cu 兩分量皆負,必出象限"
            );
        }

        // Theorem 4.2 公理 1:**每一個** A 的零空間都含零向量(A·0 = 0)。
        #[test]
        fn zero_vector_lies_in_every_null_space(a in int_matrix_any_shape()) {
            let n = a.cols();
            let t = Transformation::new(a);
            prop_assert!(t.null_space_contains(&Vector::new(n), EPS));
        }

        // Theorem 4.2 公理 2+3:核成員的線性組合仍在核裡 ——
        // Av = 0 ⟹ A(c₁v) = c₁(Av) = 0,A(c₁v + v) = c₁(Av) + Av = 0。
        // 成員由 kernel_pair 植入(依建構 Av = 0),純量整數 → 全程精確。
        #[test]
        fn null_space_closed_under_linear_combination(
            (a, v) in kernel_pair(), c1 in -5i64..=5, c2 in -5i64..=5,
        ) {
            let t = Transformation::new(a);
            prop_assert!(t.null_space_contains(&v, EPS), "植入的 v 必在核裡");
            let u = v.scale(c1 as f64);
            prop_assert!(t.null_space_contains(&u, EPS), "公理 3:c₁v 仍在核裡");
            let sum = u.add(&v).unwrap(); // 同住 ℝⁿ,unwrap 安全
            prop_assert!(t.null_space_contains(&sum, EPS), "公理 2:c₁v + v 仍在核裡");
            prop_assert!(t.null_space_contains(&v.scale(c2 as f64), EPS));
        }

        // 章節合龍:Null A 包成 PredicateSet,用題目 1 的公理機器驗題目 2 的
        // 集合 —— Theorem 4.2「Null A 是子空間」一字不差變成可跑的命題。
        #[test]
        fn null_space_as_predicate_set_passes_axioms(
            (a, v) in kernel_pair(), c1 in -5i64..=5, c in -5i64..=5,
        ) {
            let n = a.cols();
            let t = Transformation::new(a);
            let null_a = PredicateSet::new(move |x: &Vector| t.null_space_contains(x, EPS));
            prop_assert!(null_a.contains_zero(n), "公理 1");
            // u := c₁v 依線性必為成員 → 前提成立,closed_at 非空虛地驗 2+3。
            prop_assert!(null_a.closed_at(&v.scale(c1 as f64), &v, c as f64), "公理 2+3");
        }

        // 形狀律(題目驗收條件):m×n 矩陣的列空間生成集恆有 m 支、
        // 每支住在 ℝⁿ —— 與行空間(n 支、住 ℝᵐ)恰好鏡像。
        #[test]
        fn row_space_generators_shape(a in int_matrix_any_shape()) {
            let (m, n) = (a.rows(), a.cols());
            let gens = Transformation::new(a).row_space_generators();
            prop_assert_eq!(gens.len(), m, "一列一支生成元素");
            for g in &gens {
                prop_assert_eq!(g.rows(), n, "生成元素住在 ℝⁿ");
            }
        }

        // Row A = Col Aᵀ(本章核心等式):A 的每支列生成元素都通過
        // **Aᵀ 的值域**成員判定 —— 列空間的成員資格由 5-3 的行空間機器背書,
        // 兩條獨立路徑(資料提取 vs 消去法)會合。
        #[test]
        fn row_generators_live_in_column_space_of_transpose(a in int_matrix_any_shape()) {
            let t_transpose = Transformation::new(a.transpose());
            for g in Transformation::new(a).row_space_generators() {
                prop_assert!(
                    t_transpose.range_contains(&g, EPS),
                    "列向量不在 Col Aᵀ?Row A = Col Aᵀ 破功"
                );
            }
        }

        // 對偶律:row_space_generators(Aᵀ) 與 range_generating_set(A) 逐支相等
        // —— 「Aᵀ 的列」就是「A 的行」,兩支 API 從兩端讀同一份資料(精確比較)。
        #[test]
        fn transpose_swaps_row_and_column_generators(a in int_matrix_any_shape()) {
            let cols = Transformation::new(a.clone()).range_generating_set();
            let rows_of_transpose = Transformation::new(a.transpose()).row_space_generators();
            prop_assert_eq!(cols.len(), rows_of_transpose.len());
            for (c, r) in cols.iter().zip(&rows_of_transpose) {
                prop_assert!(c.equals(r));
            }
        }
    }
}
