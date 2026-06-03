//! `Span` —— span{v₁,…,vₖ} = `{ c₁v₁ + … + cₖvₖ : cᵢ ∈ ℝ }`,一組有限生成向量
//! 的**所有線性組合**所成的集合。
//!
//! 對應原始 Go 專案 commit `97fd4c1`
//! (`feat(span): add Span as a column space with membership, dimension, equality`)。
//!
//! 幾何上,span 是生成向量掃出、**通過原點**的「平直集」(flat):一個獨立方向是
//! 一條線,兩個是一個平面,以此類推。它的定義性恆等式是
//! span{v₁,…,vₖ} = Col(A) —— 以這些生成向量為**行**的矩陣 A 的 **column space**。
//!
//! 所以 `Span` 存的是矩陣 A 本身,而不只是一個成員判定函式;握有 A 才能用同一個
//! 物件回答三個不同的問題:
//!
//! - [`contains`](Span::contains) —— x 在 span 裡嗎? → A·c = x 是否一致(`is_consistent`)
//! - [`dimension`](Span::dimension) —— span 多大? → `rank(A)`
//! - [`equals`](Span::equals) —— 跟另一個是同一個平直集嗎? → 生成向量的相互包含
//!
//! 純粹的成員述詞([`PredicateSet`](crate::PredicateSet))只能答第一個;另外兩個
//! 都得有 A 在手。需要集合代數(∪ / ∩ / …)那套視角時,用
//! [`as_predicate`](Span::as_predicate) 降階回述詞集合。

use crate::{Matrix, PredicateSet, Solution, System, Vector};

/// span{generators…},以矩陣 A(生成向量為行)為內部表示的 column space。
///
/// `Clone` 是為了 [`as_predicate`](Span::as_predicate):回傳的述詞 closure 必須是
/// `'static`(不能借用 `self`),只好把整個 `Span` 複製一份 `move` 進去。
#[derive(Debug, Clone)]
pub struct Span {
    /// 原始生成向量,[`equals`](Span::equals) 要逐個拿去做相互包含檢查時需要它。
    generators: Vec<Vector>,
    /// 生成向量排成「行」的矩陣 A。空 span `{0}` 沒有矩陣 → `None`。
    matrix: Option<Matrix>,
    /// 一致性 / rank 判定用的零容差(RREF 化簡會帶進捨入誤差)。
    epsilon: f64,
}

impl Span {
    /// 建立 span{generators}。沒有任何生成向量時,它是**平凡子空間** `{0}`(「什麼都不
    /// 加」的空線性組合就是零向量),`matrix` 留為 `None`,環繞維度延到
    /// [`contains`](Span::contains) 時由查詢向量決定。假設各生成向量同維度。
    pub fn new(epsilon: f64, generators: Vec<Vector>) -> Span {
        if generators.is_empty() {
            return Span {
                generators,
                matrix: None, // matrix 留 None → {0}
                epsilon,
            };
        }

        // 要把生成向量擺成矩陣的「行」,但 `Matrix::from_rows` 是按「列」吃資料,而
        // 跨模組又碰不到 private `data` 逐格填(Go 是同 package 才那樣做)。所以先把
        // 每個生成向量當成一「列」建出 k×dim 矩陣,再 `transpose()` 翻成 dim×k ——
        // 轉置後第 j 行恰好就是第 j 個生成向量。
        let rows: Vec<Vec<f64>> = generators.iter().map(|g| g.entries().to_vec()).collect();
        let matrix = Matrix::from_rows(rows).transpose();

        Span {
            generators,
            matrix: Some(matrix),
            epsilon,
        }
    }

    /// x ∈ span{generators} 嗎?這就是成員規則:「x 是某個組合 c₁v₁ + … + cₖvₖ」
    /// 精確等價於「系統 A·c = x 有解 c」—— 因為在 column view 裡,A·c **就是**那個
    /// 加權和。於是成員資格 reduce 成一致性,正是幾何通往代數的橋。
    pub fn contains(&self, x: &Vector) -> bool {
        // 換你寫:`match &self.matrix` 分三條路 ——
        //   1. None(空 span {0}):x 是不是零向量?(`x.is_zero()`)
        //   2. Some(a):成員資格就是一致性。用 `System::new(a.clone(), x.clone())` 建
        //      系統,`Ok(system)` 時回 `system.is_consistent(self.epsilon)`。
        //   3. `System::new` 回 `Err`(維度不合):x 不在對的空間 → `false`。
        match &self.matrix {
            None => x.is_zero(), // {0} 只有零向量
            Some(a) => match System::new(a.clone(), x.clone()) {
                Ok(system) => system.is_consistent(self.epsilon),
                Err(_) => false, // 維度不合 → x 不在對的空間,自然不在 span 裡
            },
        }
    }

    /// 建構式成員判定:[`contains`](Span::contains) 給的是赤裸的 yes/no,這支交回那個「yes」
    /// 背後的**見證** —— 一組權重 c,使 `x = c₁v₁ + … + cₖvₖ`(本 span 的 generator 依 c
    /// 加權),若存在的話。
    ///
    /// 它刻意重用 [`Solution`]:「x 在 span{v₁…vₖ} 裡」**就是**「A·c=x 一致」(column view,
    /// 與 `contains` 過的同一座橋),所以「用哪些權重?」字面上就是「解那個系統求 c」。用
    /// `Solution` 當載體,讓這個恆等式顯示在型別裡。三結局透過成員資格重讀:
    /// - [`Inconsistent`](Solution::Inconsistent) —— x ∉ span,沒有權重能組出它(`contains` 為 false)。
    /// - [`Unique`](Solution::Unique) —— x ∈ span **且** generator 獨立,權重是 x 在此 span 的**唯一座標**。
    /// - [`Infinite`](Solution::Infinite) —— x ∈ span 但 generator 相依,x 有多種表示、無正則者。
    ///
    /// `combination(x)` 非 `Inconsistent` ⟺ `contains(x)` 為 true —— 建構式與布林判定由建構天然一致。
    pub fn combination(&self, x: &Vector) -> Solution {
        match &self.matrix {
            None => {
                if x.is_zero() {
                    Solution::Unique(Vector::new(0)) // {0} 只有零向量,它的權重是空和
                } else {
                    Solution::Inconsistent // {0} 以外的向量不在裡面
                }
            }
            Some(a) => match System::new(a.clone(), x.clone()) {
                Ok(system) => system.solve(self.epsilon), // 系統有解,交給 solve 判定三結局
                Err(_) => Solution::Inconsistent, // 維度不合 → x 不在對的空間,自然不在 span 裡
            },
        }
    }

    /// span 的維度:生成向量中**獨立方向**的數目 —— 可能少於生成向量的個數,因為有些
    /// 是多餘的(例如 span{(1,1),(2,2)} 是一條線,維度 1 而非 2)。
    ///
    /// 一個 span 的維度恰好是 `rank(A)`:A 的獨立行數,也就是生成向量裡的獨立方向數。
    pub fn dimension(&self) -> usize {
        match &self.matrix {
            None => 0, // {0} 的維度為 0
            Some(a) => a.rank(self.epsilon),
        }
    }

    /// 哪些 generator 可以**移除而不縮小 span** —— 消去法判定出的自由行(free columns)。
    /// 移除它們剩下 pivot 行:一組仍張出相同空間的極大獨立子集(這個 span 的一組基底)。
    ///
    /// 哪些 generator「可移除」**不唯一**(如 {e₀,e₁,e₀+e₁} 中任一個都是另兩個的組合),
    /// 這回傳消去法做的**特定**選擇;但其長度恆等於冗餘數(= nullity)。空 span 回空 vec。
    ///
    /// (Go 的 `RemovableColumns` 同 package 直接碰 `Span` 的 private matrix 取 free columns;
    /// Rust 跨模組沒有 friend access,所以在 `Span` 開這個 accessor 給 `independence` 用。)
    pub fn free_columns(&self) -> Vec<usize> {
        match &self.matrix {
            None => vec![],
            Some(a) => a.free_columns(self.epsilon),
        }
    }

    /// 內部矩陣 A(生成向量為行)的唯讀借出;空 span `{0}` 沒有矩陣,回 `None`。給
    /// `independence` 的 RREF 路徑用(同 `free_columns`,Rust 跨模組無 friend access,
    /// 只能在 `Span` 開 accessor)。
    pub fn column_matrix(&self) -> Option<&Matrix> {
        self.matrix.as_ref()
    }

    /// self 與 other 是否為**同一個** span(同一個子空間),不管它們各自用哪組生成向量
    /// 描述 —— span{e₀,e₁} 等於 span{(1,1,0),(1,-1,0)},因為兩者都是 xy 平面。
    ///
    /// span 是無限集合,這卻**可判定**、不必探測無限多個點,因為 span 是**有限生成**的:
    /// span{A} ⊆ span{B} ⟺ A 的每個生成向量都落在 span{B} 裡(生成向量都進去了,它們
    /// 的所有線性組合自然也進去了)。於是集合相等 —— 跟 `Set::equals` 同一個「⊆ 反對稱」
    /// 套路 —— reduce 成一個有限檢查。
    pub fn equals(&self, other: &Span) -> bool {
        // 換你寫:相互包含。self ⊆ other ⟺ self 的**每個** generator 都被 other 包含;
        // 反向亦然,兩者都成立才相等。提示:`self.generators` / `other.generators` 是
        // 私有欄位,但同模組存取得到;配 `.iter().all(|g| ….contains(g))` 與 `&&`。
        // 只查 generator 就夠,因為 span 對線性組合封閉 —— generator 都在,它們張出的
        // 一切就都在。
        self.generators.iter().all(|g| other.contains(g))
            && other.generators.iter().all(|g| self.contains(g))
    }

    /// Theorem 1.6 的條件 (a)+(c) 併成一句:這個 span 是不是**填滿整個** ℝ^ambient_dim?
    /// 當且僅當它的獨立方向數(= [`dimension`](Span::dimension) = `rank(A)`)等於環境空間
    /// 的軸數時成立;少一個方向,就有某個方向永遠碰不到。
    ///
    /// **易錯點**:`ambient_dim` 比的是 A 的**列數 m**(目標空間 ℝᵐ 的軸數),不是行數 n
    /// —— Theorem 1.6 的 (c) 是 full **ROW** rank。高瘦矩陣(m > n)的行再多也張不滿 ℝᵐ。
    pub fn spans_all(&self, ambient_dim: usize) -> bool {
        // 換你寫:一行。span 填滿 ℝ^ambient_dim ⟺ 它的獨立方向數(`self.dimension()`)
        // 恰好等於環境維度。關鍵:跟 `ambient_dim`(= A 的**列數** m)比,而不是行數 ——
        // 對照上方 doc 的「易錯點」。
        self.dimension() == ambient_dim
    }

    /// 從矩陣 A 的**各行**建出 span,即 column space Col(A) —— Theorem 1.6 的主角:A 的行
    /// 張滿 ℝᵐ ⟺ 這個 span 填滿整個空間([`spans_all`](Span::spans_all)`(m)`,m 是 A 列數)。
    pub fn from_columns(epsilon: f64, a: &Matrix) -> Span {
        let columns: Vec<Vector> = (0..a.cols())
            .map(|j| a.column(j).expect("j 必落在 [0, cols) 內,不可能越界"))
            .collect();
        Span::new(epsilon, columns)
    }

    /// 把 span 暴露成 [`PredicateSet<Vector>`](crate::PredicateSet),讓布林集合代數
    /// (Union / Intersection / Complement / …)也能套用到 span 上。述詞就是成員規則
    /// [`contains`](Span::contains) 本身。
    ///
    /// 對照 Go 的 `return s.Contains`(method value 自動綁定 receiver):Rust 的述詞
    /// closure 要 `'static`,不能借用 `self`,所以得把整個 `Span` 複製一份 `move` 進
    /// closure —— 等於把「method value 背後藏著一個對 receiver 的引用」這件事顯式化了。
    pub fn as_predicate(&self) -> PredicateSet<Vector> {
        let span = self.clone();
        PredicateSet::new(move |x| span.contains(x))
    }
}

/// point 是否落在以 direction 為方向、**通過原點**的那條線上,即 point ∈ span{direction}。
/// direction 為零向量時「線」退化成單點 `{0}`,這仍然會答對。
pub fn on_line(point: &Vector, direction: &Vector, epsilon: f64) -> bool {
    Span::new(epsilon, vec![direction.clone()]).contains(point)
}

/// point 是否落在由 u、v 張出、**通過原點**的平面上,即 point ∈ span{u, v}。
/// u 與 v 平行時 span 只是一條線,這仍然會答對(問的就變成「點在那條線上嗎」)。
pub fn on_plane(point: &Vector, u: &Vector, v: &Vector, epsilon: f64) -> bool {
    Span::new(epsilon, vec![u.clone(), v.clone()]).contains(point)
}

/// 回傳仿射平直集 base + span{directions} = `{ base + c₁d₁ + … + cₖdₖ }`,以
/// [`PredicateSet<Vector>`](crate::PredicateSet) 表示 —— 那條**不必通過原點**的線或
/// 平面。base 是已知落在該平直集上的任一點,directions 張出它。
///
/// 它 **reduce 回通過原點的情形**:x 落在 base + span{D} 上,恰當位移 x − base 落在
/// span{D} 裡。減掉基點等於把整個平直集滑回原點,在那裡交給 `Span` 判定成員資格。
///
///   x ∈ base + span{D}  ⟺  (x − base) ∈ span{D}。
pub fn affine_span(epsilon: f64, base: Vector, directions: Vec<Vector>) -> PredicateSet<Vector> {
    let through = Span::new(epsilon, directions);
    // x − base,寫成 x + (−1)·base,因為 Vector 沒有 Sub。預先算好 −base。
    let neg_base = base.scale(-1.0);
    PredicateSet::new(move |x: &Vector| match x.add(&neg_base) {
        Ok(diff) => through.contains(&diff),
        Err(_) => false, // 維度不合 → x 不在對的空間,自然不在這個平直集上
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPAN_EPS: f64 = 1e-9;

    /// 從字面值建向量的測試輔助(`Vector::from_vec` 的縮寫)。
    fn v(data: Vec<f64>) -> Vector {
        Vector::from_vec(data)
    }

    #[test]
    fn on_line_through_origin() {
        // 線 = { t·d : t ∈ ℝ },點在線上 ⟺ 它是 d 的純量倍數。
        let d = v(vec![1.0, 2.0, 3.0]); // ℝ³ 的方向
        assert!(
            on_line(&v(vec![2.0, 4.0, 6.0]), &d, SPAN_EPS),
            "t=2 的倍數應在線上"
        );
        assert!(
            on_line(&v(vec![-1.0, -2.0, -3.0]), &d, SPAN_EPS),
            "t=-1 的倍數應在線上"
        );
        assert!(
            on_line(&v(vec![0.0, 0.0, 0.0]), &d, SPAN_EPS),
            "原點在每條線上"
        );
        assert!(
            !on_line(&v(vec![1.0, 2.0, 4.0]), &d, SPAN_EPS),
            "不成倍數,不在線上"
        );
        assert!(
            !on_line(&v(vec![1.0, 2.0, 3.5]), &d, SPAN_EPS),
            "差一個座標,不在線上"
        );
    }

    #[test]
    fn on_plane_through_origin() {
        // ℝ³ 的 xy 平面 span{e₀,e₁}:點在上面 ⟺ z 座標為 0。
        let u = v(vec![1.0, 0.0, 0.0]); // e₀
        let w = v(vec![0.0, 1.0, 0.0]); // e₁
        assert!(
            on_plane(&v(vec![3.0, 5.0, 0.0]), &u, &w, SPAN_EPS),
            "在 xy 平面上"
        );
        assert!(
            on_plane(&v(vec![1.0, 0.0, 0.0]), &u, &w, SPAN_EPS),
            "基底向量在平面上"
        );
        assert!(
            on_plane(&v(vec![0.0, 0.0, 0.0]), &u, &w, SPAN_EPS),
            "原點在平面上"
        );
        assert!(
            !on_plane(&v(vec![3.0, 5.0, 1.0]), &u, &w, SPAN_EPS),
            "抬離平面"
        );
        assert!(
            !on_plane(&v(vec![0.0, 0.0, 2.0]), &u, &w, SPAN_EPS),
            "純 z 不在平面上"
        );
    }

    #[test]
    fn span_contains_dependent_generators() {
        // 成員資格追蹤幾何、不看生成向量個數:span{(1,1),(2,2)} 仍只是一條線,
        // 因為 (2,2) 沒帶來新方向。給了兩個向量,線外的點照樣被拒。
        let line = Span::new(SPAN_EPS, vec![v(vec![1.0, 1.0]), v(vec![2.0, 2.0])]);
        assert!(
            line.contains(&v(vec![3.0, 3.0])),
            "(3,3) 應在線 span{{(1,1),(2,2)}} 上"
        );
        assert!(!line.contains(&v(vec![3.0, 4.0])), "(3,4) 不應在該線上");
    }

    #[test]
    fn span_empty_is_trivial_subspace() {
        // span{} = {0}:沒有生成向量,唯一成員是零向量(空線性組合)。
        let trivial = Span::new(SPAN_EPS, vec![]);
        assert!(
            trivial.contains(&v(vec![0.0, 0.0, 0.0])),
            "span{{}} 應含零向量"
        );
        assert!(
            !trivial.contains(&v(vec![0.0, 0.0, 1.0])),
            "span{{}} 只含零向量"
        );
    }

    #[test]
    fn span_dimension_is_rank() {
        // 維度 = rank(A):獨立方向數,可能少於給的生成向量個數。
        assert_eq!(Span::new(SPAN_EPS, vec![]).dimension(), 0, "空 span 維度 0");
        assert_eq!(
            Span::new(SPAN_EPS, vec![v(vec![1.0, 2.0, 3.0])]).dimension(),
            1,
            "單一方向是一條線"
        );
        assert_eq!(
            Span::new(
                SPAN_EPS,
                vec![v(vec![1.0, 0.0, 0.0]), v(vec![0.0, 1.0, 0.0])]
            )
            .dimension(),
            2,
            "兩個獨立方向是一個平面"
        );
        assert_eq!(
            Span::new(SPAN_EPS, vec![v(vec![1.0, 1.0]), v(vec![2.0, 2.0])]).dimension(),
            1,
            "相依生成向量塌縮:兩個向量但只有 1 個獨立方向"
        );
        assert_eq!(
            Span::new(
                SPAN_EPS,
                vec![
                    v(vec![1.0, 0.0, 0.0]),
                    v(vec![0.0, 1.0, 0.0]),
                    v(vec![0.0, 0.0, 1.0])
                ]
            )
            .dimension(),
            3,
            "三軸張出整個 ℝ³"
        );
    }

    #[test]
    fn span_equals_is_same_subspace() {
        // 相等 ⟺ 同一個子空間,與用哪組生成向量描述無關;且必須對稱。
        let cases: Vec<(Span, Span, bool)> = vec![
            // 同一條線、不同尺度:span{(1,0,0)} = span{(2,0,0)}
            (
                Span::new(SPAN_EPS, vec![v(vec![1.0, 0.0, 0.0])]),
                Span::new(SPAN_EPS, vec![v(vec![2.0, 0.0, 0.0])]),
                true,
            ),
            // 生成向量順序與子空間無關
            (
                Span::new(
                    SPAN_EPS,
                    vec![v(vec![1.0, 0.0, 0.0]), v(vec![0.0, 1.0, 0.0])],
                ),
                Span::new(
                    SPAN_EPS,
                    vec![v(vec![0.0, 1.0, 0.0]), v(vec![1.0, 0.0, 0.0])],
                ),
                true,
            ),
            // 重點:同一個 xy 平面,用完全不同的基底描述
            (
                Span::new(
                    SPAN_EPS,
                    vec![v(vec![1.0, 0.0, 0.0]), v(vec![0.0, 1.0, 0.0])],
                ),
                Span::new(
                    SPAN_EPS,
                    vec![v(vec![1.0, 1.0, 0.0]), v(vec![1.0, -1.0, 0.0])],
                ),
                true,
            ),
            // 線是平面的真子集,兩者不相等
            (
                Span::new(SPAN_EPS, vec![v(vec![1.0, 0.0, 0.0])]),
                Span::new(
                    SPAN_EPS,
                    vec![v(vec![1.0, 0.0, 0.0]), v(vec![0.0, 1.0, 0.0])],
                ),
                false,
            ),
            // 多餘的生成向量不改變子空間
            (
                Span::new(SPAN_EPS, vec![v(vec![1.0, 1.0])]),
                Span::new(SPAN_EPS, vec![v(vec![1.0, 1.0]), v(vec![2.0, 2.0])]),
                true,
            ),
        ];

        for (a, b, want) in cases {
            assert_eq!(a.equals(&b), want, "equals 結果不符");
            assert_eq!(b.equals(&a), want, "equals 必須對稱");
        }
    }

    #[test]
    fn affine_line_not_through_origin() {
        // ℝ² 的水平線 y = 1,寫成 base (0,1) + span{(1,0)}。關鍵:原點**不**在上面 ——
        // 這正是仿射平直集與子空間的分野。
        let line = affine_span(SPAN_EPS, v(vec![0.0, 1.0]), vec![v(vec![1.0, 0.0])]);
        assert!(line.contains(&v(vec![0.0, 1.0])), "基點自己在線上");
        assert!(line.contains(&v(vec![5.0, 1.0])), "另一個 y=1 的點在線上");
        assert!(!line.contains(&v(vec![0.0, 0.0])), "原點不在這條仿射線上");
        assert!(!line.contains(&v(vec![5.0, 2.0])), "y 不對");
    }

    #[test]
    fn affine_plane_not_through_origin() {
        // ℝ³ 的平面 z = 1,寫成 base (0,0,1) + span{e₀,e₁}。
        let plane = affine_span(
            SPAN_EPS,
            v(vec![0.0, 0.0, 1.0]),
            vec![v(vec![1.0, 0.0, 0.0]), v(vec![0.0, 1.0, 0.0])],
        );
        assert!(
            plane.contains(&v(vec![3.0, 4.0, 1.0])),
            "(3,4,1) 在平面 z=1 上"
        );
        assert!(plane.contains(&v(vec![0.0, 0.0, 1.0])), "基點在平面上");
        assert!(
            !plane.contains(&v(vec![3.0, 4.0, 0.0])),
            "(3,4,0) 不在平面 z=1 上"
        );
    }

    #[test]
    fn span_line_is_subset_of_plane() {
        // 把幾何接回集合論:線 span{e₀} 坐落在平面 span{e₀,e₁} 之內。線上每個取樣點
        // 都必須也在平面裡。
        let line = Span::new(SPAN_EPS, vec![v(vec![1.0, 0.0, 0.0])]);
        let plane = Span::new(
            SPAN_EPS,
            vec![v(vec![1.0, 0.0, 0.0]), v(vec![0.0, 1.0, 0.0])],
        );
        for t in [-3.0, 0.0, 1.0, 7.5] {
            let p = v(vec![t, 0.0, 0.0]); // 線上一點
            // 「在線上 ⟹ 在平面上」,寫成等價的 ¬在線上 ∨ 在平面上。
            assert!(
                !line.contains(&p) || plane.contains(&p),
                "線上的點 {:?} 也必須在平面上",
                p.entries()
            );
        }
    }

    #[test]
    fn span_contains_rejects_wrong_dimension() {
        // 跨維度邊界:來自另一個空間的向量不可能屬於這個 span。span{(1,2,3)} 活在 ℝ³,
        // 一個 ℝ² 的查詢不在上面 —— 必須回 false,而非 crash(contains 的 Err 分支)。
        let line = Span::new(SPAN_EPS, vec![v(vec![1.0, 2.0, 3.0])]); // ℝ³ 的線
        assert!(
            !line.contains(&v(vec![1.0, 2.0])),
            "ℝ² 向量不可能在 ℝ³ 的 span 裡"
        );
    }

    #[test]
    fn affine_span_rejects_wrong_dimension() {
        // 仿射版的同一道邊界:x、base 在不同空間時位移 x − base 沒有定義,
        // 所以 x 不在平直集上(affine_span closure 的 Err 分支)。
        let plane = affine_span(
            SPAN_EPS,
            v(vec![0.0, 0.0, 1.0]),
            vec![v(vec![1.0, 0.0, 0.0])],
        );
        assert!(
            !plane.contains(&v(vec![3.0, 4.0])),
            "ℝ² 向量不在 ℝ³ 的仿射平直集上"
        );
    }

    #[test]
    fn span_as_predicate_bridges_to_set_algebra() {
        // 回到集合代數視角的橋:as_predicate 回傳的述詞必須與 contains 一致,且能像任何
        // 述詞一樣插進布林集合運算。這裡 line ∪ plane = plane,因為線落在平面內。
        let line = Span::new(SPAN_EPS, vec![v(vec![1.0, 0.0, 0.0])]);
        let plane = Span::new(
            SPAN_EPS,
            vec![v(vec![1.0, 0.0, 0.0]), v(vec![0.0, 1.0, 0.0])],
        );

        let line_pred = line.as_predicate();
        let on_line_pt = v(vec![5.0, 0.0, 0.0]);
        let off_line_pt = v(vec![0.0, 5.0, 0.0]);
        // 述詞與它包裝的方法必須給出相同判定。
        assert_eq!(line_pred.contains(&on_line_pt), line.contains(&on_line_pt));
        assert_eq!(
            line_pred.contains(&off_line_pt),
            line.contains(&off_line_pt)
        );

        // 而且它與集合代數可組合:(0,5,0) 在線外、在平面內,故在 line ∪ plane 裡。
        let union = line_pred.union(&plane.as_predicate());
        assert!(
            union.contains(&off_line_pt),
            "(0,5,0) 在平面內,必在 line ∪ plane 裡"
        );
    }

    #[test]
    fn spans_all_known_cases() {
        // 手算的對照案例,獨立於隨機 law test —— 萬一共用的 rank 機制有 bug,
        // 它就無法躲在「條件一致但全錯」的後面。
        // 2×2 單位矩陣:行 e₀、e₁ 張滿 ℝ²,rank 2 = m。
        assert!(
            Span::from_columns(SPAN_EPS, &Matrix::identity(2)).spans_all(2),
            "單位矩陣張滿自己的空間"
        );
        // 寬 2×3、內嵌單位:rank 2 = m,onto ℝ²。
        let wide = Matrix::from_rows(vec![vec![1.0, 0.0, 5.0], vec![0.0, 1.0, 7.0]]);
        assert!(
            Span::from_columns(SPAN_EPS, &wide).spans_all(2),
            "寬矩陣 full row rank 即 onto"
        );
        // 高 3×2:至多 rank 2 < 3 列,行永遠填不滿 ℝ³ —— onto 要的是 row full rank(m)。
        let tall = Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]]);
        assert!(
            !Span::from_columns(SPAN_EPS, &tall).spans_all(3),
            "高矩陣不可能 onto(列數 > 行數,方向不夠)"
        );
        // 2×2 有相依行:rank 1 < 2,span 只是 ℝ² 裡一條線。
        let deficient = Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert!(
            !Span::from_columns(SPAN_EPS, &deficient).spans_all(2),
            "rank 不足即非 onto"
        );
    }

    #[test]
    fn combination_classifies_and_witnesses() {
        // 建構式成員判定:除了 yes/no,還交回見證權重,並把答案分入 solve 的三結局 ——
        // generator 獨立時唯一座標、相依時無限多、x 不在 span 時無解;且 contains 必須一致。
        // 1. xy 平面 span{e₀,e₁}⊂ℝ³,獨立 generator:(3,5,0) 的唯一座標 (3,5)。
        let plane = Span::new(
            SPAN_EPS,
            vec![v(vec![1.0, 0.0, 0.0]), v(vec![0.0, 1.0, 0.0])],
        );
        match plane.combination(&v(vec![3.0, 5.0, 0.0])) {
            Solution::Unique(w) => {
                assert!(
                    w.approx_equals(&v(vec![3.0, 5.0]), SPAN_EPS),
                    "唯一座標應為 (3,5)"
                );
                // 權重必須真能重建 x。
                let rebuilt = Vector::linear_combination(
                    w.entries(),
                    &[v(vec![1.0, 0.0, 0.0]), v(vec![0.0, 1.0, 0.0])],
                )
                .unwrap();
                assert!(
                    rebuilt.approx_equals(&v(vec![3.0, 5.0, 0.0]), SPAN_EPS),
                    "權重未能重建 x"
                );
            }
            other => panic!("獨立 generator 上應為 Unique,得 {other:?}"),
        }
        assert!(plane.contains(&v(vec![3.0, 5.0, 0.0])));

        // 2. 同平面,但 (3,5,1) 被抬離:無權重。
        assert!(matches!(
            plane.combination(&v(vec![3.0, 5.0, 1.0])),
            Solution::Inconsistent
        ));
        assert!(!plane.contains(&v(vec![3.0, 5.0, 1.0])));

        // 3. 相依 generator span{(1,1),(2,2)} 是一條線。(3,3) 在線上,但 (3,0)、(1,1) 都能
        //    見證它,故權重**不唯一**。
        let line = Span::new(SPAN_EPS, vec![v(vec![1.0, 1.0]), v(vec![2.0, 2.0])]);
        assert!(matches!(
            line.combination(&v(vec![3.0, 3.0])),
            Solution::Infinite
        ));
        assert!(line.contains(&v(vec![3.0, 3.0])));

        // 4. 空 span {0}:零向量是**空組合** —— 唯一,權重向量長度 0(沒有 generator 可加權)。
        let empty = Span::new(SPAN_EPS, vec![]);
        match empty.combination(&v(vec![0.0, 0.0, 0.0])) {
            Solution::Unique(w) => assert_eq!(w.rows(), 0, "空 span 的權重應為 length-0"),
            other => panic!("空 span 的零向量應為 Unique,得 {other:?}"),
        }
        assert!(empty.contains(&v(vec![0.0, 0.0, 0.0])));

        // 5. 空 span {0} 只含零:非零向量無見證。
        assert!(matches!(
            empty.combination(&v(vec![0.0, 0.0, 1.0])),
            Solution::Inconsistent
        ));

        // 6. 跨維度:ℝ² 查詢對 ℝ³ 的 span,根本不在環境空間 → 無權重(System::new 的 Err 分支)。
        let r3 = Span::new(SPAN_EPS, vec![v(vec![1.0, 2.0, 3.0])]);
        assert!(matches!(
            r3.combination(&v(vec![1.0, 2.0])),
            Solution::Inconsistent
        ));
    }
}

/// Theorem 1.6 化為可執行斷言。對 m×n 矩陣 A,五句**等價**:(a) A 的行張滿 ℝᵐ、
/// (b) Ax=b 對**每個** b ∈ ℝᵐ 相容、(c) rank(A)=m(**列**數)、(d) A 的 RREF 無零列、
/// (e) 每列都有 pivot。這是「onto / 滿射」定理 —— Theorem 1.5 單一 b 相容性的「對每個 b」
/// 強化版;rank 比的是 m(列),不是 n(行),(c) 是 full **ROW** rank。
///
/// (a)(c)(d)(e) 皆可直接計算,跨隨機形狀的 A 必須給出相同判定;(b) 對所有 b 量化、無法
/// 列舉,只在定理保證的方向(onto ⟹ 每個 b 相容)以探針驗證 —— 反向不探(隨機 b 落在真
/// 子空間的機率為零,探了只是「幾乎必然」,非證明,同 [`PredicateSet`] 的 probe≠proof)。
#[cfg(test)]
mod laws {
    use super::*;
    use proptest::prelude::*;

    /// 產生 `rows×cols`、元素為 [-10, 10] 整數的矩陣(f64 下精確)。
    fn int_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-10i64..=10, cols), rows).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            )
        })
    }

    /// 產生長度 `n`、元素為 [-10, 10] 整數的向量。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()))
    }

    /// 條件 (d):A 的 RREF 沒有零列;等價於 (e) 每列都帶一個 pivot。`pivot_col` 對零列回
    /// `None`,所以「每列有 pivot」就是「沒有任一列的 `pivot_col` 是 `None`」。
    fn rref_has_no_zero_row(a: &Matrix, eps: f64) -> bool {
        // 換你寫:先把 a 化成 RREF(`a.reduced_row_echelon_form(eps)`),再檢查**每一列**
        // 都有 pivot。`rref.pivot_col(i, eps)` 對零列回 `None`,所以條件就是
        // 「(0..rref.rows()) 每個 i 都讓 pivot_col(i).is_some()」。
        let rref = a.reduced_row_echelon_form(eps);
        (0..rref.rows()).all(|i| rref.pivot_col(i, eps).is_some())
    }

    proptest! {
        /// (a)、(c)、(d)/(e) 四個可計算條件在每種形狀的隨機 A 上必須給出**同一**判定。
        #[test]
        fn theorem_1_6_conditions_agree(
            a in (1usize..=5, 1usize..=5).prop_flat_map(|(r, c)| int_matrix(r, c)),
        ) {
            const EPS: f64 = 1e-9;
            let rows = a.rows();
            let cond_a = Span::from_columns(EPS, &a).spans_all(rows); // (a)
            let cond_c = a.rank(EPS) == rows; // (c)
            let cond_de = rref_has_no_zero_row(&a, EPS); // (d)/(e)
            prop_assert!(
                cond_a == cond_c && cond_c == cond_de,
                "Theorem 1.6 判準不一致: (a)={cond_a} (c)={cond_c} (d/e)={cond_de}\n a={a:?}"
            );
        }

        /// (b) 在定理保證的方向:當 (a) 成立(A 的行張滿 ℝᵐ),隨機 b 必**全部**相容。用整數
        /// 方陣 + `prop_assume` 篩出 onto 的 A —— 整數方陣幾乎都 full rank,故 onto 分支幾乎
        /// 必被測到(不像 Go 需顯式 checked 計數防 vacuous pass)。
        #[test]
        fn onto_means_every_b_consistent(
            (a, bs) in (1usize..=4)
                .prop_flat_map(|n| (int_matrix(n, n), prop::collection::vec(int_vector(n), 1..=6))),
        ) {
            const EPS: f64 = 1e-9;
            let n = a.rows();
            prop_assume!(Span::from_columns(EPS, &a).spans_all(n)); // 只約束 onto 的 A
            for b in bs {
                let system = System::new(a.clone(), b.clone()).unwrap();
                prop_assert!(
                    system.is_consistent(EPS),
                    "(a) 成立但 Ax=b 不相容\n a={a:?}\n b={b:?}"
                );
            }
        }

        /// 「combination 回傳見證」化為隨機律:x 由 generator 線性組合而成 → 依建構在 span 裡,
        /// 故 combination 絕不對它回 Inconsistent;答案唯一時,回傳的權重必須真能重建那個 x。
        /// 並一併探測建構式 ⟺ 布林一致:contains(probe) == (combination(probe) 非 Inconsistent)。
        #[test]
        fn combination_is_a_witness(
            (gens, weights, probe) in (1usize..=4, 1usize..=4).prop_flat_map(|(dim, k)| {
                (
                    prop::collection::vec(int_vector(dim), k),
                    int_vector(k),
                    int_vector(dim),
                )
            }),
        ) {
            const EPS: f64 = 1e-9;
            let span = Span::new(EPS, gens.clone());
            // x = Σ weightsⱼ·gensⱼ 依建構在 span{gens} 裡 —— 成員資格的定義,combination 欠它見證。
            let x = Vector::linear_combination(weights.entries(), &gens).unwrap();
            let got = span.combination(&x);
            prop_assert!(
                !matches!(got, Solution::Inconsistent),
                "x 是 generator 的組合卻回 Inconsistent\n x={x:?}"
            );
            // 唯一解必須帶回能重建 x 的權重(唯一答案就是那一組真座標)。
            if let Solution::Unique(w) = &got {
                let rebuilt = Vector::linear_combination(w.entries(), &gens).unwrap();
                prop_assert!(rebuilt.approx_equals(&x, EPS), "唯一權重未能重建 x\n x={x:?}");
            }
            // 任意探針(通常落在 span 外):赤裸與建構式判定必須一致。
            let has_witness = !matches!(span.combination(&probe), Solution::Inconsistent);
            prop_assert_eq!(span.contains(&probe), has_witness, "contains 與 combination 不一致");
        }
    }
}
