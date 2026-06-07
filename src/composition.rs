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

use crate::{LinAlgError, Transformation};

impl Transformation {
    /// 合成(composition):**(U ∘ T)(x) = U(T(x))**,先施 T、再施 U ——
    /// `self` 是外層 U,讀法與數學記法 ∘ 同向:`u.compose(&t)` = U ∘ T。
    ///
    /// 課程的核心性質:**T_B ∘ T_A = T_BA** —— 合成轉換的標準矩陣就是
    /// 兩個標準矩陣的乘積。為什麼?乘法結合律是橋:
    /// U(T(x)) = B(Ax) = (BA)x —— 「先後施作兩個函數」與「先乘好矩陣
    /// 再一次施作」是同一個映射。第三單元刻乘法時的 law
    /// `multiply_is_composition_of_actions` 早就把這條存證了;本方法把它
    /// 從定理升格為 **API**:合成「就是」乘法([`Matrix::multiply`])。
    ///
    /// 維度相容性(題目驗收):T 的 codomain 必須 = U 的 domain(中間的 ℝᵐ
    /// 要接得上),否則 [`LinAlgError::DimensionMismatch`]。**不需要自己檢查**:
    /// 這恰好就是 `multiply` 的 can_multiply 條件(BA 可乘 ⟺ B 的行數 =
    /// A 的列數 ⟺ 同一個 m)—— `?` 或 `map` 讓錯誤自己傳播,驗證規則單一真相。
    ///
    /// 四題裡唯一**不收 epsilon** 的:乘法是精確運算,無消去、無判零 ——
    /// 哪些運算碰消去、哪些不碰,從簽名就看得出來。
    ///
    /// 實作提示:一行 —— `multiply` 回 `Result<Matrix, _>`,差一步包成
    /// `Transformation`:`Result::map` 收建構子(point-free 風格可直接
    /// `map(Transformation::new)`)。
    ///
    /// [`Matrix::multiply`]: crate::Matrix::multiply
    pub fn compose(&self, t: &Transformation) -> Result<Transformation, LinAlgError> {
        self.matrix().multiply(t.matrix()).map(Transformation::new)
    }

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

    /// 逆轉換(inverse transformation):**T⁻¹ 的標準矩陣 = A⁻¹**(Theorem 2.13:
    /// T 可逆 ⟺ A 可逆,且 T⁻¹ = T_{A⁻¹})。
    ///
    /// 「可逆」的函數定義:存在 U 使 **U ∘ T = I 且 T ∘ U = I**(走過去能走回來,
    /// 而且每一點都回到原地)—— 這樣的 U 只有雙射(1-1 且 onto)才配得出來
    /// (Theorem 2.12),而 1-1 且 onto ⟺ rank = n = m ⟺ 方陣滿秩 ⟺ A 可逆:
    /// 函數視角的「可逆」與矩陣視角的「可逆」在 IMT 會師,操作的字典補上最後一格
    /// (⁻¹ ↔ ⁻¹)。
    ///
    /// 失敗分層(委派 [`Matrix::inverse`] 原樣傳播,題目的 String 錯誤升級成
    /// 可 `match` 的 enum):
    /// - 非方陣 → [`LinAlgError::NotSquare`](帶實際形狀):換空間的映射
    ///   (ℝⁿ → ℝᵐ,n ≠ m)連「回到原空間」都談不上;
    /// - 方陣但奇異 → [`LinAlgError::NotInvertible`]:塌縮的方向回不去
    ///   (不 1-1:多個輸入擠在同一輸出,「逆」不知道該回哪一個)。
    ///
    /// `epsilon`:Gauss-Jordan 的 pivot 判零門檻(同 [`Matrix::inverse`])。
    ///
    /// 實作提示:與 `compose` 同一個收法 —— `Result<Matrix, _>` 差一步建構子,
    /// `map(Transformation::new)`。兩題長得一模一樣不是巧合:**操作的字典
    /// (∘ ↔ ×、⁻¹ ↔ ⁻¹)本來就是同一個形狀** —— 矩陣運算算完,包回函數視角。
    ///
    /// [`Matrix::inverse`]: crate::Matrix::inverse
    pub fn inverse(&self, epsilon: f64) -> Result<Transformation, LinAlgError> {
        self.matrix().inverse(epsilon).map(Transformation::new)
    }
}

#[cfg(test)]
mod tests {
    use crate::{LinAlgError, Matrix, Transformation, Vector};

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

    /// 練習 2 核心性質:U ∘ T 的標準矩陣 = B·A(矩陣序與 ∘ 同向)。
    /// 旋轉 90°(U)接在 x 軸反射(T)後面 —— 手算 B·A 釘住乘法方向:
    /// B = [[0,−1],[1,0]]、A = [[1,0],[0,−1]] → BA = [[0,1],[1,0]](swap)。
    /// 整數元素在 f64 下精確,用精確 equals。
    #[test]
    fn compose_standard_matrix_is_product_ba() {
        let u = Transformation::new(Matrix::from_rows(vec![vec![0.0, -1.0], vec![1.0, 0.0]]));
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, -1.0]]));
        let c = u.compose(&t).unwrap();
        assert!(
            c.matrix()
                .equals(&Matrix::from_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]])),
            "C = B·A:先反射、再旋轉 = 對角線鏡射(swap)"
        );
    }

    /// 題目驗收:維度鏈 —— U 為 p×m、T 為 m×n,合成是 p×n(ℝⁿ → ℝᵖ):
    /// 2×3 ∘ 3×4 = 2×4,中間的 ℝ³ 被「乘掉」,函數視角讀作
    /// ℝ⁴ —T→ ℝ³ —U→ ℝ²(domain 取 T 的、codomain 取 U 的)。
    #[test]
    fn compose_chains_dimensions_through_middle_space() {
        let u = Transformation::new(Matrix::new(2, 3)); // ℝ³ → ℝ²
        let t = Transformation::new(Matrix::new(3, 4)); // ℝ⁴ → ℝ³
        let c = u.compose(&t).unwrap();
        assert_eq!(c.dimensions(), (4, 2), "U ∘ T: ℝ⁴ → ℝ²");
    }

    /// 題目驗收:中間空間接不上(T 落在 ℝ²,U 卻從 ℝ³ 出發)→
    /// DimensionMismatch —— 由 multiply 的 can_multiply 檢查傳播,
    /// 不是 compose 自己另寫的驗證(單一真相)。
    #[test]
    fn compose_rejects_mismatched_middle_space() {
        let u = Transformation::new(Matrix::new(2, 3)); // ℝ³ → ℝ²
        let t = Transformation::new(Matrix::new(2, 4)); // ℝ⁴ → ℝ²:落點 ≠ U 的起點
        assert_eq!(u.compose(&t).unwrap_err(), LinAlgError::DimensionMismatch);
    }

    /// 題目驗收:C(x) = U(T(x)) —— 單一見證(全稱版在 laws):
    /// x 走「先 T 後 U」兩步,與走合成 C 一步,落在同一點。
    #[test]
    fn compose_apply_agrees_with_sequential_application() {
        let u = Transformation::new(Matrix::from_rows(vec![vec![1.0, 1.0], vec![0.0, 2.0]]));
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![3.0, 0.0, 1.0],
            vec![-1.0, 2.0, 0.0],
        ]));
        let x = Vector::from_vec(vec![1.0, 2.0, -1.0]);
        let two_steps = u.apply(&t.apply(&x).unwrap()).unwrap();
        let one_step = u.compose(&t).unwrap().apply(&x).unwrap();
        assert!(one_step.equals(&two_steps), "兩條路必須會合");
    }

    /// 合成**不可交換**:U ∘ T ≠ T ∘ U —— 先反射再旋轉 ≠ 先旋轉再反射
    /// (矩陣乘法不可交換的函數視角;這正是 ∘ 要分左右的原因)。
    #[test]
    fn compose_is_not_commutative() {
        let u = Transformation::new(Matrix::from_rows(vec![vec![0.0, -1.0], vec![1.0, 0.0]]));
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, -1.0]]));
        let ut = u.compose(&t).unwrap();
        let tu = t.compose(&u).unwrap();
        assert!(!ut.matrix().equals(tu.matrix()), "BA ≠ AB");
    }

    /// 練習 3 題目原例:A = [[1,2],[3,5]] → A⁻¹ = [[−5,2],[3,−1]]
    /// (det = −1;此例的 Gauss-Jordan 全程整數,殘差為零,但比較慣例
    /// 仍走 approx —— inverse 經過消去,精確 equals 不是它的契約)。
    #[test]
    fn inverse_of_textbook_example() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 5.0]]));
        let t_inv = t.inverse(1e-9).unwrap();
        assert!(
            t_inv.matrix().approx_equals(
                &Matrix::from_rows(vec![vec![-5.0, 2.0], vec![3.0, -1.0]]),
                1e-9
            ),
            "T⁻¹ 的標準矩陣 = A⁻¹(Theorem 2.13)"
        );
    }

    /// 幾何例:剪切「推 2」的逆 = 剪切「推 −2」—— 怎麼變形的就怎麼推回去,
    /// 逆轉換的幾何直觀比代數公式先一步知道答案。
    #[test]
    fn inverse_of_shear_is_reverse_shear() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![0.0, 1.0]]));
        let t_inv = t.inverse(1e-9).unwrap();
        assert!(t_inv.matrix().approx_equals(
            &Matrix::from_rows(vec![vec![1.0, -2.0], vec![0.0, 1.0]]),
            1e-9
        ));
    }

    /// 題目驗收(一):非方陣 → NotSquare(帶實際形狀)—— 換空間的映射
    /// 連「回到原空間」都談不上,與「方陣但奇異」是兩層不同的失敗。
    #[test]
    fn inverse_rejects_non_square() {
        let t = Transformation::new(Matrix::new(2, 3));
        assert_eq!(
            t.inverse(1e-9).unwrap_err(),
            LinAlgError::NotSquare { rows: 2, cols: 3 }
        );
    }

    /// 題目驗收(二):方陣但 rank 不足 → NotInvertible —— 行成比例,
    /// 整條直線被吸到原點(不 1-1),「逆」不知道該回哪一點。
    #[test]
    fn inverse_rejects_singular_square() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]));
        assert_eq!(t.inverse(1e-9).unwrap_err(), LinAlgError::NotInvertible);
    }

    /// 題目驗收(三):T⁻¹(T(x)) = x —— 單一見證(全稱版在 laws):
    /// x 被 T 送出去、再被 T⁻¹ 接回來,落回原地。
    #[test]
    fn inverse_undoes_transformation_witness() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 5.0]]));
        let t_inv = t.inverse(1e-9).unwrap();
        let x = Vector::from_vec(vec![4.0, -7.0]);
        let round_trip = t_inv.apply(&t.apply(&x).unwrap()).unwrap();
        assert!(round_trip.approx_equals(&x, 1e-9), "T⁻¹(T(x)) ≠ x");
    }
}

/// 合成與可逆性的 property test —— 沿 5-3 的傳統:**跨練習交叉對帳**,
/// 隨練習推進逐條累積(策略沿「先抽形狀、再抽內容」的依賴式兩階段抽樣)。
#[cfg(test)]
mod laws {
    use crate::{Matrix, Transformation, Vector};
    use proptest::prelude::*;

    /// 消去法判零門檻(整數輸入的殘差遠低於此,沿 range laws 的 EPS)。
    const EPS: f64 = 1e-9;

    /// 涉及 `inverse` 的**等式比較**容差:A⁻¹ 帶消去殘差,乘回去、再求逆等
    /// 連續運算會放大誤差 —— 沿 inverse 章 laws 的慣例放寬到 1e-6。
    /// (EPS 是「判零門檻」、EQ_EPS 是「等式容差」,兩個容差各司其職 ——
    /// 這正是「epsilon 由呼叫端視運算數量級指定」慣例的用意。)
    const EQ_EPS: f64 = 1e-6;

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

    /// 長度 `n`、元素為 [-10, 10] 整數的向量。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()))
    }

    /// 可合成的一對(B: p×m、A: m×n)連同一支住在 A 定義域的 x ∈ ℝⁿ ——
    /// 中間維 m 與輸入維 n 都在同一次 flat_map 共用,**依建構**保證接得上
    /// (免 prop_assume 丟樣本,沿 tall / wide 的同一招)。
    fn composable_pair_with_input() -> impl Strategy<Value = (Matrix, Matrix, Vector)> {
        (1usize..=3, 1usize..=3, 1usize..=3)
            .prop_flat_map(|(p, m, n)| (int_matrix(p, m), int_matrix(m, n), int_vector(n)))
    }

    /// 可合成的三鏈(C: p×m、B: m×n、A: n×q)—— 結合律要三個才擺得開。
    fn composable_triple() -> impl Strategy<Value = (Matrix, Matrix, Matrix)> {
        (1usize..=3, 1usize..=3, 1usize..=3, 1usize..=3)
            .prop_flat_map(|(p, m, n, q)| (int_matrix(p, m), int_matrix(m, n), int_matrix(n, q)))
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

        // 練習 2 的定義律(題目驗收的全稱版):∀ x,(U ∘ T)(x) = U(T(x)) ——
        // 「先乘好矩陣、一步走完」與「兩個函數先後施作」必須處處會合。
        // 小整數乘加在 f64 下精確 → 精確 equals(維度依建構接得上,unwrap 安全)。
        #[test]
        fn compose_apply_agrees_with_sequential_application(
            (b, a, x) in composable_pair_with_input(),
        ) {
            let u = Transformation::new(b);
            let t = Transformation::new(a);
            let one_step = u.compose(&t).unwrap().apply(&x).unwrap();
            let two_steps = u.apply(&t.apply(&x).unwrap()).unwrap();
            prop_assert!(one_step.equals(&two_steps), "x={x:?}");
        }

        // 結合律:(U ∘ T) ∘ S = U ∘ (T ∘ S) —— 矩陣乘法結合律的函數視角。
        // 函數合成「天生」結合(兩邊都是 x ↦ U(T(S(x)));這正是矩陣乘法
        // 結合律最漂亮的證明 —— Transformation 在 ∘ 下構成 monoid。
        #[test]
        fn compose_is_associative((c, b, a) in composable_triple()) {
            let u = Transformation::new(c);
            let t = Transformation::new(b);
            let s = Transformation::new(a);
            let left = u.compose(&t).unwrap().compose(&s).unwrap();
            let right = u.compose(&t.compose(&s).unwrap()).unwrap();
            prop_assert!(left.matrix().equals(right.matrix()));
        }

        // 單位元:I_m ∘ T = T = T ∘ I_n —— identity(5-1)在 ∘ 下中立,
        // monoid 的另一半;注意左右兩支單位的**維度不同**(各接一端)。
        #[test]
        fn identity_is_neutral_for_composition(a in int_matrix_any_shape()) {
            let t = Transformation::new(a.clone());
            let left = Transformation::identity(t.codomain_dim()).compose(&t).unwrap();
            let right = t.compose(&Transformation::identity(t.domain_dim())).unwrap();
            prop_assert!(left.matrix().equals(&a), "I_m ∘ T = T");
            prop_assert!(right.matrix().equals(&a), "T ∘ I_n = T");
        }

        // 練習 3 的定義律(題目驗收的全稱版):∀ x,T⁻¹(T(x)) = x 且
        // T(T⁻¹(y)) = y —— 「逆」的本分:兩個方向都要回到原地
        // (可逆矩陣靠 prop_assume 篩,沿 inverse 章慣例;拒絕率低)。
        #[test]
        fn inverse_undoes_transformation(a in int_matrix(3, 3), x in int_vector(3)) {
            prop_assume!(a.is_invertible(EPS));
            let t = Transformation::new(a);
            let t_inv = t.inverse(EPS).unwrap();
            let there_and_back = t_inv.apply(&t.apply(&x).unwrap()).unwrap();
            prop_assert!(there_and_back.approx_equals(&x, EQ_EPS), "T⁻¹(T(x)) ≠ x");
            let back_and_there = t.apply(&t_inv.apply(&x).unwrap()).unwrap();
            prop_assert!(back_and_there.approx_equals(&x, EQ_EPS), "T(T⁻¹(y)) ≠ y");
        }

        // compose 與 inverse 會師:T⁻¹ ∘ T = Iₙ = T ∘ T⁻¹ —— 可逆的函數定義
        // (matrix 層的 inverse_satisfies_definition 驗過 A·A⁻¹ = I,
        //  這裡用本章自己的兩個 API 把同一條定義以函數詞彙重說一遍)。
        #[test]
        fn inverse_composes_to_identity(a in int_matrix(3, 3)) {
            prop_assume!(a.is_invertible(EPS));
            let t = Transformation::new(a);
            let t_inv = t.inverse(EPS).unwrap();
            let id = Matrix::identity(3);
            prop_assert!(
                t_inv.compose(&t).unwrap().matrix().approx_equals(&id, EQ_EPS),
                "T⁻¹ ∘ T ≠ I"
            );
            prop_assert!(
                t.compose(&t_inv).unwrap().matrix().approx_equals(&id, EQ_EPS),
                "T ∘ T⁻¹ ≠ I"
            );
        }

        // 襪子鞋子定理:(U ∘ T)⁻¹ = T⁻¹ ∘ U⁻¹ —— 解開的順序與穿上相反
        // (先穿襪再穿鞋,脫的時候先脫鞋再脫襪)。乘積可逆由 inverse 章的
        // product_invertible_iff_both_factors_are 保證,unwrap 安全。
        #[test]
        fn inverse_of_composition_reverses_order(
            b in int_matrix(3, 3),
            a in int_matrix(3, 3),
        ) {
            prop_assume!(b.is_invertible(EPS) && a.is_invertible(EPS));
            let u = Transformation::new(b);
            let t = Transformation::new(a);
            let left = u.compose(&t).unwrap().inverse(EPS).unwrap();
            let right = t.inverse(EPS).unwrap().compose(&u.inverse(EPS).unwrap()).unwrap();
            prop_assert!(left.matrix().approx_equals(right.matrix(), EQ_EPS));
        }

        // 對合律:(T⁻¹)⁻¹ = T —— 「逆的逆」回到自己(連兩次 Gauss-Jordan,
        // 殘差最大的一條,EQ_EPS 在這裡承重)。
        #[test]
        fn inverse_is_involution(a in int_matrix(3, 3)) {
            prop_assume!(a.is_invertible(EPS));
            let t = Transformation::new(a.clone());
            let back = t.inverse(EPS).unwrap().inverse(EPS).unwrap();
            prop_assert!(back.matrix().approx_equals(&a, EQ_EPS), "(T⁻¹)⁻¹ ≠ T");
        }

        // Theorem 2.12 存證(練習 1 ↔ 3 ↔ 5-3 三方交叉對帳):可逆 ⟺ 雙射
        // (1-1 且 onto)—— inverse 的 Ok/Err 與兩個述詞的合取必須一致。
        // 形狀隨機:非方陣兩邊都 false(Err(NotSquare) vs rank 搆不到兩端),
        // 方陣則在 IMT 會合 —— 這條是練習 4 函數視角實作的根據。
        #[test]
        fn invertible_iff_one_to_one_and_onto(a in int_matrix_any_shape()) {
            let t = Transformation::new(a);
            prop_assert_eq!(
                t.inverse(EPS).is_ok(),
                t.is_one_to_one(EPS) && t.is_onto(EPS)
            );
        }
    }
}
