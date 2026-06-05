//! 可逆矩陣(Invertible Matrix)—— 基本矩陣、可逆判定與反矩陣。
//!
//! 筆記「可逆矩陣」章:對 n×n 方陣 A,若存在 B 使 **AB = BA = Iₙ**,稱 A 可逆、
//! B 為 A 的反矩陣 A⁻¹(且唯一)。本模組依筆記的鋪陳分三步:
//!
//! 1. **基本矩陣 E**:對 Iₙ 施一次列基本運算所得;左乘 E 等於直接施作該列運算
//!    (Proposition),且每個 E 都可逆、其逆是同型的 E(逆向列運算)。
//! 2. **可逆判定**:可逆矩陣定理(IMT)—— A 可逆 ⟺ RREF(A) = Iₙ ⟺ rank(A) = n ⟺ …
//! 3. **反矩陣**:消去過程逐步累乘 E(P ← E·P),A 化到 Iₙ 時 P 即 A⁻¹ ——
//!    把 Theorem 2.3 的「PA = R,P 為基本矩陣之乘積」直接寫成演算法。
//!
//! 與 `elimination` 同款佈局:方法掛在 [`Matrix`] 上(`impl Matrix`),但本模組跨在
//! `matrix` 模組外、碰不到 private 的 `data` 欄位 —— 一律走 public API,再次驗證
//! 先前刻的公開介面足以表達新概念。

use crate::{LinAlgError, Matrix};

impl Matrix {
    /// 基本矩陣(列交換):對 Iₙ 施**一次** Rᵢ ↔ Rⱼ 所得的 n×n 矩陣。
    ///
    /// 定義即實作:[`identity`](Matrix::identity) + 一次 [`swap_rows`](Matrix::swap_rows),
    /// 驗證原封委派給 ERO 本身 —— 建構子與列運算的合法規則只有單一真相。
    /// `i == j` 沿 `swap_rows` 的語意是無害 no-op,得到 Iₙ(Iₙ 自己也是基本矩陣)。
    ///
    /// 左乘它 = 對任意(列數相容的)矩陣施同一個列運算:`E·A` 就是 A 交換 i、j 兩列
    /// (筆記的 Proposition,見 laws `left_multiply_swap_acts_as_swap_rows`)。
    ///
    /// `i` 或 `j` 越界(`>= n`)→ [`LinAlgError::IndexOutOfRange`]。
    pub fn elementary_swap(n: usize, i: usize, j: usize) -> Result<Matrix, LinAlgError> {
        let mut e = Matrix::identity(n);
        e.swap_rows(i, j)?; // 越界驗證原封委派給 ERO(identity(n) 的 rows 即 n)
        Ok(e)
    }

    /// 基本矩陣(列伸縮):對 Iₙ 施**一次** Rᵢ → c·Rᵢ 所得 —— 即對角線第 i 個元素
    /// 換成 c 的 Iₙ。
    ///
    /// `i` 越界 → [`LinAlgError::IndexOutOfRange`];`c == 0.0` →
    /// [`LinAlgError::ScaleByZero`](乘 0 抹掉整列、不可逆,不算 elementary ——
    /// 與筆記「每一個基本矩陣都是可逆的」一致,擋在建構期)。
    pub fn elementary_scale(n: usize, i: usize, c: f64) -> Result<Matrix, LinAlgError> {
        let mut e = Matrix::identity(n);
        e.scale_row(i, c)?; // 越界 / 乘零驗證原封委派給 ERO
        Ok(e)
    }

    /// 基本矩陣(列倍加):對 Iₙ 施**一次** R_dst → R_dst + c·R_src 所得 ——
    /// 即 (dst, src) 位置多一個 c 的 Iₙ。
    ///
    /// `dst` 或 `src` 越界 → [`LinAlgError::IndexOutOfRange`];`dst == src` →
    /// [`LinAlgError::SameRow`](把一列折進自己會塌成純量縮放,c = −1 時不可逆)。
    pub fn elementary_add_scaled(
        n: usize,
        dst: usize,
        src: usize,
        c: f64,
    ) -> Result<Matrix, LinAlgError> {
        let mut e = Matrix::identity(n);
        e.add_scaled_row(dst, src, c)?; // 越界 / 同列驗證原封委派給 ERO
        Ok(e)
    }

    /// 可逆判定:方陣 A 可逆 ⟺ 存在 B 使 AB = BA = Iₙ。
    ///
    /// 定義裡的「存在 B」無法直接檢查 —— 可逆矩陣定理(IMT,筆記 Theorem 2.6)
    /// 救場:對 n×n 方陣,RREF(A) = Iₙ、rank(A) = n、nullity(A) = 0、行向量線性
    /// 獨立、Ax = b 恆有唯一解⋯⋯皆與可逆**等價**,任挑一個可計算的當實作即可。
    /// 本方法選定其一;其餘等價條件全數寫成 laws 隨機互驗(見 `imt_*` 系列)——
    /// **IMT 本身就是這個函式的測試**。
    ///
    /// 述詞回 `bool`(沿 `can_multiply` / `is_stochastic` 慣例):可逆性只對方陣
    /// 有定義,非方陣直接回 `false`;要精確區分「非方陣」這個失敗原因的呼叫端,
    /// 用批 3 的 `inverse`(回 `LinAlgError::NotSquare`)。
    ///
    /// `epsilon`:消去過程的判零門檻(同 [`rank`](Matrix::rank) /
    /// [`reduced_row_echelon_form`](Matrix::reduced_row_echelon_form))。
    pub fn is_invertible(&self, epsilon: f64) -> bool {
        if self.rows() != self.cols() {
            return false; // 非方陣直接回 false(可逆性未定義)
        }
        self.reduced_row_echelon_form(epsilon)
            .equals(&Matrix::identity(self.rows()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementary_swap_permutes_identity_rows() {
        let e = Matrix::elementary_swap(3, 0, 2).unwrap();
        let want = Matrix::from_rows(vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ]);
        assert!(e.equals(&want), "e={e:?}");
    }

    /// i == j 沿 `swap_rows` 的 no-op 語意:得到 Iₙ 本身 —— Iₙ 也是基本矩陣
    /// (「施一次什麼都不變的列運算」)。
    #[test]
    fn elementary_swap_same_row_yields_identity() {
        let e = Matrix::elementary_swap(3, 1, 1).unwrap();
        assert!(e.equals(&Matrix::identity(3)), "e={e:?}");
    }

    #[test]
    fn elementary_scale_puts_scalar_on_diagonal() {
        let e = Matrix::elementary_scale(2, 1, 3.0).unwrap();
        let want = Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 3.0]]);
        assert!(e.equals(&want), "e={e:?}");
    }

    #[test]
    fn elementary_add_scaled_puts_coefficient_off_diagonal() {
        // R₂ += −2·R₀:I₃ 的 (2, 0) 位置變成 −2,其餘不動。
        let e = Matrix::elementary_add_scaled(3, 2, 0, -2.0).unwrap();
        let want = Matrix::from_rows(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![-2.0, 0.0, 1.0],
        ]);
        assert!(e.equals(&want), "e={e:?}");
    }

    /// 筆記 Proposition 的具體案例:左乘 E = 施作該列運算(一般形式見 laws)。
    #[test]
    fn left_multiplying_elementary_applies_the_row_operation() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let e = Matrix::elementary_add_scaled(2, 1, 0, -3.0).unwrap();
        let got = e.multiply(&a).unwrap();
        // R₁ += −3·R₀:[3, 4] + (−3)·[1, 2] = [0, −2]
        let want = Matrix::from_rows(vec![vec![1.0, 2.0], vec![0.0, -2.0]]);
        assert!(got.equals(&want), "got={got:?}");
    }

    /// 消去殘差的判零門檻(沿 elimination tests 的 REF_EPSILON)。
    const EPS: f64 = 1e-9;

    /// IMT 最直觀的兩端:Iₙ 可逆;零矩陣不可逆(筆記的「不可逆情況」之一)。
    #[test]
    fn identity_is_invertible_zero_matrix_is_not() {
        assert!(Matrix::identity(3).is_invertible(EPS));
        assert!(!Matrix::new(3, 3).is_invertible(EPS));
    }

    #[test]
    fn full_rank_square_matrix_is_invertible() {
        let m = Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 1.0]]);
        assert!(m.is_invertible(EPS));
    }

    /// 第二列 = 2×第一列 → RREF 含零列 → 不可逆(筆記:RREF 含零列者無逆矩陣)。
    #[test]
    fn rank_deficient_matrix_is_not_invertible() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert!(!m.is_invertible(EPS));
    }

    /// 可逆性只對方陣定義:非方陣回 `false`(述詞慣例),即使它「長得像」截斷的 I。
    #[test]
    fn non_square_matrix_is_not_invertible() {
        let m = Matrix::from_rows(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        assert!(!m.is_invertible(EPS));
    }

    /// 1×1 邊界:[c] 可逆 ⟺ c ≠ 0(逆是 [1/c])—— 可逆性最小的非平凡案例。
    #[test]
    fn one_by_one_invertible_iff_entry_nonzero() {
        assert!(Matrix::from_rows(vec![vec![5.0]]).is_invertible(EPS));
        assert!(!Matrix::from_rows(vec![vec![0.0]]).is_invertible(EPS));
    }

    /// 驗證原封委派給底層 ERO:三種失敗各回對應的錯誤種類,與直接呼叫 ERO 一致。
    #[test]
    fn constructors_propagate_ero_validation() {
        assert_eq!(
            Matrix::elementary_swap(2, 0, 2).unwrap_err(),
            LinAlgError::IndexOutOfRange { index: 2, len: 2 }
        );
        assert_eq!(
            Matrix::elementary_scale(2, 0, 0.0).unwrap_err(),
            LinAlgError::ScaleByZero
        );
        assert_eq!(
            Matrix::elementary_add_scaled(2, 1, 1, 2.0).unwrap_err(),
            LinAlgError::SameRow
        );
    }
}

/// 基本矩陣的 property test —— 把筆記的 Proposition 與「可逆性」寫成 for-all 形式。
#[cfg(test)]
mod laws {
    use super::*;
    use crate::{Solution, System, Vector, is_linearly_independent};
    use proptest::prelude::*;

    /// 消去殘差的判零門檻(沿 elimination laws 的 `nullity_agrees_with_solve`:
    /// 整數矩陣相乘、消去後殘差遠低於此,而整數矩陣的 det 是整數,可逆 / 奇異
    /// 之間沒有「灰色地帶」)。
    const EPS: f64 = 1e-7;

    /// 固定 `rows×cols`、小整數元素的矩陣(f64 下加減乘完全精確,可用精確 `equals`)。
    fn int_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-5i64..=5, cols), rows).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            )
        })
    }

    /// 小的非零整數純量(避開 ScaleByZero,且整數係數下乘積精確)。
    fn nonzero_int() -> impl Strategy<Value = f64> {
        prop_oneof![-3i64..=-1, 1i64..=3].prop_map(|c| c as f64)
    }

    /// 長度 `n`、小整數元素的向量(植入已知解用)。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-5i64..=5, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()))
    }

    proptest! {
        // 筆記 Proposition:左乘 E_swap = 直接 swap_rows。注意 A 刻意取 3×4 非方陣 ——
        // 列運算只看列數,E 的尺寸 n 由 A 的「列數」決定,行數無關。
        #[test]
        fn left_multiply_swap_acts_as_swap_rows(
            a in int_matrix(3, 4), i in 0usize..3, j in 0usize..3,
        ) {
            let e = Matrix::elementary_swap(3, i, j).unwrap();
            let via_e = e.multiply(&a).unwrap();
            let mut direct = a.clone();
            direct.swap_rows(i, j).unwrap();
            prop_assert!(via_e.equals(&direct), "E·A ≠ swap(A)\n a={a:?}\n i={i} j={j}");
        }

        #[test]
        fn left_multiply_scale_acts_as_scale_row(
            a in int_matrix(3, 4), i in 0usize..3, c in nonzero_int(),
        ) {
            let e = Matrix::elementary_scale(3, i, c).unwrap();
            let via_e = e.multiply(&a).unwrap();
            let mut direct = a.clone();
            direct.scale_row(i, c).unwrap();
            prop_assert!(via_e.equals(&direct), "E·A ≠ scale(A)\n a={a:?}\n i={i} c={c}");
        }

        #[test]
        fn left_multiply_add_scaled_acts_as_add_scaled_row(
            a in int_matrix(3, 4), dst in 0usize..3, step in 1usize..3, c in nonzero_int(),
        ) {
            let src = (dst + step) % 3; // dst ≠ src 依建構成立(同 elimination laws 的手法)
            let e = Matrix::elementary_add_scaled(3, dst, src, c).unwrap();
            let via_e = e.multiply(&a).unwrap();
            let mut direct = a.clone();
            direct.add_scaled_row(dst, src, c).unwrap();
            prop_assert!(
                via_e.equals(&direct),
                "E·A ≠ add_scaled(A)\n a={a:?}\n dst={dst} src={src} c={c}"
            );
        }

        // 筆記「可逆性」:每個基本矩陣都可逆,且逆是**同型**的基本矩陣(逆向列運算)。
        // 依可逆的定義驗 AB = BA = Iₙ 兩個方向。swap 的逆是它自己(換兩次 = 不換)。
        #[test]
        fn swap_is_its_own_inverse(i in 0usize..4, j in 0usize..4) {
            let e = Matrix::elementary_swap(4, i, j).unwrap();
            let id = Matrix::identity(4);
            prop_assert!(e.multiply(&e).unwrap().equals(&id), "E·E ≠ I\n i={i} j={j}");
        }

        // scale(c) 的逆是 scale(1/c)。1/c 是真實浮點(1/3 在 f64 不精確)——
        // 沿本 repo 慣例改用 approx_equals(_, 1e-9),不硬用精確 equals。
        #[test]
        fn scale_inverse_is_reciprocal_scale(i in 0usize..4, c in nonzero_int()) {
            let e = Matrix::elementary_scale(4, i, c).unwrap();
            let e_inv = Matrix::elementary_scale(4, i, 1.0 / c).unwrap();
            let id = Matrix::identity(4);
            prop_assert!(
                e.multiply(&e_inv).unwrap().approx_equals(&id, 1e-9),
                "E·E⁻¹ ≠ I\n i={i} c={c}"
            );
            prop_assert!(
                e_inv.multiply(&e).unwrap().approx_equals(&id, 1e-9),
                "E⁻¹·E ≠ I\n i={i} c={c}"
            );
        }

        // add_scaled(c) 的逆是 add_scaled(−c):加上去再扣回來。整數係數 → 精確 equals。
        #[test]
        fn add_scaled_inverse_negates_coefficient(
            dst in 0usize..4, step in 1usize..4, c in nonzero_int(),
        ) {
            let src = (dst + step) % 4;
            let e = Matrix::elementary_add_scaled(4, dst, src, c).unwrap();
            let e_inv = Matrix::elementary_add_scaled(4, dst, src, -c).unwrap();
            let id = Matrix::identity(4);
            prop_assert!(
                e.multiply(&e_inv).unwrap().equals(&id),
                "E·E⁻¹ ≠ I\n dst={dst} src={src} c={c}"
            );
            prop_assert!(
                e_inv.multiply(&e).unwrap().equals(&id),
                "E⁻¹·E ≠ I\n dst={dst} src={src} c={c}"
            );
        }

        // ── IMT(Theorem 2.6)系列:is_invertible 選了一個條件當實作,
        //    其餘等價條件在此互驗 —— 定理本身就是測試。 ──

        // IMT 條件 2 + 3:可逆 ⟺ RREF = Iₙ ⟺ rank = n。
        #[test]
        fn imt_rref_and_rank_agree(m in int_matrix(3, 3)) {
            let invertible = m.is_invertible(EPS);
            let rref_is_identity = m
                .reduced_row_echelon_form(EPS)
                .approx_equals(&Matrix::identity(3), EPS);
            let full_rank = m.rank(EPS) == 3;
            prop_assert_eq!(invertible, rref_is_identity, "可逆 ⟺ RREF = I 斷裂\n m={:?}", m);
            prop_assert_eq!(invertible, full_rank, "可逆 ⟺ rank = n 斷裂\n m={:?}", m);
        }

        // IMT 條件 6:可逆 ⟺ nullity = 0(一個自由變數都沒有)。
        #[test]
        fn imt_nullity_agrees(m in int_matrix(3, 3)) {
            prop_assert_eq!(
                m.is_invertible(EPS),
                m.nullity(EPS) == 0,
                "可逆 ⟺ nullity = 0 斷裂\n m={:?}", m
            );
        }

        // IMT 條件 7:可逆 ⟺ 行向量線性獨立(接上 independence 模組)。
        #[test]
        fn imt_column_independence_agrees(m in int_matrix(3, 3)) {
            let cols: Vec<Vector> = (0..3).map(|j| m.column(j).unwrap()).collect();
            prop_assert_eq!(
                m.is_invertible(EPS),
                is_linearly_independent(EPS, &cols),
                "可逆 ⟺ 行向量獨立 斷裂\n m={:?}", m
            );
        }

        // IMT 條件 5/8 的可計算版:植入解 x*(b := A·x*,系統必一致)→
        // 可逆 ⟺ solve 回 Unique(不可逆時有自由變數 → Infinite)。
        #[test]
        fn imt_unique_solution_agrees(m in int_matrix(3, 3), x_star in int_vector(3)) {
            let b = m.multiply_vector(&x_star).unwrap();
            let s = System::new(m.clone(), b).unwrap();
            let unique = matches!(s.solve(EPS), Solution::Unique(_));
            prop_assert_eq!(
                m.is_invertible(EPS),
                unique,
                "可逆 ⟺ 唯一解 斷裂\n m={:?}", m
            );
        }

        // 接回批 1:每個基本矩陣都可逆(筆記「可逆性」,改在判定器上驗)。
        #[test]
        fn elementary_matrices_are_invertible(
            i in 0usize..3, j in 0usize..3,
            dst in 0usize..3, step in 1usize..3, c in nonzero_int(),
        ) {
            let src = (dst + step) % 3;
            prop_assert!(Matrix::elementary_swap(3, i, j).unwrap().is_invertible(EPS));
            prop_assert!(Matrix::elementary_scale(3, i, c).unwrap().is_invertible(EPS));
            prop_assert!(
                Matrix::elementary_add_scaled(3, dst, src, c).unwrap().is_invertible(EPS)
            );
        }

        // 轉置保可逆性 —— Theorem 2.2「(Aᵀ)⁻¹ = (A⁻¹)ᵀ」的判定面(批 3 驗值的等式)。
        #[test]
        fn transpose_preserves_invertibility(m in int_matrix(3, 3)) {
            prop_assert_eq!(
                m.is_invertible(EPS),
                m.transpose().is_invertible(EPS),
                "可逆 ⟺ Aᵀ 可逆 斷裂\n m={:?}", m
            );
        }

        // 乘積的可逆性(筆記 Corollary 的判定面):A、B 都可逆 ⟺ AB 可逆。
        // ⟸ 方向也成立:任一個奇異,乘積必奇異(rank(AB) ≤ min(rank A, rank B))。
        #[test]
        fn product_invertible_iff_both_factors_are(
            a in int_matrix(3, 3), b in int_matrix(3, 3),
        ) {
            let product = a.multiply(&b).unwrap();
            prop_assert_eq!(
                a.is_invertible(EPS) && b.is_invertible(EPS),
                product.is_invertible(EPS),
                "都可逆 ⟺ 乘積可逆 斷裂\n a={:?}\n b={:?}", a, b
            );
        }
    }
}
