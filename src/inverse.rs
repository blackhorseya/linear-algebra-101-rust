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
    use proptest::prelude::*;

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
    }
}
