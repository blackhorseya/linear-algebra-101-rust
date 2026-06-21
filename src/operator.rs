//! 線性運算子的矩陣表示(Matrix Representations of Linear Operators)—— 單元 7-3
//! (講義 4.5 / 7.4)。
//!
//! **線性運算子**是定義域與對應域同一個空間的線性轉換 T: V → V(本章 V = ℝⁿ,
//! 故 T 由**方陣**誘導)。同一個運算子,換一組基底看,矩陣就不同 —— 這章談的就是
//! 「運算子」這個抽象物件,與「在某基底下的矩陣」這個具體表示之間的橋。
//!
//! # 核心:T 相對於基底 B 的矩陣 `[T]_B`
//!
//! 設 B = (b₁,…,bₙ) 是 V 的一組基底。把每個基底向量經 T 送出去、再讀它在**同一組
//! 基底 B** 下的座標,直放成 column,就得到 T 的 **B-矩陣**:
//!
//! ```text
//! [T]_B = [ [T(b₁)]_B | [T(b₂)]_B | … | [T(bₙ)]_B ].
//! ```
//!
//! 這正是 [`b_matrix`] —— 它把上一單元的 [`crate::coordinates`](求 `[x]_B`)接在
//! [`crate::Transformation`] 的施作(求 `T(bᵢ)`)之後,兩個既有積木的合成。
//!
//! # 本章只新增兩個函式,定理全走 laws
//!
//! 五道練習裡,只有兩個帶來**新計算**:[`b_matrix`](定義)與
//! [`reconstruct_standard_matrix`](由基底影像反求標準矩陣,`A = M·B⁻¹`)。其餘三條都是
//! 既有 `inverse` / `multiply` / `coordinates` 的組合,**沒有下游消費者的包裝函式不進公開
//! API**(見 repo 的 reuse 原則),改寫成 `mod laws` 的隨機試驗律,把定理對著兩個新函式與
//! 既有積木當場對帳:
//!
//! - **Theorem 4.12(座標轉換 / 相似)**:`[T]_B = B⁻¹AB`,其中 A 是 T 的標準矩陣、B 的
//!   **行**就是基底。把 [`b_matrix`] 的**定義路徑**與 `B⁻¹AB` 的**閉式路徑**對帳
//!   (law `b_matrix_equals_inverse_a_b`)。於是 `[T]_B` 與 A **相似**:同一運算子在不同
//!   基底下的兩個矩陣。
//! - **相似是對稱關係**:若 `B = P⁻¹AP` 則 `A = P B P⁻¹`(見證矩陣換成 P⁻¹)——
//!   law `similarity_is_symmetric`,純既有運算,stub 階段就綠(reuse 的見證)。
//! - **Theorem 7.10(映射性質)**:`[T(v)]_B = [T]_B · [v]_B` —— 抽象空間裡的線性運算,
//!   完全可由座標向量與矩陣乘法在 ℝⁿ 中模擬(law `b_matrix_maps_coordinates`)。
//! - **運算子由基底影像唯一決定**:給 `{bᵢ}` 與 `{T(bᵢ)}` 即可反求 A
//!   (law `reconstruct_recovers_operator`);標準基底時這恰好退回既有
//!   [`crate::standard_matrix`](law `reconstruct_with_standard_basis_is_standard_matrix`)。

use crate::{LinAlgError, Matrix, Vector};

/// T 相對於有序基底 `basis` 的矩陣表示 `[T]_B` —— 各 column 是基底向量影像的座標
/// `[T(bᵢ)]_B`。
///
/// 收泛型 `F: Fn(&Vector) -> Vector`(沿 [`crate::verify_linearity`] /
/// [`crate::standard_matrix`] 的慣例:定義適用於**任意**映射,Theorem 4.12 才回頭說
/// 「矩陣誘導的那種」也能用 `B⁻¹AB` 算)。
///
/// 失敗模式原封由 [`crate::coordinates`] 傳上來:`basis` 不是 `T(bᵢ)` 所在空間的一組
/// 基底時(不生成或相依),座標未定義 → [`LinAlgError::NotABasis`]。
///
/// # 你的實作(單元 7-3 練習 1)
///
/// 對每個 `bᵢ`:算 `T(bᵢ)`,再用 [`crate::coordinates`]`(epsilon, &T(bᵢ), basis)` 求
/// `[T(bᵢ)]_B`(用 `?` 傳播錯誤)。這些座標向量「直放」成 `[T]_B` 的各 column —— 組裝法
/// 可仿 [`crate::standard_matrix`]:逐列收 `A[i][j] = coords_j[i]` 再 [`Matrix::from_rows`],
/// 或 `from_rows(每個座標當一列).transpose()`。
pub fn b_matrix<F>(epsilon: f64, t: F, basis: &[Vector]) -> Result<Matrix, LinAlgError>
where
    F: Fn(&Vector) -> Vector,
{
    basis
        .iter()
        .map(|bi| {
            let t_bi = t(bi);
            crate::coordinates(epsilon, &t_bi, basis)
        })
        .collect::<Result<Vec<Vector>, LinAlgError>>()
        .map(|coords| {
            let rows: Vec<Vec<f64>> = (0..basis.len())
                .map(|i| coords.iter().map(|c| c.entries()[i]).collect())
                .collect();
            Matrix::from_rows(rows)
        })
}

/// 由「基底 + 其影像」反求運算子的標準矩陣 A:要求 `A·bᵢ = imagesᵢ` 對所有 i 成立。
///
/// 把基底排成矩陣 B(各 column 一個 `bᵢ`)、影像排成 M(各 column 一個 `imagesᵢ`),則
/// `A·B = M`,於是 **`A = M·B⁻¹`** —— 這也說明了「線性運算子由它在一組基底上的影像**唯一**
/// 決定」(B 可逆 ⟹ A 唯一)。
///
/// 失敗模式:`basis` 與 `images` 數量不符 → [`LinAlgError::CountMismatch`];空輸入 →
/// [`LinAlgError::EmptyInput`];B 不可逆(行不成基底,含非方陣)→ 原封由
/// [`Matrix::inverse`] 傳上來([`LinAlgError::NotInvertible`] / [`LinAlgError::NotSquare`])。
///
/// # 你的實作(單元 7-3 練習 5)
///
/// 守門數量與空輸入後,把 `basis` / `images` 各自組成矩陣 B / M(每個向量一個 column,
/// 組裝法同 [`b_matrix`]),再回傳 `M.multiply(&B.inverse(epsilon)?)`。
pub fn reconstruct_standard_matrix(
    epsilon: f64,
    basis: &[Vector],
    images: &[Vector],
) -> Result<Matrix, LinAlgError> {
    if basis.len() != images.len() {
        return Err(LinAlgError::CountMismatch);
    }
    if basis.is_empty() {
        return Err(LinAlgError::EmptyInput);
    }

    let b_rows: Vec<Vec<f64>> = (0..basis.len())
        .map(|i| basis.iter().map(|b| b.entries()[i]).collect())
        .collect();
    let m_rows: Vec<Vec<f64>> = (0..images.len())
        .map(|i| images.iter().map(|img| img.entries()[i]).collect())
        .collect();

    let b = Matrix::from_rows(b_rows);
    let m = Matrix::from_rows(m_rows);

    m.multiply(&b.inverse(epsilon)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RREF 與 inverse 都帶捨入,故座標 / 矩陣以小容差判定。
    const EPS: f64 = 1e-9;

    fn v(data: Vec<f64>) -> Vector {
        Vector::from_vec(data)
    }

    /// x 軸反射的標準矩陣 A = diag(1, −1)。
    fn reflect_x() -> Matrix {
        Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, -1.0]])
    }

    /// 練習 1 題目原例:在**標準基底** E = {e₁, e₂} 下,x 軸反射的 B-矩陣就是它的標準矩陣
    /// 本身 —— `[T]_E = A`。標準基底下「換座標」是 identity,故 B-矩陣與標準矩陣重合。
    #[test]
    fn b_matrix_in_standard_basis_is_the_standard_matrix() {
        let reflect = reflect_x();
        let basis = vec![
            Vector::standard(2, 0).unwrap(),
            Vector::standard(2, 1).unwrap(),
        ];
        let got = b_matrix(EPS, |x| reflect.multiply_vector(x).unwrap(), &basis)
            .expect("標準基底是合法基底");
        assert!(
            got.approx_equals(&reflect, EPS),
            "[T]_E 應 = A = diag(1,−1),得 {got:?}"
        );
    }

    /// 同一個運算子,換一組基底,矩陣就不同 —— 但**相似**。x 軸反射在傾斜基底
    /// B = {(1,1), (1,−1)} 下:T(1,1) = (1,−1) = b₂、T(1,−1) = (1,1) = b₁,故 `[T(b₁)]_B = (0,1)`、
    /// `[T(b₂)]_B = (1,0)`,B-矩陣是**交換矩陣** [[0,1],[1,0]]。它與標準矩陣 diag(1,−1) 不相等
    /// (`approx_equals` 為 false),卻是同一個反射 —— 相似但不相等,正是「矩陣表示依基底而變」。
    #[test]
    fn b_matrix_in_tilted_basis_is_similar_but_not_equal() {
        let reflect = reflect_x();
        let basis = vec![v(vec![1.0, 1.0]), v(vec![1.0, -1.0])];
        let got = b_matrix(EPS, |x| reflect.multiply_vector(x).unwrap(), &basis)
            .expect("{(1,1),(1,−1)} 是 ℝ² 的基底");

        let swap = Matrix::from_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        assert!(
            got.approx_equals(&swap, EPS),
            "傾斜基底下 [T]_B 應為交換矩陣 [[0,1],[1,0]],得 {got:?}"
        );
        assert!(
            !got.approx_equals(&reflect, EPS),
            "同一運算子、不同基底:[T]_B 不該等於標準矩陣 A(相似但不相等)"
        );
    }

    /// 練習 5 具體案例:給基底與其影像,反求標準矩陣 A。
    /// 取 A = [[2,1],[0,3]]、basis = {(1,0),(1,1)},則影像 = {A·(1,0), A·(1,1)} = {(2,0),(3,3)}。
    /// `reconstruct` 應由這兩對還原出 A;再驗 `A·bᵢ = imagesᵢ`。
    #[test]
    fn reconstruct_recovers_known_operator() {
        let basis = vec![v(vec![1.0, 0.0]), v(vec![1.0, 1.0])];
        let images = vec![v(vec![2.0, 0.0]), v(vec![3.0, 3.0])];
        let got = reconstruct_standard_matrix(EPS, &basis, &images).expect("基底可逆");

        let want = Matrix::from_rows(vec![vec![2.0, 1.0], vec![0.0, 3.0]]);
        assert!(
            got.approx_equals(&want, EPS),
            "重建 A 應為 [[2,1],[0,3]],得 {got:?}"
        );

        // 定義性質:重建出的 A 必須把每個基底向量送到指定影像。
        for (bi, img) in basis.iter().zip(&images) {
            let mapped = got.multiply_vector(bi).unwrap();
            assert!(
                mapped.approx_equals(img, EPS),
                "A·{bi:?} 應 = {img:?},得 {mapped:?}"
            );
        }
    }

    /// 數量不符就反求不出運算子 —— 基底與影像必須一一對應。錯誤是值,不 panic。
    #[test]
    fn reconstruct_rejects_count_mismatch() {
        let basis = vec![v(vec![1.0, 0.0]), v(vec![1.0, 1.0])];
        let images = vec![v(vec![2.0, 0.0])]; // 只有一個影像
        assert_eq!(
            reconstruct_standard_matrix(EPS, &basis, &images).unwrap_err(),
            LinAlgError::CountMismatch
        );
    }
}

#[cfg(test)]
mod laws {
    use super::*;
    use crate::{coordinates, is_basis, standard_matrix};
    use proptest::prelude::*;

    /// 產生 `n×n`、元素為 [-10, 10] 整數的方陣(f64 下精確)—— 幾乎必為 full rank,其行
    /// 就構成 ℝ^n 的一組基底。(同 `coordinates.rs` 的同名策略。)
    fn int_square_matrix(n: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-10i64..=10, n), n).prop_map(|grid| {
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
        /// **Theorem 4.12**:`[T]_B = B⁻¹AB`。把 [`b_matrix`] 的**定義路徑**(對基底取像、
        /// 求座標、組 column)與**閉式路徑**(B⁻¹AB)當場對帳 —— B 的行就是基底、A 是 T 的標準
        /// 矩陣。兩條獨立路得同一個矩陣,既證 `b_matrix` 正確,也演出「`[T]_B` 與 A 相似」。
        #[test]
        fn b_matrix_equals_inverse_a_b(
            (b, a) in (1usize..=4).prop_flat_map(|dim| {
                (int_square_matrix(dim), int_square_matrix(dim))
            }),
        ) {
            const SOLVE_EPS: f64 = 1e-9;
            const COMPARE_EPS: f64 = 1e-6;

            let dim = b.cols();
            let basis: Vec<Vector> = (0..dim).map(|j| b.column(j).unwrap()).collect();
            prop_assume!(is_basis(SOLVE_EPS, dim, &basis));

            // 定義路徑:[T]_B,各 column = [A·bⱼ]_B。
            let via_def = b_matrix(
                SOLVE_EPS,
                |x| a.multiply_vector(x).expect("方陣作用於同維向量"),
                &basis,
            )
            .expect("行成基底 ⟹ b_matrix 不該出錯");

            // 閉式路徑(Theorem 4.12):B⁻¹AB。
            let via_formula = b
                .inverse(SOLVE_EPS)
                .expect("基底 ⟹ B 可逆")
                .multiply(&a)
                .expect("同維方陣可乘")
                .multiply(&b)
                .expect("同維方陣可乘");

            prop_assert!(
                via_def.approx_equals(&via_formula, COMPARE_EPS),
                "[T]_B:定義 = {via_def:?},B⁻¹AB = {via_formula:?}"
            );
        }

        /// **Theorem 7.10(映射性質)**:`[T(v)]_B = [T]_B · [v]_B`。抽象空間裡「先作用 T、再
        /// 求座標」,等於「先求座標、再左乘 B-矩陣」—— 線性運算可完全由座標向量 + 矩陣乘法在
        /// ℝⁿ 中模擬。這是「B-矩陣」這個表示之所以有用的根本理由。
        #[test]
        fn b_matrix_maps_coordinates(
            (b, a, x) in (1usize..=4).prop_flat_map(|dim| {
                (int_square_matrix(dim), int_square_matrix(dim), int_vector(dim))
            }),
        ) {
            const SOLVE_EPS: f64 = 1e-9;
            const COMPARE_EPS: f64 = 1e-6;

            let dim = b.cols();
            let basis: Vec<Vector> = (0..dim).map(|j| b.column(j).unwrap()).collect();
            prop_assume!(is_basis(SOLVE_EPS, dim, &basis));

            let bm = b_matrix(
                SOLVE_EPS,
                |y| a.multiply_vector(y).expect("方陣作用於同維向量"),
                &basis,
            )
            .expect("行成基底 ⟹ b_matrix 不該出錯");

            // 左式 [T(v)]_B:先作用 T = Av,再求座標。
            let tv = a.multiply_vector(&x).expect("方陣作用於同維向量");
            let lhs = coordinates(SOLVE_EPS, &tv, &basis).expect("基底 ⟹ 座標存在");

            // 右式 [T]_B · [v]_B:先求座標,再左乘 B-矩陣。
            let vb = coordinates(SOLVE_EPS, &x, &basis).expect("基底 ⟹ 座標存在");
            let rhs = bm.multiply_vector(&vb).expect("[T]_B 為方陣,維度相容");

            prop_assert!(
                lhs.approx_equals(&rhs, COMPARE_EPS),
                "[T(v)]_B = {lhs:?},[T]_B·[v]_B = {rhs:?}"
            );
        }

        /// **相似是對稱關係**:若 A 相似於 B(見證矩陣 P,`B = P⁻¹AP`),則 B 也相似於 A
        /// (見證矩陣 P⁻¹,`A = (P⁻¹)⁻¹·B·P⁻¹ = P·B·P⁻¹`)。純既有 inverse/multiply 組合,
        /// 不依賴本章新函式 —— **stub 階段即綠**,是「相似性的內容已由既有積木表達、無須包裝
        /// 函式」的見證。
        #[test]
        fn similarity_is_symmetric(
            (a, p) in (1usize..=4).prop_flat_map(|dim| {
                (int_square_matrix(dim), int_square_matrix(dim))
            }),
        ) {
            const SOLVE_EPS: f64 = 1e-9;
            const COMPARE_EPS: f64 = 1e-6;

            let dim = p.cols();
            let basis: Vec<Vector> = (0..dim).map(|j| p.column(j).unwrap()).collect();
            prop_assume!(is_basis(SOLVE_EPS, dim, &basis)); // ⟹ P 可逆

            let p_inv = p.inverse(SOLVE_EPS).expect("基底 ⟹ P 可逆");

            // A 相似於 B,見證 P:B = P⁻¹AP。
            let bm = p_inv
                .multiply(&a)
                .expect("同維方陣可乘")
                .multiply(&p)
                .expect("同維方陣可乘");

            // 對稱:B 也相似於 A,見證 P⁻¹ —— A = P·B·P⁻¹ 應還原 A。
            let recovered = p
                .multiply(&bm)
                .expect("同維方陣可乘")
                .multiply(&p_inv)
                .expect("同維方陣可乘");

            prop_assert!(
                recovered.approx_equals(&a, COMPARE_EPS),
                "P·(P⁻¹AP)·P⁻¹ 應 = A,得 {recovered:?}"
            );
        }

        /// **運算子由基底影像唯一決定**:取隨機運算子 A 與基底 B,造出影像 `imagesᵢ = A·bᵢ`,
        /// 則 [`reconstruct_standard_matrix`] 必還原出 A。B 可逆保證了「唯一」—— 同一組影像反求
        /// 不出第二個 A。
        #[test]
        fn reconstruct_recovers_operator(
            (b, a) in (1usize..=4).prop_flat_map(|dim| {
                (int_square_matrix(dim), int_square_matrix(dim))
            }),
        ) {
            const SOLVE_EPS: f64 = 1e-9;
            const COMPARE_EPS: f64 = 1e-6;

            let dim = b.cols();
            let basis: Vec<Vector> = (0..dim).map(|j| b.column(j).unwrap()).collect();
            prop_assume!(is_basis(SOLVE_EPS, dim, &basis));

            let images: Vec<Vector> = basis
                .iter()
                .map(|bi| a.multiply_vector(bi).expect("方陣作用於同維向量"))
                .collect();

            let recovered = reconstruct_standard_matrix(SOLVE_EPS, &basis, &images)
                .expect("基底可逆 ⟹ 可反求");

            prop_assert!(
                recovered.approx_equals(&a, COMPARE_EPS),
                "由 {{A·bᵢ}} 反求應還原 A,得 {recovered:?}"
            );
        }

        /// **標準基底時退回既有 [`crate::standard_matrix`]**:當基底就是 E = {eᵢ} 時,
        /// `reconstruct_standard_matrix(E, {A·eᵢ})` = A —— 因為 `A·eᵢ` 恰是 A 的第 i 行,
        /// 反求即「把各行擺回去」。這正是單元 5-2 `standard_matrix`(對 T 做標準基底取樣)做的
        /// 事,故兩者對同一運算子應給同一個矩陣。把本章新函式接回轉換章。
        #[test]
        fn reconstruct_with_standard_basis_is_standard_matrix(
            a in (1usize..=4).prop_flat_map(int_square_matrix),
        ) {
            const SOLVE_EPS: f64 = 1e-9;
            const COMPARE_EPS: f64 = 1e-6;

            let dim = a.cols();
            let std_basis: Vec<Vector> =
                (0..dim).map(|i| Vector::standard(dim, i).unwrap()).collect();
            let images: Vec<Vector> = std_basis
                .iter()
                .map(|e| a.multiply_vector(e).expect("方陣作用於同維向量"))
                .collect();

            let via_reconstruct = reconstruct_standard_matrix(SOLVE_EPS, &std_basis, &images)
                .expect("標準基底可逆");
            let via_standard =
                standard_matrix(dim, |x| a.multiply_vector(x).expect("方陣作用於同維向量"))
                    .expect("dim ≥ 1");

            prop_assert!(
                via_reconstruct.approx_equals(&a, COMPARE_EPS),
                "標準基底反求應還原 A,得 {via_reconstruct:?}"
            );
            prop_assert!(
                via_reconstruct.approx_equals(&via_standard, COMPARE_EPS),
                "reconstruct = {via_reconstruct:?},standard_matrix = {via_standard:?}"
            );
        }
    }
}
