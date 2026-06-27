//! 特徵值與特徵向量(Eigenvalues & Eigenvectors)—— 單元 8-1(講義 5.1)。
//!
//! 這是課程進入「進階主題」的第一章。前面所有轉換都把整個空間搬來搬去;這裡問一個
//! 更尖銳的問題:對一個運算子 T(方陣 A),**哪些非零向量 v 只被「伸縮」、方向不變**?
//! 即存在純量 λ 使
//!
//! > **A·v = λ·v**(v ≠ 0)
//!
//! 這樣的 v 叫**特徵向量(eigenvector)**、λ 叫對應的**特徵值(eigenvalue)**。把
//! A·v = λv 移項成 **(A − λI)·v = 0**,特徵向量就是 **Null(A − λI) 裡的非零向量** ——
//! 一句話把這整章接回子空間機器:**特徵空間 Eλ = Null(A − λI)**。
//!
//! ## 這章新增什麼(reuse 分析:5 題 → 2 新函式 + 1 新子空間積木 + 定理走 laws)
//!
//! 沿用本專案慣例(見 `prefer-reuse-and-laws-over-new-api`):只有「帶來新計算」的
//! 才進公開 API,定理寫成 `mod laws` 的隨機驗證。
//!
//! - **練習 1** [`is_eigenpair`]:`A·v = λv` 且 v ≠ 0 的定義性檢查 —— 全章的判準基石。
//! - **練習 3** [`characteristic_matrix`]:`A − λI` —— 把「找特徵向量」翻譯成「找零空間」的
//!   閘門矩陣。
//! - **練習 4** [`eigenspace_basis`]:`Eλ = Null(A − λI)` 的一組基底 —— 串接練習 3 與
//!   **本章真正的新演算法** [`Transformation::null_space_basis`](crate::Transformation::null_space_basis)
//!   (零空間基底萃取器)。
//!   筆記說「沿用 6-2 的 NullSpace 邏輯」,但本 repo 的 6-2 只刻了
//!   [`null_space_contains`](crate::Transformation::null_space_contains)(會員判定)與
//!   [`nullity`](crate::Matrix::nullity)(只數維度)—— **從未有基底萃取器**,故它在
//!   `subspace.rs` 補上,正好補齊「矩陣子空間基底三兄弟」:`range_basis`(Col A)、
//!   `row_space_basis`(Row A)、**`null_space_basis`(Null A)**。
//! - **練習 5** [`has_real_eigenvalues_2x2`]:2×2 特徵多項式判別式 —— 90° 旋轉沒有實特徵值
//!   的幾何直觀(正式算法在 5.2 節)。
//! - **練習 2**(運算子版特徵檢查)**不進 API**:運算子的特徵向量 = 其標準矩陣的特徵向量,
//!   這是定理不是新計算 —— 經 [`standard_matrix`](crate::standard_matrix) 橋接 closure 與
//!   矩陣,寫成 `mod laws` 的 `operator_eigen_agrees_with_standard_matrix`。
//!
//! 與 `transformation` / `subspace` 同款佈局:本模組碰不到 private 的 `data`,一律走
//! public API。

use crate::{LinAlgError, Matrix, Transformation, Vector};

/// **練習 1 —— 特徵對(eigenpair)驗證器**:v 是否為 A 對應特徵值 λ 的特徵向量,即
/// **A·v = λv** 且 **v ≠ 0**(容差 `epsilon` 內)。
///
/// 兩個非顯而易見的判準,缺一不可:
/// - **v 必須非零**:依定義特徵向量不能是零向量(否則 A·0 = λ·0 對**任何** λ 都成立,
///   λ 就失去意義)。零向量 → `false`。
/// - **維度 / 形狀不合 → `false`**:A 非方陣、或 v 不在 A 的定義域(`A·v` 與 `λv` 維度對不上)
///   時,談不上特徵對(沿 [`null_space_contains`](crate::Transformation::null_space_contains)
///   / [`verify_linearity`](crate::verify_linearity) 的 bool 述詞慣例 —— 不在同一空間就回
///   `false`,不 panic)。
///
/// 實作提示:[`Matrix::multiply_vector`] 算 `A·v`(回 `Result`,維度檢查隨之繼承 ——
/// `Err` 恰好折成 `false`);λv 是 [`Vector::scale`];兩者用
/// [`Vector::approx_equals`]`(_, epsilon)` 比;非零用 [`Vector::is_approx_zero`]`(epsilon)`
/// 的反面。先擋零向量,再算、再比。
pub fn is_eigenpair(epsilon: f64, a: &Matrix, v: &Vector, lambda: f64) -> bool {
    if v.is_approx_zero(epsilon) {
        return false;
    }
    match a.multiply_vector(v) {
        Ok(av) => av.approx_equals(&v.scale(lambda), epsilon),
        Err(_) => false,
    }
}

/// **練習 3 —— 特徵閘門矩陣**:回傳 **M = A − λI**(`I` 為同階單位矩陣)。
///
/// 這是全章的轉軸:`A·v = λv ⟺ (A − λI)·v = 0`,於是「找特徵向量」變成「找
/// `M = A − λI` 的零空間非零向量」。**只動主對角線**(每個 `A[i][i]` 減 λ),其餘元素照搬。
///
/// 回 `Result`:`A − λI` 只對**方陣**有定義(λI 要同階),非方陣 → [`LinAlgError::NotSquare`]
/// (沿 [`Matrix::power`](crate::Matrix::power) 的先例:同樣「需要方陣」的運算)。
///
/// 實作提示:守住 [`Matrix::is_square`] 後,`λI` = [`Matrix::identity`]`(n)` 再
/// [`Matrix::scalar_multiply`]`(λ)`;`A − λI` 可寫成 `A + (−λ)I`,用 [`Matrix::add`]
/// (兩者皆 n×n,守門後相加不會失敗 —— 同 `power` 的 `expect` 手法,別假裝它會錯)。
pub fn characteristic_matrix(a: &Matrix, lambda: f64) -> Result<Matrix, LinAlgError> {
    if !a.is_square() {
        return Err(LinAlgError::NotSquare {
            rows: a.rows(),
            cols: a.cols(),
        });
    }
    let n = a.rows();
    let identity = Matrix::identity(n).scalar_multiply(lambda);
    Ok(a.add(&identity.scalar_multiply(-1.0))
        .expect("characteristic_matrix: A, λI 同階,加法不會失敗;若失敗,是程式 bug"))
}

/// **練習 4 —— 特徵空間基底**:`Eλ = Null(A − λI)` 的一組基底向量。
///
/// 特徵空間是「對應特徵值 λ 的所有特徵向量,再加上零向量」——它恰好是
/// `M = A − λI` 的零空間(故是子空間)。回傳的每個向量都滿足 `A·v = λv`、且彼此線性獨立
/// (由零空間基底演算法保證);其**個數 = `Eλ` 的維度 = λ 的幾何重數 = nullity(A − λI)**。
///
/// λ **不是** A 的特徵值時,`Eλ = {0}` → **回空 `Vec`**(無非零特徵向量,維度 0)。
///
/// 實作提示:這題幾乎是一行的合縫 —— [`characteristic_matrix`]`(a, lambda)?` 取得
/// `M`,包成 [`Transformation::new`](crate::Transformation::new),呼叫本章新增的
/// [`null_space_basis`](crate::Transformation::null_space_basis)`(epsilon)`。
/// `Result` 沿 `characteristic_matrix` 的非方陣失敗一路傳上來(`?`)。
pub fn eigenspace_basis(epsilon: f64, a: &Matrix, lambda: f64) -> Result<Vec<Vector>, LinAlgError> {
    let m = characteristic_matrix(a, lambda)?;
    let t = Transformation::new(m);
    Ok(t.null_space_basis(epsilon))
}

/// **練習 5 —— 2×2 是否有實特徵值**:用特徵多項式 `t² − tr(A)·t + det(A) = 0` 的**判別式**
/// 判定(預習 5.2 的幾何直觀)。
///
/// 對 `A = [[a, b], [c, d]]`,判別式 `Δ = tr² − 4·det = (a − d)² + 4bc`。**Δ ≥ 0 ⟺ 有實特徵值**
/// (Δ < 0 時兩特徵值是共軛複數 —— 如 90° 旋轉把每個向量轉開、沒有方向被保留)。
///
/// 驗收觀念:**對稱矩陣**(b = c)必有實特徵值 —— 此時 `Δ = (a − d)² + 4b² ≥ 0` 恆成立
/// (見 laws `symmetric_2x2_always_has_real_eigenvalues`)。
///
/// 非 2×2 輸入 → `false`(本判別式只對 2×2 成立,沿 bool 述詞「形狀不合即 false」慣例;
/// 不用 epsilon —— 旋轉的 Δ 明顯為負、對稱的明顯 ≥ 0,邊界 `Δ == 0` 視為有實根)。
///
/// 實作提示:守住 `rows() == 2 && cols() == 2` 後,用 [`Matrix::row`] 取出 a, b, c, d,
/// 直接算 `(a − d)² + 4bc >= 0`(比 `tr² − 4det` 少一次相減、數值更穩)。
pub fn has_real_eigenvalues_2x2(a: &Matrix) -> bool {
    if a.rows() != 2 || a.cols() != 2 {
        return false;
    }
    let row0 = a.row(0).unwrap();
    let row1 = a.row(1).unwrap();
    let a = row0[0];
    let b = row0[1];
    let c = row1[0];
    let d = row1[1];
    (a - d).powi(2) + 4.0 * b * c >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // ---- 練習 1:is_eigenpair ----

    /// 筆記題目原例:A·v = 4v —— v 是 λ = 4 的特徵向量。
    #[test]
    fn is_eigenpair_accepts_textbook_example() {
        let a = Matrix::from_rows(vec![
            vec![5.0, 2.0, 1.0],
            vec![-2.0, 1.0, -1.0],
            vec![2.0, 2.0, 4.0],
        ]);
        let v = Vector::from_vec(vec![1.0, -1.0, 1.0]);
        // A·v = [5−2+1, −2−1−1, 2−2+4] = [4, −4, 4] = 4·v
        assert!(is_eigenpair(EPS, &a, &v, 4.0));
    }

    /// 同一個 v、錯的 λ → 不是特徵對。
    #[test]
    fn is_eigenpair_rejects_wrong_lambda() {
        let a = Matrix::from_rows(vec![
            vec![5.0, 2.0, 1.0],
            vec![-2.0, 1.0, -1.0],
            vec![2.0, 2.0, 4.0],
        ]);
        let v = Vector::from_vec(vec![1.0, -1.0, 1.0]);
        assert!(!is_eigenpair(EPS, &a, &v, 3.0), "A·v = 4v ≠ 3v");
    }

    /// 零向量永遠不是特徵向量(定義排除)—— 即使 A·0 = λ·0 對任何 λ 成立。
    #[test]
    fn is_eigenpair_rejects_zero_vector() {
        let a = Matrix::identity(3);
        assert!(!is_eigenpair(EPS, &a, &Vector::new(3), 1.0));
    }

    /// 維度不合(v 不在 domain)→ false,不 panic。
    #[test]
    fn is_eigenpair_rejects_dimension_mismatch() {
        let a = Matrix::identity(3);
        assert!(!is_eigenpair(
            EPS,
            &a,
            &Vector::from_vec(vec![1.0, 0.0]),
            1.0
        ));
    }

    // ---- 練習 3:characteristic_matrix ----

    /// 筆記題目原例:A − 3I —— 只有主對角線被減 3。
    #[test]
    fn characteristic_matrix_subtracts_lambda_on_diagonal() {
        let a = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        let m = characteristic_matrix(&a, 3.0).unwrap();
        assert!(m.equals(&Matrix::from_rows(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, -2.0, 2.0],
            vec![0.0, 2.0, -2.0],
        ])));
    }

    /// 非方陣沒有 A − λI(λI 無從同階)→ NotSquare。
    #[test]
    fn characteristic_matrix_rejects_non_square() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // 2×3
        assert_eq!(
            characteristic_matrix(&a, 1.0).unwrap_err(),
            LinAlgError::NotSquare { rows: 2, cols: 3 }
        );
    }

    // ---- 練習 4:eigenspace_basis ----

    /// 筆記題目原例:B 對 λ = 3 的特徵空間是二維(幾何重數 2)。
    /// 不寫死「哪兩個向量」(基底表示不唯一),改驗:個數 = 2、每個都是 λ = 3 的特徵對。
    #[test]
    fn eigenspace_basis_recovers_two_dimensional_eigenspace() {
        let b = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        let basis = eigenspace_basis(EPS, &b, 3.0).unwrap();
        assert_eq!(basis.len(), 2, "E₃ 是二維(λ = 3 幾何重數 2)");
        for v in &basis {
            assert!(
                is_eigenpair(EPS, &b, v, 3.0),
                "基底向量必為 λ = 3 的特徵向量"
            );
        }
    }

    /// λ 不是特徵值 → 特徵空間只有零向量 → 空基底。
    #[test]
    fn eigenspace_basis_is_empty_for_non_eigenvalue() {
        let b = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        assert!(
            eigenspace_basis(EPS, &b, 7.0).unwrap().is_empty(),
            "7 不是 B 的特徵值,E₇ = {{0}}"
        );
    }

    // ---- 練習 5:has_real_eigenvalues_2x2 ----

    /// 90° 旋轉沒有實特徵值(把每個方向都轉開)—— Δ = (0−0)² + 4·(−1)·1 = −4 < 0。
    #[test]
    fn rotation_90_has_no_real_eigenvalues() {
        let rot = Matrix::from_rows(vec![vec![0.0, -1.0], vec![1.0, 0.0]]);
        assert!(!has_real_eigenvalues_2x2(&rot));
    }

    /// 對稱矩陣必有實特徵值;非均勻縮放(對角)也有(Δ ≥ 0)。
    #[test]
    fn symmetric_and_diagonal_have_real_eigenvalues() {
        let symmetric = Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        let scaling = Matrix::from_rows(vec![vec![2.0, 0.0], vec![0.0, 3.0]]);
        assert!(has_real_eigenvalues_2x2(&symmetric));
        assert!(has_real_eigenvalues_2x2(&scaling));
    }

    /// 非 2×2 → false(判別式只對 2×2 成立)。
    #[test]
    fn has_real_eigenvalues_2x2_rejects_non_2x2() {
        assert!(!has_real_eigenvalues_2x2(&Matrix::identity(3)));
    }
}

/// 教材定理的隨機驗證(「for all」形式)。整數策略配精確 / 1e-9 容差(整數加減乘在 f64 下
/// 精確;唯有 `eigenspace_basis` 經 RREF 帶除法,容差吸收捨入殘差)。
#[cfg(test)]
mod laws {
    use super::*;
    use crate::{is_linearly_independent, standard_matrix};
    use proptest::prelude::*;

    const EPS: f64 = 1e-9;

    /// 固定 n×n、元素為小整數的方陣。
    fn int_square_matrix(n: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-5i64..=5, n), n).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            )
        })
    }

    /// 隨機階數(1..=4)的整數方陣。
    fn int_square_any() -> impl Strategy<Value = Matrix> {
        (1usize..=4).prop_flat_map(int_square_matrix)
    }

    /// 對角矩陣 diag(d₀…dₙ₋₁):eᵢ 是已知的特徵向量(特徵值 dᵢ),laws 用來「植入」特徵對。
    fn diagonal_matrix(diag: &[f64]) -> Matrix {
        let n = diag.len();
        Matrix::from_rows(
            (0..n)
                .map(|i| (0..n).map(|j| if i == j { diag[i] } else { 0.0 }).collect())
                .collect(),
        )
    }

    /// 對角線(小整數)連同一個合法索引 `i < n` —— 植入「eᵢ 是 λ = dᵢ 的特徵向量」。
    fn diagonal_with_index() -> impl Strategy<Value = (Vec<f64>, usize)> {
        (1usize..=4).prop_flat_map(|n| {
            (
                prop::collection::vec((-5i64..=5).prop_map(|v| v as f64), n),
                0..n,
            )
        })
    }

    /// 對稱 2×2 矩陣 [[a, b], [b, d]](小整數)。
    fn symmetric_2x2() -> impl Strategy<Value = Matrix> {
        (-9i64..=9, -9i64..=9, -9i64..=9).prop_map(|(a, b, d)| {
            Matrix::from_rows(vec![vec![a as f64, b as f64], vec![b as f64, d as f64]])
        })
    }

    proptest! {
        // 練習 1(植入版):對角 D 的 eᵢ 必是 λ = dᵢ 的特徵向量(D·eᵢ = dᵢ·eᵢ 精確成立)。
        #[test]
        fn diagonal_standard_vectors_are_eigenpairs((diag, i) in diagonal_with_index()) {
            let d = diagonal_matrix(&diag);
            let ei = Vector::standard(diag.len(), i).unwrap();
            prop_assert!(is_eigenpair(EPS, &d, &ei, diag[i]));
        }

        // 練習 1(否定):零向量對**任何** A、任何 λ 都不是特徵對。
        #[test]
        fn zero_vector_is_never_an_eigenvector(a in int_square_any(), lambda in -5i64..=5) {
            let n = a.cols();
            prop_assert!(!is_eigenpair(EPS, &a, &Vector::new(n), lambda as f64));
        }

        // 練習 2(運算子 = 標準矩陣特徵,經 Theorem 2.9 橋接):把對角 D 當運算子 closure,
        // standard_matrix 取樣重建出 A == D,於是「運算子的特徵對」與「其標準矩陣的特徵對」
        // 是同一回事 —— 運算子版檢查不必另立函式。
        #[test]
        fn operator_eigen_agrees_with_standard_matrix((diag, i) in diagonal_with_index()) {
            let d = diagonal_matrix(&diag);
            let a = standard_matrix(diag.len(), |x| d.multiply_vector(x).unwrap()).unwrap();
            prop_assert!(a.approx_equals(&d, EPS), "Thm 2.9:取樣重建運算子的標準矩陣");
            let ei = Vector::standard(diag.len(), i).unwrap();
            prop_assert!(is_eigenpair(EPS, &a, &ei, diag[i]));
        }

        // 練習 3:characteristic_matrix 只動主對角線 —— 對角格減 λ、其餘照搬。
        #[test]
        fn characteristic_matrix_touches_only_diagonal(a in int_square_any(), lambda in -5i64..=5) {
            let n = a.cols();
            let m = characteristic_matrix(&a, lambda as f64).unwrap();
            prop_assert_eq!((m.rows(), m.cols()), (n, n), "形狀不變");
            for i in 0..n {
                for j in 0..n {
                    let expected = a.row(i).unwrap()[j] - if i == j { lambda as f64 } else { 0.0 };
                    prop_assert!((m.row(i).unwrap()[j] - expected).abs() <= EPS);
                }
            }
        }

        // 練習 4(驗收條件):特徵空間基底的每個向量都是該 λ 的特徵向量、且彼此獨立。
        // 用對角 D + λ = dᵢ 植入,保證 Eλ 非平凡。
        #[test]
        fn eigenspace_basis_vectors_are_independent_eigenvectors(
            (diag, i) in diagonal_with_index(),
        ) {
            let d = diagonal_matrix(&diag);
            let lambda = diag[i];
            let basis = eigenspace_basis(EPS, &d, lambda).unwrap();
            prop_assert!(!basis.is_empty(), "λ = dᵢ 是特徵值,Eλ 非平凡");
            prop_assert!(is_linearly_independent(EPS, &basis), "基底必線性獨立");
            for v in &basis {
                prop_assert!(is_eigenpair(EPS, &d, v, lambda), "基底向量必為特徵向量");
            }
        }

        // 練習 4(幾何重數):dim Eλ = nullity(A − λI)。對任意 λ 成立(多半兩邊皆 0)——
        // 把特徵空間維度接回零空間維度,Eλ = Null(A − λI) 的直接後果。
        #[test]
        fn eigenspace_dimension_equals_nullity(a in int_square_any(), lambda in -5i64..=5) {
            const SOLVE_EPS: f64 = 1e-7; // RREF 除法殘差
            let m = characteristic_matrix(&a, lambda as f64).unwrap();
            let basis = eigenspace_basis(SOLVE_EPS, &a, lambda as f64).unwrap();
            prop_assert_eq!(basis.len(), m.nullity(SOLVE_EPS), "dim Eλ ≠ nullity(A − λI)");
        }

        // 練習 5(驗收條件):對稱 2×2 必有實特徵值 —— Δ = (a−d)² + 4b² ≥ 0 恆成立。
        #[test]
        fn symmetric_2x2_always_has_real_eigenvalues(a in symmetric_2x2()) {
            prop_assert!(has_real_eigenvalues_2x2(&a));
        }
    }
}
