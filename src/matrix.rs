//! Matrix —— 二維矩陣型別與其基本運算。
//!
//! 對應原始 Go 專案的第一個 feat commit:
//! `feat: implement Matrix with equality, addition and scalar multiply`。

use std::fmt;

/// 線性代數運算的錯誤型別。
///
/// 手刷 enum、不依賴外部 crate,呼叫端可用 `match` 精確區分錯誤種類 ——
/// 這是 Rust 相對於 Go「sentinel error + 字串」的型別安全版本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinAlgError {
    /// 兩個矩陣維度不一致,無法進行該運算。
    /// 對應 Go 版的 `ErrDimensionMismatch`。
    DimensionMismatch,
}

impl fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinAlgError::DimensionMismatch => {
                write!(
                    f,
                    "dimension mismatch: matrices must have the same dimensions"
                )
            }
        }
    }
}

impl std::error::Error for LinAlgError {}

/// 一個以 row-major `Vec<Vec<f64>>` 儲存的二維矩陣。
///
/// 維度不另存欄位,而是從 `data` 導出(見 [`rows`](Matrix::rows) /
/// [`cols`](Matrix::cols))—— `data` 是唯一真相來源,沒有「rows/cols 與 data
/// 對不上」的不變式要維護。欄位 private,只能透過方法存取。
#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<Vec<f64>>,
}

impl Matrix {
    /// 建立一個 `rows × cols` 的矩陣,所有元素初始化為 0。
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            data: vec![vec![0.0; cols]; rows],
        }
    }

    /// 精確比較兩矩陣是否相等:維度相同,且每個對應元素完全一致。
    pub fn equals(&self, other: &Matrix) -> bool {
        self.rows() == other.rows() && self.cols() == other.cols() && self.data == other.data
    }

    /// 在容差 `epsilon` 內近似比較兩矩陣 —— 浮點運算後比較結果時用這個,而非
    /// 精確的 [`equals`](Matrix::equals)。`epsilon` 由呼叫端明確指定:容差該多大
    /// 取決於前面運算的數量級,不該寫死;傳 `0.0` 即退化為精確比較。
    pub fn approx_equals(&self, other: &Matrix, epsilon: f64) -> bool {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            return false;
        }
        for (row_a, row_b) in self.data.iter().zip(&other.data) {
            for (&a, &b) in row_a.iter().zip(row_b) {
                if (a - b).abs() > epsilon {
                    return false;
                }
            }
        }
        true
    }

    /// 是否為方陣(rows == cols)。
    pub fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    /// 是否為零矩陣(所有元素皆為 0)。
    pub fn is_zero(&self) -> bool {
        self.data.iter().flatten().all(|&v| v == 0.0)
    }

    /// 逐元素相加,回傳新矩陣。
    ///
    /// 維度不合時回傳 `Err(LinAlgError::DimensionMismatch)` —— 把「這個運算可能
    /// 失敗」提升到型別層級,呼叫端被 `Result` 逼著面對維度條件。
    pub fn add(&self, other: &Matrix) -> Result<Matrix, LinAlgError> {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            return Err(LinAlgError::DimensionMismatch);
        }
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(row_a, row_b)| {
                row_a
                    .iter()
                    .zip(row_b)
                    .map(|(&a, &b)| a + b)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        Ok(Matrix { data })
    }

    /// 純量乘法:每個元素乘上 `scalar`,回傳新矩陣。
    pub fn scalar_multiply(&self, scalar: f64) -> Matrix {
        let data = self
            .data
            .iter()
            .map(|row| row.iter().map(|&v| v * scalar).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();
        Matrix { data }
    }

    /// 矩陣的列數(rows)—— 從 `data` 的外層長度導出。
    pub fn rows(&self) -> usize {
        self.data.len()
    }

    /// 矩陣的行數(cols)—— 取第一列的長度;空矩陣(0 列)為 0。
    pub fn cols(&self) -> usize {
        self.data.first().map_or(0, |row| row.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White-box 測試輔助:直接從 row-major 字面值建出 `Matrix`。導出版表示法下
    /// 維度由 `data` 決定,包起來就好 —— 對應 Go 的 `matrixFrom`。
    fn matrix_from(data: Vec<Vec<f64>>) -> Matrix {
        Matrix { data }
    }

    #[test]
    fn new_matrix_is_zero_initialized() {
        for (rows, cols) in [(2usize, 2usize), (3, 1), (1, 4)] {
            let m = Matrix::new(rows, cols);
            assert_eq!(m.rows(), rows, "rows 不符");
            assert_eq!(m.cols(), cols, "cols 不符");
            assert_eq!(m.data.len(), rows, "外層長度應為 rows");
            for row in &m.data {
                assert_eq!(row.len(), cols, "每列長度應為 cols");
                assert!(row.iter().all(|&v| v == 0.0), "新矩陣必須全為 0");
            }
        }
    }

    #[test]
    fn equals_compares_dimensions_and_elements() {
        let a = matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        // 完全相同
        assert!(a.equals(&matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]])));
        // 數值不同
        assert!(!a.equals(&matrix_from(vec![vec![1.0, 2.0], vec![3.0, 5.0]])));
        // 維度不同
        assert!(!a.equals(&matrix_from(vec![vec![1.0, 2.0, 3.0]])));
    }

    #[test]
    fn approx_equals_respects_tolerance() {
        let a = matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.000001]]);
        // 差 1e-6:在 1e-3 容差內算相等,但精確比較(epsilon=0)不相等
        assert!(a.approx_equals(&b, 1e-3));
        assert!(!a.approx_equals(&b, 0.0));
        // 維度不同永遠不相等
        assert!(!a.approx_equals(&matrix_from(vec![vec![1.0, 2.0, 3.0]]), 1e-3));
    }

    #[test]
    fn is_square_detects_square_matrices() {
        assert!(matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).is_square());
        assert!(!matrix_from(vec![vec![1.0], vec![2.0], vec![3.0]]).is_square()); // tall
        assert!(!matrix_from(vec![vec![1.0, 2.0, 3.0]]).is_square()); // wide
    }

    #[test]
    fn rows_and_cols_report_dimensions() {
        let m = matrix_from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        // 導出版的取捨:0 列的空矩陣,cols 也只能是 0
        let empty = matrix_from(vec![]);
        assert_eq!(empty.rows(), 0);
        assert_eq!(empty.cols(), 0);
    }

    #[test]
    fn is_zero_detects_zero_matrices() {
        assert!(matrix_from(vec![vec![0.0, 0.0], vec![0.0, 0.0]]).is_zero());
        assert!(!matrix_from(vec![vec![0.0, 0.0], vec![0.0, 1.0]]).is_zero());
    }

    #[test]
    fn add_sums_elementwise() {
        let a = matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = matrix_from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let sum = a.add(&b).expect("同維度相加不應出錯");
        assert_eq!(sum.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    }

    #[test]
    fn add_rejects_dimension_mismatch() {
        let a = matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = matrix_from(vec![vec![1.0, 2.0, 3.0]]);
        assert_eq!(a.add(&b).unwrap_err(), LinAlgError::DimensionMismatch);
    }

    #[test]
    fn scalar_multiply_scales_every_element() {
        let m = matrix_from(vec![vec![1.0, -2.0], vec![3.0, 4.0]]);
        assert_eq!(
            m.scalar_multiply(2.0).data,
            vec![vec![2.0, -4.0], vec![6.0, 8.0]]
        );
        assert_eq!(
            m.scalar_multiply(0.0).data,
            vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        );
        assert_eq!(
            m.scalar_multiply(-1.0).data,
            vec![vec![-1.0, 2.0], vec![-3.0, -4.0]]
        );
    }
}

/// Theorem 1.1 —— 矩陣加法與純量乘法的代數律,用 property test 驗證。
///
/// 定理是「for all」敘述,程式無法*證明*(那要 proof assistant),只能*驗證*:
/// proptest 自動產生大量隨機輸入,一個反例就推翻,且會把反例 **shrink** 成最小案例。
///
/// 兩種策略對應兩種比較:
/// - `int_matrix` 產生小整數值。整數在 f64 下加減乘**完全精確**,可用精確 `equals`。
/// - `real_matrix` 產生真實浮點值。實數運算有捨入誤差,定律只在容差內成立,須用
///   `approx_equals(_, 1e-9)` —— 這正是「為什麼加法律用整數、純量律用實數」的取捨。
#[cfg(test)]
mod theorem_1_1_laws {
    use super::*;
    use proptest::prelude::*;

    /// 產生 `rows×cols`、元素為 [-10, 10] 整數的矩陣(f64 下精確)。
    fn int_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-10i64..=10, cols), rows).prop_map(|grid| {
            Matrix {
                data: grid
                    .into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            }
        })
    }

    /// 產生 `rows×cols`、元素為 [-100, 100] 真實浮點的矩陣。
    fn real_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-100.0f64..100.0, cols), rows)
            .prop_map(|data| Matrix { data })
    }

    proptest! {
        // (a) A + B = B + A — 交換律(整數,精確)。【範本】
        #[test]
        fn add_commutative(a in int_matrix(3, 3), b in int_matrix(3, 3)) {
            let ab = a.add(&b).unwrap();
            let ba = b.add(&a).unwrap();
            prop_assert!(ab.equals(&ba), "A+B != B+A\n A={a:?}\n B={b:?}");
        }

        // (e) (st)A = s(tA) — 純量結合律(真實浮點,approx)。【範本:浮點測試】
        #[test]
        fn scalar_associative(a in real_matrix(3, 3), s in -10.0f64..10.0, t in -10.0f64..10.0) {
            let left = a.scalar_multiply(s * t);
            let right = a.scalar_multiply(t).scalar_multiply(s);
            prop_assert!(left.approx_equals(&right, 1e-9), "(st)A != s(tA)");
        }

        // ===== 以下為 homework:把 todo!() 換成真正的驗證,讓測試轉綠 =====

        // (b) (A+B)+C = A+(B+C) — 加法結合律(整數,精確 equals)。
        #[test]
        fn add_associative(
            a in int_matrix(3, 3),
            b in int_matrix(3, 3),
            c in int_matrix(3, 3),
        ) {
            let ab = a.add(&b).unwrap();
            let left = ab.add(&c).unwrap();
            let bc = b.add(&c).unwrap();
            let right = a.add(&bc).unwrap();
            prop_assert!(left.equals(&right), "(A+B)+C != A+(B+C)\n A={a:?}\n B={b:?}\n C={c:?}");
        }

        // (c) A + O = A — 加法單位元(O = 零矩陣 Matrix::new(rows, cols))。
        #[test]
        fn add_identity(a in int_matrix(3, 3)) {
            let o = Matrix::new(a.rows(), a.cols()); // O 的維度要跟 A 一樣
            let sum = a.add(&o).unwrap();
            prop_assert!(sum.equals(&a), "A + O != A\n A={a:?}");
        }

        // (d) A + (−A) = O — 加法反元素(−A = a.scalar_multiply(-1.0))。
        #[test]
        fn add_inverse(a in int_matrix(3, 3)) {
            let neg = a.scalar_multiply(-1.0);
            let sum = a.add(&neg).unwrap();
            let o = Matrix::new(a.rows(), a.cols()); // 零矩陣的維度要跟 A 一樣
            prop_assert!(sum.equals(&o), "A + (-A) != O\n A={a:?}");
        }

        // (f) s(A+B) = sA + sB — 純量對矩陣加法分配(真實浮點,approx_equals)。
        #[test]
        fn scalar_distributes_over_add(
            a in real_matrix(3, 3),
            b in real_matrix(3, 3),
            s in -10.0f64..10.0,
        ) {
            let ab = a.add(&b).unwrap();
            let left = ab.scalar_multiply(s);
            let sa = a.scalar_multiply(s);
            let sb = b.scalar_multiply(s);
            let right = sa.add(&sb).unwrap();
            prop_assert!(left.approx_equals(&right, 1e-9), "s(A+B) != sA + sB\n A={a:?}\n B={b:?}\n s={s:?}");
        }

        // (g) (s+t)A = sA + tA — 純量加法分配(真實浮點,approx_equals)。
        #[test]
        fn scalar_sum_distributes(
            a in real_matrix(3, 3),
            s in -10.0f64..10.0,
            t in -10.0f64..10.0,
        ) {
            let left = a.scalar_multiply(s + t);
            let sa = a.scalar_multiply(s);
            let ta = a.scalar_multiply(t);
            let right = sa.add(&ta).unwrap();
            prop_assert!(left.approx_equals(&right, 1e-9), "(s+t)A != sA + tA\n A={a:?}\n s={s:?}\n t={t:?}");
        }
    }
}
