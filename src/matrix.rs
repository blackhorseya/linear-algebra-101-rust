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
/// 欄位皆為 private,只能透過方法存取 —— 對應 Go 版用 private fields 封裝
/// 內部狀態的設計,確保矩陣永遠維持「rows / cols 與 data 形狀一致」的有效狀態。
#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    /// 建立一個 `rows × cols` 的矩陣,所有元素初始化為 0。
    pub fn new(rows: usize, cols: usize) -> Matrix {
        let data = vec![vec![0.0; cols]; rows];
        Matrix { rows, cols, data }
    }

    /// 精確比較兩矩陣是否相等:維度相同,且每個對應元素完全一致。
    pub fn equals(&self, other: &Matrix) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] != other.data[i][j] {
                    return false;
                }
            }
        }
        true
    }

    /// 是否為方陣(rows == cols)。
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// 是否為零矩陣(所有元素皆為 0)。
    pub fn is_zero(&self) -> bool {
        for row in &self.data {
            for &value in row {
                if value != 0.0 {
                    return false;
                }
            }
        }
        true
    }

    /// 逐元素相加,回傳新矩陣。
    ///
    /// 維度不合時回傳 `Err(LinAlgError::DimensionMismatch)` —— 把「這個運算可能
    /// 失敗」提升到型別層級,呼叫端被 `Result` 逼著面對維度條件。
    pub fn add(&self, other: &Matrix) -> Result<Matrix, LinAlgError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(LinAlgError::DimensionMismatch);
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        Ok(result)
    }

    /// 純量乘法:每個元素乘上 `scalar`,回傳新矩陣。
    pub fn scalar_multiply(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White-box 測試輔助:直接設定 private 欄位,從 row-major 字面值建出 `Matrix`。
    /// 因為測試與實作同在一個 module,才能存取私有欄位 —— 對應 Go 的 `matrixFrom`。
    fn matrix_from(data: Vec<Vec<f64>>) -> Matrix {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Matrix { rows, cols, data }
    }

    #[test]
    fn new_matrix_is_zero_initialized() {
        for (rows, cols) in [(2usize, 2usize), (3, 1), (1, 4)] {
            let m = Matrix::new(rows, cols);
            assert_eq!(m.rows, rows, "rows 不符");
            assert_eq!(m.cols, cols, "cols 不符");
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
    fn is_square_detects_square_matrices() {
        assert!(matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).is_square());
        assert!(!matrix_from(vec![vec![1.0], vec![2.0], vec![3.0]]).is_square()); // tall
        assert!(!matrix_from(vec![vec![1.0, 2.0, 3.0]]).is_square()); // wide
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
