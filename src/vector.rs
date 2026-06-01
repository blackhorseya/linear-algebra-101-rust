//! Vector —— 行向量(column vector)型別與其基本運算。
//!
//! 對應原始 Go 專案 commit `e35ab10`
//! (`feat(vector): implement Vector with add, scale, equality`)。

use crate::LinAlgError;

/// 一個 column vector —— matrix 的特例(只有一行)。
///
/// 與 [`Matrix`](crate::Matrix) 一致採導出表示法:只存 `data`,長度從 `data`
/// 算出(見 [`rows`](Vector::rows));`cols()` 恆為 1。欄位 private,只透過方法存取。
#[derive(Debug, Clone)]
pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    /// 建立長度為 `rows` 的零向量。
    pub fn new(rows: usize) -> Vector {
        Vector {
            data: vec![0.0; rows],
        }
    }

    /// 逐元素相加,回傳新向量。
    ///
    /// 長度不符時回傳 `Err(LinAlgError::DimensionMismatch)` —— 與 `Matrix::add`
    /// 共用同一個錯誤型別。
    pub fn add(&self, other: &Vector) -> Result<Vector, LinAlgError> {
        if self.rows() != other.rows() {
            return Err(LinAlgError::DimensionMismatch);
        }
        let summed_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Vector { data: summed_data })
    }

    /// 在容差 `epsilon` 內近似比較兩向量;長度不符回傳 `false`。
    pub fn approx_equals(&self, other: &Vector, epsilon: f64) -> bool {
        if self.rows() != other.rows() {
            return false;
        }
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| (a - b).abs() <= epsilon)
    }

    /// 精確比較兩向量是否相等(委派給 `approx_equals(other, 0.0)`,epsilon=0 即精確)。
    pub fn equals(&self, other: &Vector) -> bool {
        self.approx_equals(other, 0.0)
    }

    /// 是否為零向量(所有元素皆為 0)。
    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|&x| x == 0.0)
    }

    /// 純量乘法:每個元素乘上 `scalar`,回傳新向量。
    pub fn scale(&self, scalar: f64) -> Vector {
        let scaled_data: Vec<f64> = self.data.iter().map(|x| x * scalar).collect();
        Vector { data: scaled_data }
    }

    /// 向量的長度(rows)—— 從 `data` 的長度導出。
    pub fn rows(&self) -> usize {
        self.data.len()
    }

    /// column vector 的行數恆為 1。
    pub fn cols(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White-box 測試輔助:直接從字面值建出 `Vector`(設定 private 欄位)。
    /// 對應 Go 的 `vectorFrom`。
    fn vector_from(data: Vec<f64>) -> Vector {
        Vector { data }
    }

    #[test]
    fn new_vector_is_zero_initialized() {
        for rows in [0usize, 1, 3, 5] {
            let v = Vector::new(rows);
            assert_eq!(v.rows(), rows, "長度不符");
            assert_eq!(v.cols(), 1, "column vector 的 cols 恆為 1");
            assert_eq!(v.data.len(), rows);
            assert!(v.data.iter().all(|&x| x == 0.0), "新向量必須全為 0");
        }
    }

    #[test]
    fn add_sums_elementwise() {
        let a = vector_from(vec![1.0, 2.0, 3.0]);
        let b = vector_from(vec![4.0, 5.0, 6.0]);
        let sum = a.add(&b).expect("同長度相加不應出錯");
        assert_eq!(sum.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn add_rejects_length_mismatch() {
        let a = vector_from(vec![1.0, 2.0, 3.0]);
        let b = vector_from(vec![1.0, 2.0]);
        assert_eq!(a.add(&b).unwrap_err(), LinAlgError::DimensionMismatch);
    }

    #[test]
    fn equals_compares_length_and_elements() {
        let a = vector_from(vec![1.0, 2.0, 3.0]);
        assert!(a.equals(&vector_from(vec![1.0, 2.0, 3.0])));
        assert!(!a.equals(&vector_from(vec![1.0, 2.0, 4.0])));
        assert!(!a.equals(&vector_from(vec![1.0, 2.0]))); // 長度不同
    }

    #[test]
    fn approx_equals_respects_tolerance() {
        let a = vector_from(vec![1.0]);
        // 差 1e-12:在 1e-9 容差內算相等,精確比較(epsilon=0)不相等
        assert!(a.approx_equals(&vector_from(vec![1.0 + 1e-12]), 1e-9));
        assert!(!a.approx_equals(&vector_from(vec![1.0 + 1e-12]), 0.0));
        // 差 0.1:超過容差
        assert!(!a.approx_equals(&vector_from(vec![1.1]), 1e-9));
        // 長度不同永遠不相等
        assert!(!a.approx_equals(&vector_from(vec![1.0, 2.0]), 1e-9));
    }

    #[test]
    fn is_zero_detects_zero_vectors() {
        assert!(vector_from(vec![0.0, 0.0, 0.0]).is_zero());
        assert!(!vector_from(vec![0.0, 0.0, 1.0]).is_zero());
        assert!(!vector_from(vec![-1.0, 0.0]).is_zero());
    }

    #[test]
    fn scale_multiplies_every_element() {
        let v = vector_from(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.scale(2.0).data, vec![2.0, 4.0, 6.0]);
        assert_eq!(v.scale(0.0).data, vec![0.0, 0.0, 0.0]);
        assert_eq!(v.scale(-1.0).data, vec![-1.0, -2.0, -3.0]);
    }
}
