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

    /// 第 `i` 個標準基底向量 **eᵢ** ∈ Rʳᵒʷˢ:長度 `rows`,只有第 `i` 格是 1、
    /// 其餘為 0。它是 rows×rows 單位矩陣的第 `i` 行,也是「每個向量都是基底向量
    /// 的線性組合」這句話裡的那組基底 —— 純量就是各 eᵢ 的座標
    /// (見 [`linear_combination`](Vector::linear_combination))。
    ///
    /// `i` 型別是 `usize`,所以**負索引在 Rust 無法表示**(`standard(3, -1)` 編譯
    /// 不過),不像 Go 還要 runtime 檢查 `i < 0`;唯一要把關的是越界:
    /// `i >= rows` → [`LinAlgError::IndexOutOfRange`]。
    pub fn standard(rows: usize, i: usize) -> Result<Vector, LinAlgError> {
        // 1. 越界(i >= rows)→ Err(LinAlgError::IndexOutOfRange { index: i, len: rows })
        // 2. 否則:用 Vector::new(rows) 取零向量,把第 i 格設成 1.0,Ok(它)
        //    (在 impl 內部可直接碰 private 的 data 欄位;設值需要 `let mut`)
        if i >= rows {
            Err(LinAlgError::IndexOutOfRange {
                index: i,
                len: rows,
            })
        } else {
            let mut v = Vector::new(rows);
            v.data[i] = 1.0;
            Ok(v)
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

    /// 線性組合:計算 `Σ scalarsᵢ · vectorsᵢ`(第 i 個向量依第 i 個純量縮放後相加)。
    ///
    /// 這是 span、線性獨立、矩陣乘法的基石 —— 把向量集合當「基底候選」,純量就是
    /// 各基底向量的「座標」。設計成**關聯函式**(無 `self`):它作用在一組向量上、
    /// 產出新的 `Vector`,像具名建構子,因此掛在 `Vector` 之下而非散成 free function。
    ///
    /// 錯誤(三種語意不同的失敗,呼叫端可 `match` 區分):
    /// - `scalars` 與 `vectors` 數量不符 → [`LinAlgError::CountMismatch`]
    /// - `vectors` 為空(無從決定結果維度)→ [`LinAlgError::EmptyInput`]
    /// - 各向量維度不一致 → [`LinAlgError::DimensionMismatch`](由 `add` 把關)
    pub fn linear_combination(scalars: &[f64], vectors: &[Vector]) -> Result<Vector, LinAlgError> {
        if scalars.len() != vectors.len() {
            return Err(LinAlgError::CountMismatch);
        }
        if vectors.is_empty() {
            return Err(LinAlgError::EmptyInput);
        }
        let dim = vectors[0].rows();
        let mut result = Vector::new(dim);
        for (scalar, vector) in scalars.iter().zip(vectors.iter()) {
            if vector.rows() != dim {
                return Err(LinAlgError::DimensionMismatch);
            }
            let scaled = vector.scale(*scalar);
            result = result.add(&scaled)?;
        }
        Ok(result)
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

    #[test]
    fn standard_builds_unit_vectors() {
        // e₀ in R³、e₂ in R³、e₀ in R¹ —— 只有第 i 格為 1,其餘為 0
        assert_eq!(Vector::standard(3, 0).unwrap().data, vec![1.0, 0.0, 0.0]);
        assert_eq!(Vector::standard(3, 2).unwrap().data, vec![0.0, 0.0, 1.0]);
        assert_eq!(Vector::standard(1, 0).unwrap().data, vec![1.0]);
    }

    #[test]
    fn standard_rejects_out_of_range_index() {
        // 合法索引是 [0, rows),所以 i == rows 已越界
        assert_eq!(
            Vector::standard(3, 3).unwrap_err(),
            LinAlgError::IndexOutOfRange { index: 3, len: 3 }
        );
        // 遠超範圍,len 仍如實回報為 3
        assert_eq!(
            Vector::standard(3, 99).unwrap_err(),
            LinAlgError::IndexOutOfRange { index: 99, len: 3 }
        );
        // 註:Go 還測了「負索引」,但 Rust 的 `i: usize` 讓負值無法表示,
        // 該 case 在編譯期就被擋掉,不需要(也無法)寫成 runtime 測試。
    }

    #[test]
    fn standard_basis_reconstructs_coordinates() {
        // 標準基底之所以是「基底」的定義性質:以座標當權重對 eᵢ 做線性組合,
        // 會精確還原出帶那組座標的向量。 Σ coordsᵢ · eᵢ == [coords]
        let coords = [2.0, 3.0, 5.0];
        let basis: Vec<Vector> = (0..coords.len())
            .map(|i| Vector::standard(coords.len(), i).unwrap())
            .collect();
        let reconstructed = Vector::linear_combination(&coords, &basis).unwrap();
        assert_eq!(reconstructed.data, vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn linear_combination_weights_and_sums() {
        // 2·[1,0] + 3·[0,1] = [2,3] —— 純量即標準基底下的座標
        let basis = [vector_from(vec![1.0, 0.0]), vector_from(vec![0.0, 1.0])];
        assert_eq!(
            Vector::linear_combination(&[2.0, 3.0], &basis)
                .unwrap()
                .data,
            vec![2.0, 3.0]
        );

        // 2·[1,2,3] + (-1)·[4,5,6] = [-2,-1,0]
        let vs = [
            vector_from(vec![1.0, 2.0, 3.0]),
            vector_from(vec![4.0, 5.0, 6.0]),
        ];
        assert_eq!(
            Vector::linear_combination(&[2.0, -1.0], &vs).unwrap().data,
            vec![-2.0, -1.0, 0.0]
        );

        // 單一向量退化為 scale:5·[1,2] = [5,10]
        let one = [vector_from(vec![1.0, 2.0])];
        assert_eq!(
            Vector::linear_combination(&[5.0], &one).unwrap().data,
            vec![5.0, 10.0]
        );

        // 全零純量 → 零向量
        let vs2 = [vector_from(vec![1.0, 2.0]), vector_from(vec![3.0, 4.0])];
        assert_eq!(
            Vector::linear_combination(&[0.0, 0.0], &vs2).unwrap().data,
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn linear_combination_rejects_count_mismatch() {
        let basis = [vector_from(vec![1.0, 0.0]), vector_from(vec![0.0, 1.0])];
        // 3 個純量、2 個向量
        assert_eq!(
            Vector::linear_combination(&[1.0, 2.0, 3.0], &basis).unwrap_err(),
            LinAlgError::CountMismatch
        );
    }

    #[test]
    fn linear_combination_rejects_dimension_mismatch() {
        // 向量長度不一致(2 vs 1)
        let vs = [vector_from(vec![1.0, 2.0]), vector_from(vec![3.0])];
        assert_eq!(
            Vector::linear_combination(&[1.0, 1.0], &vs).unwrap_err(),
            LinAlgError::DimensionMismatch
        );
    }

    #[test]
    fn linear_combination_rejects_empty() {
        // 空輸入:數量相符(0 == 0)但無從決定結果維度
        assert_eq!(
            Vector::linear_combination(&[], &[]).unwrap_err(),
            LinAlgError::EmptyInput
        );
    }
}
