//! DiagonalMatrix —— 對角矩陣的專用型別,用「結構知識」換 O(n) 乘法。
//!
//! 筆記「特殊矩陣」題:對角陣相乘仍是對角陣,且分量只是對角線元素逐元相乘 ——
//! O(n³) 的一般乘法塌縮成 O(n)。但「驗證輸入是對角陣」本身就要掃 O(n²) 個元素,
//! 若每次乘法都重驗,O(n) 便名不符實。解法是 **parse, don't validate**:驗證只在
//! [`DiagonalMatrix::from_matrix`] 付一次,之後**型別本身就是「這是對角陣」的
//! 證明**。內部只存 n 個對角線元素,off-diagonal 的 0 連表示都不表示 —— 不合法
//! 狀態(對角陣卻有 off-diagonal 非零)根本無法建構,自然沒有「忘了驗證」的漏洞。

use crate::error::LinAlgError;
use crate::matrix::Matrix;

/// n×n 對角矩陣,只存主對角線的 n 個元素。
///
/// 與 [`Matrix`] 同款設計:欄位 private、維度從資料導出(`dimension()` =
/// `diag.len()`),`diag` 是唯一真相來源。
#[derive(Debug, Clone)]
pub struct DiagonalMatrix {
    diag: Vec<f64>,
}

impl DiagonalMatrix {
    /// 直接從對角線元素建構。**不會失敗**:任意 n 個實數都定義一個合法的 n×n
    /// 對角陣(含空向量 → 0×0、含 0 元素 → 不可逆但仍是對角陣),故回
    /// `DiagonalMatrix` 而非 `Result` —— 對比 [`from_matrix`](DiagonalMatrix::from_matrix)
    /// 要驗證的是「一般矩陣」這個更大的輸入空間。
    pub fn new(diag: Vec<f64>) -> DiagonalMatrix {
        DiagonalMatrix { diag }
    }

    /// 矩陣的維度 n(n×n 的 n)—— 從 `diag` 長度導出。對角陣必為方陣,
    /// rows = cols = n,一個數字就夠。
    pub fn dimension(&self) -> usize {
        self.diag.len()
    }

    /// 唯讀借出對角線元素。
    pub fn entries(&self) -> &[f64] {
        &self.diag
    }

    /// 從一般 [`Matrix`] 轉換(parse):驗證它真的是對角陣,**整個輸入空間的
    /// 驗證成本(O(n²))在這裡一次付清**,之後的運算都能信任型別。
    ///
    /// 失敗模式有兩層,各有精確錯誤:
    /// - 非方陣:連「主對角線」都定義不全 → [`LinAlgError::NotSquare`](重用
    ///   `power` 那個帶形狀的 variant)。
    /// - 方陣但主對角線以外有量值 > `epsilon` 的元素 → [`LinAlgError::NotDiagonal`]。
    ///
    /// `epsilon` 沿用本 crate 的浮點判零慣例(同 `is_row_echelon_form`):由呼叫端
    /// 視數量級指定,傳 `0.0` 即精確檢查。
    pub fn from_matrix(m: &Matrix, epsilon: f64) -> Result<DiagonalMatrix, LinAlgError> {
        if m.rows() != m.cols() {
            return Err(LinAlgError::NotSquare {
                rows: m.rows(),
                cols: m.cols(),
            });
        }
        // 單趟掃描:驗 off-diagonal 的同時收對角線,每列只借一次。row(i) 的 i 來自
        // 0..rows() 不可能越界 —— 用 expect 把不變式說死(用 ? 反而會把不會發生的
        // IndexOutOfRange 漏進 from_matrix 的錯誤契約)。
        let n = m.rows();
        let mut diag = Vec::with_capacity(n);
        for i in 0..n {
            let row = m.row(i).expect("i < rows 由迴圈範圍保證");
            for (j, &value) in row.iter().enumerate() {
                if i != j && value.abs() > epsilon {
                    return Err(LinAlgError::NotDiagonal);
                }
            }
            diag.push(row[i]);
        }
        Ok(DiagonalMatrix { diag })
    }

    /// 嵌回一般 [`Matrix`]:對角線放 `diag`、其餘補 0。這是 `from_matrix` 的
    /// 逆向(往返律見 laws `roundtrip_through_matrix`),也是對角快路徑與一般
    /// `multiply` 對拍的橋(laws `fast_path_agrees_with_general_multiply`)。
    pub fn to_matrix(&self) -> Matrix {
        // 同 Matrix::identity 的形狀,對角線放 diag[i] 而非 1。enumerate 直接走
        // diag,省掉索引中介。
        let n = self.dimension();
        let rows = self
            .diag
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                let mut row = vec![0.0; n];
                row[i] = d;
                row
            })
            .collect();
        Matrix::from_rows(rows)
    }

    /// 對角陣乘法 —— **O(n)**:`(D₁·D₂)ᵢᵢ = d₁ᵢ · d₂ᵢ`,逐元相乘即可。
    ///
    /// 為什麼一般定義 `Σₖ aᵢₖ·bₖⱼ` 會塌縮:aᵢₖ 只在 k = i 非零、bₖⱼ 只在 k = j
    /// 非零,交集要 i = k = j —— off-diagonal(i ≠ j)整列和必為 0,對角線只剩
    /// 一項 d₁ᵢ·d₂ᵢ。三層迴圈的工作量被「結構知識」直接消掉兩層。
    ///
    /// 維度不合(n₁ ≠ n₂)回 [`LinAlgError::DimensionMismatch`],與 `Matrix`
    /// 家族一致。順帶一提:對角陣乘法**可交換**(逐元乘積交換)—— 一般矩陣
    /// 失去的交換律,在這個子代數裡回來了(laws `diagonal_multiply_commutes`)。
    pub fn multiply(&self, other: &DiagonalMatrix) -> Result<DiagonalMatrix, LinAlgError> {
        if self.dimension() != other.dimension() {
            return Err(LinAlgError::DimensionMismatch);
        }
        let diag = self
            .diag
            .iter()
            .zip(&other.diag)
            .map(|(&d1, &d2)| d1 * d2)
            .collect();
        Ok(DiagonalMatrix { diag })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_stores_diagonal_entries() {
        let d = DiagonalMatrix::new(vec![2.0, 3.0, 5.0]);
        assert_eq!(d.dimension(), 3);
        assert_eq!(d.entries(), &[2.0, 3.0, 5.0]);
        // 空對角陣 = 0×0,合法(同 Matrix 的退化慣例)
        assert_eq!(DiagonalMatrix::new(vec![]).dimension(), 0);
    }

    #[test]
    fn from_matrix_extracts_diagonal() {
        let m = Matrix::from_rows(vec![vec![2.0, 0.0], vec![0.0, 3.0]]);
        let d = DiagonalMatrix::from_matrix(&m, 0.0).expect("對角矩陣應轉換成功");
        assert_eq!(d.entries(), &[2.0, 3.0]);

        // 單位矩陣是對角陣(對角線全 1)
        let i = DiagonalMatrix::from_matrix(&Matrix::identity(3), 0.0).unwrap();
        assert_eq!(i.entries(), &[1.0, 1.0, 1.0]);

        // 對角線含 0 也合法:「對角陣」不要求可逆,零方陣也是對角陣
        let z = DiagonalMatrix::from_matrix(&Matrix::new(2, 2), 0.0).unwrap();
        assert_eq!(z.entries(), &[0.0, 0.0]);
    }

    #[test]
    fn from_matrix_judges_zero_within_epsilon() {
        // off-diagonal 的 1e-12:1e-9 容差內算零 → 可轉換;精確檢查 → NotDiagonal
        let m = Matrix::from_rows(vec![vec![2.0, 1e-12], vec![0.0, 3.0]]);
        assert!(DiagonalMatrix::from_matrix(&m, 1e-9).is_ok());
        assert_eq!(
            DiagonalMatrix::from_matrix(&m, 0.0).unwrap_err(),
            LinAlgError::NotDiagonal
        );
    }

    #[test]
    fn from_matrix_rejects_non_square_and_off_diagonal() {
        // 非方陣:主對角線定義不全 → NotSquare(重用 power 那個帶形狀的 variant)
        let rect = Matrix::from_rows(vec![vec![1.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]]);
        assert_eq!(
            DiagonalMatrix::from_matrix(&rect, 0.0).unwrap_err(),
            LinAlgError::NotSquare { rows: 2, cols: 3 }
        );
        // 方陣但 off-diagonal 有實質非零 → NotDiagonal
        let m = Matrix::from_rows(vec![vec![1.0, 5.0], vec![0.0, 2.0]]);
        assert_eq!(
            DiagonalMatrix::from_matrix(&m, 1e-9).unwrap_err(),
            LinAlgError::NotDiagonal
        );
    }

    #[test]
    fn to_matrix_embeds_diagonal() {
        let d = DiagonalMatrix::new(vec![2.0, 3.0]);
        assert!(
            d.to_matrix()
                .equals(&Matrix::from_rows(vec![vec![2.0, 0.0], vec![0.0, 3.0]]))
        );
        // 全 1 對角線嵌回去就是單位矩陣
        assert!(
            DiagonalMatrix::new(vec![1.0; 3])
                .to_matrix()
                .equals(&Matrix::identity(3))
        );
    }

    #[test]
    fn multiply_is_entrywise_on_diagonal() {
        let a = DiagonalMatrix::new(vec![2.0, 3.0]);
        let b = DiagonalMatrix::new(vec![5.0, 7.0]);
        let ab = a.multiply(&b).expect("同維對角陣應可乘");
        assert_eq!(ab.entries(), &[10.0, 21.0]);

        // 全 1 對角陣(= I)是乘法單位元
        let i = DiagonalMatrix::new(vec![1.0, 1.0]);
        assert_eq!(a.multiply(&i).unwrap().entries(), &[2.0, 3.0]);
    }

    #[test]
    fn multiply_rejects_dimension_mismatch() {
        let a = DiagonalMatrix::new(vec![1.0, 2.0]);
        let b = DiagonalMatrix::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(a.multiply(&b).unwrap_err(), LinAlgError::DimensionMismatch);
    }
}

/// 對角陣該滿足的代數律 —— 重點不是「對角陣自己對不對」,而是**快路徑與一般
/// 路徑的語意一致性**:優化只准改變成本,不准改變答案。
#[cfg(test)]
mod laws {
    use super::*;
    use proptest::prelude::*;

    /// 長度 `n`、元素為 [-10, 10] 整數的對角陣(f64 下精確,可用精確比較)。
    fn int_diag(n: usize) -> impl Strategy<Value = DiagonalMatrix> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| DiagonalMatrix::new(xs.into_iter().map(|v| v as f64).collect()))
    }

    proptest! {
        // 【驗收律】快路徑 = 一般路徑:O(n) 對角乘法嵌回 Matrix 後,必須跟
        // O(n³) 的一般 multiply 算出一模一樣的結果。
        #[test]
        fn fast_path_agrees_with_general_multiply(a in int_diag(4), b in int_diag(4)) {
            let fast = a.multiply(&b).unwrap().to_matrix();
            let general = a.to_matrix().multiply(&b.to_matrix()).unwrap();
            prop_assert!(fast.equals(&general), "對角快路徑 != 一般乘法\n a={a:?}\n b={b:?}");
        }

        // 對角陣乘法可交換:D₁D₂ = D₂D₁(逐元乘積交換)—— 與一般矩陣的
        // multiply_is_not_commutative 對照:「對角」這個子代數把交換律找回來了。
        #[test]
        fn diagonal_multiply_commutes(a in int_diag(4), b in int_diag(4)) {
            let ab = a.multiply(&b).unwrap();
            let ba = b.multiply(&a).unwrap();
            prop_assert_eq!(ab.entries(), ba.entries(), "D₁D₂ != D₂D₁");
        }

        // 嵌入往返:to_matrix 再 from_matrix 應完整還原。to_matrix 產的
        // off-diagonal 是精確的 0,epsilon 用 0.0 即可。
        #[test]
        fn roundtrip_through_matrix(d in int_diag(4)) {
            let back = DiagonalMatrix::from_matrix(&d.to_matrix(), 0.0).unwrap();
            prop_assert_eq!(back.entries(), d.entries(), "往返應還原對角線");
        }
    }
}
