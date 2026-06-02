//! Matrix —— 二維矩陣型別與其基本運算。
//!
//! 對應原始 Go 專案的第一個 feat commit:
//! `feat: implement Matrix with equality, addition and scalar multiply`。

use crate::error::LinAlgError;
use crate::vector::Vector;

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

    /// 直接用 row-major 的資料建出矩陣 —— 「給我這些列」的公開值建構子。
    ///
    /// 在此之前公開能建的矩陣只有零矩陣([`new`](Matrix::new))與單位矩陣
    /// ([`identity`](Matrix::identity)),缺一個從給定值建構的入口(對應測試裡的
    /// 白箱 `matrix_from`)。`System::to_augmented_matrix` 等跨模組組裝也需要它。
    ///
    /// 維度沿用導出表示法:`rows()` = 列數、`cols()` = 第一列長度。各列應等長,
    /// 此建構子不另做檢查(與 `matrix_from` 一致),由呼叫端負責給規則的資料。
    pub fn from_rows(data: Vec<Vec<f64>>) -> Matrix {
        Matrix { data }
    }

    /// n×n 單位矩陣 **Iₙ**:主對角線為 1、其餘為 0。它是矩陣乘法的單位元 ——
    /// 對任意向量 `I·x = x`(等有了 matrix×matrix,還有 `I·A = A·I = A`)。
    ///
    /// 結構上 **Iₙ 的第 j 個 column 正是 eⱼ**(見
    /// [`Vector::standard`](crate::Vector::standard)):這就是乘上 I 不改變向量的原因 ——
    /// column view 下 `I·x` 把 x 重建成 `x₀·e₀ + x₁·e₁ + …`,即 x 自己。
    ///
    /// 不會失敗(`n = 0` 給 0×0 空矩陣),故回 `Matrix` 而非 `Result`。
    pub fn identity(n: usize) -> Matrix {
        Matrix {
            data: (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == j { 1.0 } else { 0.0 })
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>(),
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

    /// 是否為 **(column-)stochastic** 矩陣:方陣、每個元素 ≥ 0、且**每一 column**
    /// 和為 1(容差 `epsilon` 內)。這是 Markov chain 的轉移矩陣。
    ///
    /// 為何查 **column** 而非 row:本 crate 把向量建模成 column vector、用 `A·v`
    /// 相乘 —— column-stochastic 的 P 會把任一機率向量(各分量 ≥ 0、總和 1)映成
    /// 另一個機率向量,於是 `P·P·…·v` 就是在鏈上往前走。要檢查 row-stochastic,
    /// 看它的轉置:`self.transpose().is_stochastic(epsilon)`。
    ///
    /// `epsilon` 吸收 column 和的浮點誤差(對應 [`approx_equals`](Matrix::approx_equals)),
    /// 傳 `0.0` 即精確檢查。
    pub fn is_stochastic(&self, epsilon: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        for j in 0..self.cols() {
            let mut sum = 0.0;
            for i in 0..self.rows() {
                let v = self.data[i][j];
                if v < 0.0 {
                    return false; // 機率不能為負
                }
                sum += v;
            }
            if (sum - 1.0).abs() > epsilon {
                return false; // column 和不在容差內
            }
        }
        true
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

    /// 矩陣–向量乘積 `A·v`:m×n 矩陣作用在長度 n 的向量上,得到長度 m 的向量。
    ///
    /// 以 **row view** 實作 —— 結果第 i 格 = `self` 第 i 列與 `v` 的 dot product。
    /// 但更值得記住的是等價的 **column view**:`A·v` 是「用 `v` 各分量當權重,對
    /// `self` 各 column 做線性組合」,而 `A·eⱼ` 恰好取出第 j 個 column(見測試
    /// `multiply_vector_by_standard_basis_picks_column`)。這是 span、column space、
    /// 解 `Ax=b` 的基石。
    ///
    /// 維度:`self.cols()` 必須等於 `v.rows()`,結果落在 Rᵐ(= `self.rows()`);
    /// 不合則回 `Err(LinAlgError::DimensionMismatch)`。
    pub fn multiply_vector(&self, v: &Vector) -> Result<Vector, LinAlgError> {
        if self.cols() != v.rows() {
            return Err(LinAlgError::DimensionMismatch);
        }
        let x = v.entries();
        let data = self
            .data
            .iter()
            .map(|row| row.iter().zip(x).map(|(&a, &xi)| a * xi).sum())
            .collect::<Vec<f64>>();
        Ok(Vector::from_vec(data))
    }

    /// 取出第 `j` 個 column 當作 column [`Vector`] ∈ Rʳᵒʷˢ。
    ///
    /// 這把 column view 從隱喻變成第一級操作。代數上,column 抽取就是恆等式
    /// **`A·eⱼ = column(j)`** —— 用第 j 個標準基底向量乘 A,恰好選出第 j 個 column
    /// (見 laws 測試 `multiply_by_standard_vector_equals_column`)。
    ///
    /// `j: usize` 讓負索引無法表示(編譯期擋下),唯一要把關的是越界:
    /// `j >= cols()` → [`LinAlgError::IndexOutOfRange`](重用 `standard` 那個 variant)。
    pub fn column(&self, j: usize) -> Result<Vector, LinAlgError> {
        if j >= self.cols() {
            Err(LinAlgError::IndexOutOfRange {
                index: j,
                len: self.cols(),
            })
        } else {
            let data = (0..self.rows())
                .map(|i| self.data[i][j])
                .collect::<Vec<f64>>();
            Ok(Vector::from_vec(data))
        }
    }

    /// 唯讀借出第 `i` 列(`&[f64]`)。與 [`column`](Matrix::column) 對稱,但回傳的是
    /// 原始 slice 而非 `Vector` —— 本 crate 的 `Vector` 是 *column* vector,一個 row
    /// 並不是 column vector,硬包成 `Vector` 反而失真。
    ///
    /// `i: usize` 讓負索引無法表示;`i >= rows()` → [`LinAlgError::IndexOutOfRange`]。
    pub fn row(&self, i: usize) -> Result<&[f64], LinAlgError> {
        if i >= self.rows() {
            return Err(LinAlgError::IndexOutOfRange {
                index: i,
                len: self.rows(),
            });
        }
        Ok(&self.data[i])
    }

    /// 轉置:沿主對角線翻轉 —— `self` 的 `(i, j)` 變成結果的 `(j, i)`,
    /// `m×n` 矩陣因此變成 `n×m`。
    pub fn transpose(&self) -> Matrix {
        // 結果的第 j 列 = self 的第 j 行(由各列第 j 個元素組成)。
        let data = (0..self.cols())
            .map(|j| {
                (0..self.rows())
                    .map(|i| self.data[i][j])
                    .collect::<Vec<f64>>()
            })
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
    fn identity_builds_diagonal_ones() {
        assert_eq!(Matrix::identity(1).data, vec![vec![1.0]]);
        assert_eq!(
            Matrix::identity(2).data,
            vec![vec![1.0, 0.0], vec![0.0, 1.0]]
        );
        assert_eq!(
            Matrix::identity(3).data,
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ]
        );
        // Iₙ 必為方陣
        for n in [1usize, 2, 3, 5] {
            assert!(Matrix::identity(n).is_square(), "Iₙ 必為方陣");
        }
    }

    #[test]
    fn identity_columns_are_standard_vectors() {
        // I·x = x 背後的結構事實:Iₙ 的第 j 個 column 正是 eⱼ。
        // 用 column view 取出:I·eⱼ 取 I 的第 j 個 column,而它應等於 eⱼ 自己。
        const N: usize = 4;
        let id = Matrix::identity(N);
        for j in 0..N {
            let ej = Vector::standard(N, j).unwrap();
            let got = id.multiply_vector(&ej).expect("I·eⱼ 不應出錯");
            assert!(got.equals(&ej), "I 的第 {j} 個 column 應為 e{j}");
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
    fn is_stochastic_checks_columns_nonneg_and_sum_one() {
        // 接受:I 的每個 column 都是某個 eⱼ(一個 1、其餘 0),非負且和為 1
        assert!(Matrix::identity(3).is_stochastic(0.0));
        // 接受:置換矩陣(I 的 column 重排)
        assert!(matrix_from(vec![vec![0.0, 1.0], vec![1.0, 0.0]]).is_stochastic(0.0));
        // 接受:均勻 1/2,精確和為 1
        assert!(matrix_from(vec![vec![0.5, 0.5], vec![0.5, 0.5]]).is_stochastic(0.0));

        // 容差:column 1 和為 1 + 1e-12,在 1e-9 內算過、精確檢查(eps=0)不過。
        // (註:像 1/3 三次或 0.1+0.2+0.7 常在 f64 下恰好等於 1.0,故刻意造一個已知偏差)
        let rounding = matrix_from(vec![vec![0.5, 0.5], vec![0.5, 0.5 + 1e-12]]);
        assert!(rounding.is_stochastic(1e-9));
        assert!(!rounding.is_stochastic(0.0));

        // 拒絕:column 0 和為 1.2(機率質量過多)
        assert!(!matrix_from(vec![vec![0.6, 0.5], vec![0.6, 0.5]]).is_stochastic(1e-9));
        // 拒絕:column 0 和為 1 但含 -0.2(機率不能為負)
        assert!(!matrix_from(vec![vec![1.2, 0.0], vec![-0.2, 1.0]]).is_stochastic(1e-9));
        // 拒絕:非方陣永遠不是 stochastic
        assert!(!matrix_from(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]).is_stochastic(0.0));
    }

    #[test]
    fn is_stochastic_checks_columns_not_rows() {
        // row/column 對偶:本 crate 查 COLUMN(配合 column vector 的 A·v)。
        // 一個「每 ROW 和為 1」的矩陣,要轉置後才是 column-stochastic。
        let row_stochastic = matrix_from(vec![vec![0.5, 0.5], vec![1.0, 0.0]]); // 每列和為 1
        assert!(
            !row_stochastic.is_stochastic(1e-9),
            "row-stochastic 不該被當成 column-stochastic"
        );
        assert!(
            row_stochastic.transpose().is_stochastic(1e-9),
            "其轉置才是 column-stochastic"
        );
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

    #[test]
    fn transpose_swaps_rows_and_cols() {
        // 寬變高:2×3 → 3×2
        let wide = matrix_from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let t = wide.transpose();
        assert_eq!(t.data, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
        assert_eq!((t.rows(), t.cols()), (3, 2));

        // 方陣:沿對角線反射
        let square = matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(
            square.transpose().data,
            vec![vec![1.0, 3.0], vec![2.0, 4.0]]
        );

        // 列向量變行向量:1×3 → 3×1
        let row = matrix_from(vec![vec![1.0, 2.0, 3.0]]);
        assert_eq!(row.transpose().data, vec![vec![1.0], vec![2.0], vec![3.0]]);
    }

    #[test]
    fn multiply_vector_computes_product() {
        // 方陣 2×2:[1·5+2·6, 3·5+4·6] = [17, 39]
        let a = matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let got = a
            .multiply_vector(&Vector::from_vec(vec![5.0, 6.0]))
            .unwrap();
        assert!(
            got.equals(&Vector::from_vec(vec![17.0, 39.0])),
            "A·v 應為 [17, 39]"
        );

        // 非方陣 2×3 壓縮維度:結果落在 R²
        // [1·1+2·0+3·(-1), 4·1+5·0+6·(-1)] = [-2, -2]
        let b = matrix_from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let got = b
            .multiply_vector(&Vector::from_vec(vec![1.0, 0.0, -1.0]))
            .unwrap();
        assert!(got.equals(&Vector::from_vec(vec![-2.0, -2.0])));
        assert_eq!(got.rows(), 2, "2×3 · (長度3) 結果應為長度 2");

        // 單位矩陣是乘法單位元:I·x = x
        let i = matrix_from(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let x = Vector::from_vec(vec![7.0, 8.0]);
        assert!(i.multiply_vector(&x).unwrap().equals(&x), "I·x 應為 x");
    }

    #[test]
    fn multiply_vector_rejects_dimension_mismatch() {
        // 2×2 的 cols=2,乘長度 3 的向量 → 維度不合
        let a = matrix_from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(
            a.multiply_vector(&v).unwrap_err(),
            LinAlgError::DimensionMismatch
        );
    }

    #[test]
    fn multiply_vector_by_standard_basis_picks_column() {
        // column view 的定義性質:A·eⱼ 取出 A 的第 j 個 column。這也是為什麼
        // I·x = x、為什麼「Ax 是 column 的線性組合」這句話字面上成立。
        let a = matrix_from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // 2×3
        let want_cols = [
            Vector::from_vec(vec![1.0, 4.0]),
            Vector::from_vec(vec![2.0, 5.0]),
            Vector::from_vec(vec![3.0, 6.0]),
        ];
        for (j, want) in want_cols.iter().enumerate() {
            let e = Vector::standard(a.cols(), j).unwrap();
            let got = a.multiply_vector(&e).expect("A·eⱼ 不應出錯");
            assert!(got.equals(want), "A·e{j} 應為第 {j} 個 column");
        }
    }

    #[test]
    fn multiply_vector_on_empty_matrix_returns_zero_vector() {
        // row-view 的紅利:退化 2×0 矩陣乘 R⁰ 向量,結果維度由「列數」決定,
        // 自然得 R² 零向量(Ok)。Go 的 column-view 在此會回 Err(沒有 column
        // 可組合)—— 我們選 row-view 正是為了讓這個邊界天生正確。
        let a = Matrix::new(2, 0); // 2×0
        let v = Vector::new(0); // R⁰
        let got = a
            .multiply_vector(&v)
            .expect("row view 對 m×0 應回零向量而非 Err");
        assert_eq!(got.rows(), 2, "結果維度應為列數 2");
        assert!(got.is_zero());
    }

    #[test]
    fn column_extracts_jth_column() {
        let a = matrix_from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // 2×3
        assert!(
            a.column(0)
                .unwrap()
                .equals(&Vector::from_vec(vec![1.0, 4.0]))
        );
        assert!(
            a.column(1)
                .unwrap()
                .equals(&Vector::from_vec(vec![2.0, 5.0]))
        );
        assert!(
            a.column(2)
                .unwrap()
                .equals(&Vector::from_vec(vec![3.0, 6.0]))
        );
        // 每個 column 落在 R^rows
        assert_eq!(a.column(0).unwrap().rows(), 2, "column 應 ∈ R^rows");
    }

    #[test]
    fn column_rejects_out_of_range_index() {
        let a = matrix_from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // 2×3, cols=3
        // j == cols 已越界(合法是 [0, 3))
        assert_eq!(
            a.column(3).unwrap_err(),
            LinAlgError::IndexOutOfRange { index: 3, len: 3 }
        );
        // 註:Go 還測「負索引」,但 j: usize 在 Rust 編譯期就排除,無需 runtime 測試。
    }

    #[test]
    fn from_rows_builds_from_row_major_values() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!((m.rows(), m.cols()), (2, 3));
        assert_eq!(m.data, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn row_borrows_ith_row() {
        let m = matrix_from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // 2×3
        assert_eq!(m.row(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(m.row(1).unwrap(), &[4.0, 5.0, 6.0]);
        // i == rows 已越界(合法是 [0, 2))
        assert_eq!(
            m.row(2).unwrap_err(),
            LinAlgError::IndexOutOfRange { index: 2, len: 2 }
        );
    }
}

/// 教材定理的 property test —— 用 proptest 驗證 Matrix 該滿足的代數律。
///
/// 定理是「for all」敘述,程式無法*證明*(那要 proof assistant),只能*驗證*:
/// proptest 自動產生大量隨機輸入,一個反例就推翻,且會把反例 **shrink** 成最小案例。
///
/// 兩種策略對應兩種比較:
/// - `int_matrix` 產生小整數值。整數在 f64 下加減乘**完全精確**,可用精確 `equals`。
/// - `real_matrix` 產生真實浮點值。實數運算有捨入誤差,定律只在容差內成立,須用
///   `approx_equals(_, 1e-9)` —— 這正是「為什麼加法律用整數、純量律用實數」的取捨。
#[cfg(test)]
mod laws {
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

    /// 產生長度 `n`、元素為 [-10, 10] 整數的向量(f64 下精確,可用 `is_zero` 精確判定)。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()))
    }

    /// 機率單體(probability simplex)上的一點:n 個非負實數、總和為 1。
    /// 由嚴格正的樣本正規化而得(下界 1e-6 確保總和不為 0、除法安全)。
    fn simplex_point(n: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(1e-6f64..1.0, n).prop_map(|mut col| {
            let sum: f64 = col.iter().sum();
            for x in &mut col {
                *x /= sum;
            }
            col
        })
    }

    /// 產生 n×n 的 column-stochastic 矩陣:每個 column 都是單體上的一點。
    /// 先產生各 column,再轉成 row-major(`data[i][j] = columns[j][i]`)。
    fn stochastic_matrix(n: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(simplex_point(n), n).prop_map(move |columns| {
            let data = (0..n)
                .map(|i| (0..n).map(|j| columns[j][i]).collect())
                .collect();
            Matrix { data }
        })
    }

    /// 機率向量:長度 n、各分量 ≥ 0、總和為 1(單體上的一點)。
    fn probability_vector(n: usize) -> impl Strategy<Value = Vector> {
        simplex_point(n).prop_map(Vector::from_vec)
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

        // ===== Theorem 1.2 —— 轉置(transpose)的性質 =====
        // 刻意用非方陣 2×3:每條律對方陣也成立,但只有非方陣形狀會真正考驗
        // 維度交換 m×n → n×m,抓出把 i/j 寫反的 transpose。

        // (a) (A+B)^T = A^T + B^T — 轉置對加法分配(整數,精確)。【範本】
        #[test]
        fn transpose_of_sum(a in int_matrix(2, 3), b in int_matrix(2, 3)) {
            let left = a.add(&b).unwrap().transpose(); // (A+B)^T
            let right = a.transpose().add(&b.transpose()).unwrap(); // A^T + B^T
            prop_assert!(left.equals(&right), "(A+B)^T != A^T + B^T\n A={a:?}\n B={b:?}");
        }

        // (b) (sA)^T = s(A^T) — 縮放與轉置可交換(真實浮點,approx_equals)。
        #[test]
        fn transpose_of_scalar_multiple(a in real_matrix(2, 3), s in -10.0f64..10.0) {
            let left = a.scalar_multiply(s).transpose();
            let right = a.transpose().scalar_multiply(s);
            prop_assert!(left.approx_equals(&right, 1e-9), "(sA)^T != s(A^T)\n A={a:?}\n s={s:?}");
        }

        // (c) (A^T)^T = A — 轉置兩次回到自己(involution)(整數,精確)。
        #[test]
        fn transpose_involution(a in int_matrix(2, 3)) {
            let transposed_twice = a.transpose().transpose();
            prop_assert!(transposed_twice.equals(&a), "(A^T)^T != A\n A={a:?}");
        }

        // ===== 矩陣–向量乘積的零性質 =====
        // 兩條都從 column view 立刻看出來,且都強調結果維度落在 Rᵐ(A 的列數),
        // 不是 Rⁿ —— 用整數產生器,結果可用精確 `is_zero` 判定。

        // A·0 = 0 —— 任意矩陣乘零向量得零向量。0 ∈ Rⁿ,結果 ∈ Rᵐ。
        #[test]
        fn multiply_by_zero_vector_is_zero(a in int_matrix(2, 3)) {
            let zero = Vector::new(a.cols()); // 0 ∈ Rⁿ(n = A.cols())
            let got = a.multiply_vector(&zero).unwrap();
            prop_assert_eq!(got.rows(), a.rows(), "A·0 維度應為 A.rows()");
            prop_assert!(got.is_zero(), "A·0 應為零向量\n A={a:?}");
        }

        // O·v = 0 —— 零矩陣乘任意向量得零向量。
        #[test]
        fn zero_matrix_times_vector_is_zero(v in int_vector(3)) {
            let o = Matrix::new(2, 3); // 零矩陣 O,2×3
            let got = o.multiply_vector(&v).unwrap();
            prop_assert_eq!(got.rows(), o.rows(), "O·v 維度應為 O.rows()");
            prop_assert!(got.is_zero(), "O·v 應為零向量\n v={v:?}");
        }

        // ===== Theorem 1.3 —— 矩陣–向量乘積的性質 =====
        // (d) A·eⱼ = column(j)、(f) A·0 = 0、(g) O·v = 0 已在上面驗過;以下補其餘。
        // (a)(c)(h) 整數精確,(b) 含實數純量故 approx,(e) 是「矩陣由其作用唯一決定」的關鍵律。

        // (d) A·eⱼ = column(j) —— column 抽取的代數恆等式:用第 j 個標準基底向量乘 A,
        // 恰好選出第 j 個 column。這是 column view 的公式,也是「Ax 是 column 的
        // 線性組合」字面成立的原因。用非方陣 2×3:eⱼ ∈ Rⁿ、column ∈ Rᵐ。整數值精確。
        #[test]
        fn multiply_by_standard_vector_equals_column(a in int_matrix(2, 3)) {
            for j in 0..a.cols() {
                let want = a.column(j).unwrap();
                let e = Vector::standard(a.cols(), j).unwrap();
                let got = a.multiply_vector(&e).unwrap();
                prop_assert!(got.equals(&want), "A·e{j} != column({j})\n A={a:?}");
            }
        }

        // (a) A(u + v) = Au + Av —— 對向量加法分配(乘積在向量引數上是 additive)。整數精確。
        #[test]
        fn matrix_vector_distributes_over_vector_add(
            a in int_matrix(2, 3),
            u in int_vector(3),
            v in int_vector(3),
        ) {
            let left = a.multiply_vector(&u.add(&v).unwrap()).unwrap(); // A(u+v)
            let au = a.multiply_vector(&u).unwrap();
            let av = a.multiply_vector(&v).unwrap();
            let right = au.add(&av).unwrap(); // Au + Av
            prop_assert!(left.equals(&right), "A(u+v) != Au+Av\n A={a:?}\n u={u:?}\n v={v:?}");
        }

        // (b) A(cu) = c(Au) = (cA)u —— 與純量相容(homogeneous)。c 是隨機實數 → approx。
        #[test]
        fn matrix_vector_homogeneous_in_scalar(
            a in int_matrix(2, 3),
            u in int_vector(3),
            c in -10.0f64..10.0,
        ) {
            let left = a.multiply_vector(&u.scale(c)).unwrap(); // A(cu)
            let c_au = a.multiply_vector(&u).unwrap().scale(c); // c(Au)
            let ca_u = a.scalar_multiply(c).multiply_vector(&u).unwrap(); // (cA)u
            prop_assert!(left.approx_equals(&c_au, 1e-9), "A(cu) != c(Au)\n A={a:?}\n u={u:?}\n c={c}");
            prop_assert!(left.approx_equals(&ca_u, 1e-9), "A(cu) != (cA)u\n A={a:?}\n u={u:?}\n c={c}");
        }

        // (c) (A + B)u = Au + Bu —— 對矩陣加法分配(在矩陣引數上是 additive)。整數精確。
        #[test]
        fn matrix_vector_distributes_over_matrix_add(
            a in int_matrix(2, 3),
            b in int_matrix(2, 3),
            u in int_vector(3),
        ) {
            let left = a.add(&b).unwrap().multiply_vector(&u).unwrap(); // (A+B)u
            let au = a.multiply_vector(&u).unwrap();
            let bu = b.multiply_vector(&u).unwrap();
            let right = au.add(&bu).unwrap(); // Au + Bu
            prop_assert!(left.equals(&right), "(A+B)u != Au+Bu\n A={a:?}\n B={b:?}\n u={u:?}");
        }

        // (e) 【關鍵律】矩陣由它在向量上的作用唯一決定:若 Bw = Aw 對所有 w,則 B = A。
        // 不需要「所有 w」—— 由 (a)+(d),只要探測 n 個標準基底向量 eⱼ 就足夠。
        // 驗兩個方向:正向用 eⱼ 重建 A;反向(反證)改動一個元素,只有對應那行的
        // C·eⱼ 會偏離 A·eⱼ。整數精確。
        #[test]
        fn matrix_is_determined_by_action_on_standard_vectors(a in int_matrix(2, 3)) {
            // 正向:用 eⱼ 把每一行探測出來,組回 B,應等於 A。
            let columns: Vec<Vector> = (0..a.cols())
                .map(|j| a.multiply_vector(&Vector::standard(a.cols(), j).unwrap()).unwrap())
                .collect();
            let reconstructed = Matrix {
                data: (0..a.rows())
                    .map(|i| columns.iter().map(|col| col.entries()[i]).collect())
                    .collect(),
            };
            prop_assert!(reconstructed.equals(&a), "用 eⱼ 重建的 B != A\n A={a:?}");

            // 反向(反證):改動 a 的一個元素得到 c,則「c·eⱼ != a·eⱼ」恰好只在
            // 被改元素所在那一行 j 成立。
            let mut c = a.clone();
            let (mod_row, mod_col) = (0usize, 0usize);
            c.data[mod_row][mod_col] += 10.0;
            for j in 0..a.cols() {
                let e = Vector::standard(a.cols(), j).unwrap();
                let col_a = a.multiply_vector(&e).unwrap();
                let col_c = c.multiply_vector(&e).unwrap();
                if j == mod_col {
                    prop_assert!(!col_a.equals(&col_c), "改動的第 {j} 行應不同\n A={a:?}");
                } else {
                    prop_assert!(col_a.equals(&col_c), "未改動的第 {j} 行應相同\n A={a:?}");
                }
            }
        }

        // (h) Iₙv = v —— 單位矩陣是乘法單位元。整數精確。
        #[test]
        fn identity_times_vector_is_identity(v in int_vector(4)) {
            let iv = Matrix::identity(v.rows()).multiply_vector(&v).unwrap();
            prop_assert!(iv.equals(&v), "Iₙv != v\n v={v:?}");
        }

        // ===== Stochastic 矩陣 —— 機率保持 =====
        // 定義性質:column-stochastic 的 P 把機率向量映成機率向量(留在單體
        // {x : xᵢ ≥ 0, Σxᵢ = 1} 上)。證明只用到「每個 column 和為 1」:
        //   Σᵢ(Pv)ᵢ = Σᵢ Σⱼ Pᵢⱼ vⱼ = Σⱼ vⱼ(Σᵢ Pᵢⱼ) = Σⱼ vⱼ·1 = Σⱼ vⱼ = 1。
        // 這正是本 crate 選 column-stochastic 而非 row-stochastic 的原因 ——
        // 它才跟 column vector 的 A·v 相容。元素是正規化實數,故用容差比較。
        #[test]
        fn stochastic_preserves_probability(
            p in stochastic_matrix(4),
            v in probability_vector(4),
        ) {
            prop_assert!(p.is_stochastic(1e-9), "產生器應給出 stochastic 矩陣\n P={p:?}");

            let pv = p.multiply_vector(&v).unwrap();

            // 機率質量不會變負(非負數的乘積與和仍非負)
            prop_assert!(
                pv.entries().iter().all(|&x| x >= 0.0),
                "Pv 出現負分量\n P={p:?}\n v={v:?}"
            );
            // 總機率質量守恆為 1
            let sum: f64 = pv.entries().iter().sum();
            prop_assert!((sum - 1.0).abs() <= 1e-9, "Σ(Pv) = {sum}, want 1\n P={p:?}\n v={v:?}");
        }
    }
}
