//! Determinant(行列式)—— 把整個方陣總結成一個純量。
//!
//! 筆記「Determinants」章(單元 5-5,講義 3.1–3.2)。Chapter 2 的字典
//! (函數 ↔ 矩陣)編完之後,Chapter 3 問一個新問題:**一個 n×n 矩陣能不能
//! 壓縮成一個數,而且這個數還保留「可逆與否」的全部資訊?** 答案是 det A ——
//! IMT 再添一條等價句:A 可逆 ⟺ det A ≠ 0(Theorem 3.4(a))。
//!
//! 本章與前兩章氣質不同:不再是「零新演算法的積木接線」,而是**同一個數、
//! 三種算法** —— 三條路必須算出同一個值,這就是本章 laws 對帳網的主軸:
//!
//! | 練習 | 方法 | 成本 | 角色 |
//! |---|---|---|---|
//! | 2 `determinant_recursive` | 餘因子展開(定義) | O(n!) | 定義本身,教學版 |
//! | 3 `determinant_triangular` | 對角線乘積(Theorem 3.2) | O(n) | 特例 fast path |
//! | 4 `determinant` | Gaussian 消去(Theorem 3.3) | O(n³) | 實用版,得正名 |
//!
//! 練習 1 的 [`submatrix`](Matrix::submatrix) 是定義的原料(A₍ᵢⱼ₎);
//! 練習 5(Theorem 3.4 三大性質)是 laws 的收官 —— 拿 det 把本章與
//! 可逆矩陣章、乘法章、轉置縫起來。
//!
//! 方法掛在 [`Matrix`] 上,但本模組跨在 `matrix` 模組外、碰不到 private 的
//! `data` 欄位 —— 一律走 public API(沿 `elimination` 模組的傳統)。

use crate::{LinAlgError, Matrix};

impl Matrix {
    /// 子矩陣(submatrix)A₍ᵢⱼ₎:移除第 `row` 列與第 `col` 行(0-based)後
    /// 剩下的 (rows−1)×(cols−1) 矩陣 —— 行列式餘因子展開的原料:
    /// det A = Σⱼ (−1)^(1+j) a₁ⱼ det A₍₁ⱼ₎ 裡的 A₍₁ⱼ₎ 就是它。
    /// (教材索引 1-based、程式 0-based:教材的 A₁ⱼ 是這裡的 `submatrix(0, j-1)`。)
    ///
    /// 教材只對 n×n(n ≥ 2)談 A₍ᵢⱼ₎,但「刪一列一行」對任何形狀都自然成立,
    /// 不另設方陣限制。**1×1 的邊界**(拍板):回 `Ok`(0×0 空矩陣)而非錯誤
    /// —— 0×0 是合法的 `Matrix` 值(維度從 `data` 導出,空 `data` 即 0×0),
    /// 而 det 的遞迴 base case 是 1×1、**永遠不會對 1×1 取子矩陣**:邊界全
    /// 定義,錯誤面只剩索引越界一種。注意導出表示法的退化:單列矩陣刪掉唯一
    /// 列後沒有列可量寬度,`cols()` 回 0(rows = 0 ⟹ cols = 0)。
    ///
    /// `row` ≥ `rows()` 或 `col` ≥ `cols()` →
    /// [`LinAlgError::IndexOutOfRange`](帶出錯的索引與對應邊界;先檢查
    /// `row`、再檢查 `col`)。
    ///
    /// 原矩陣不被更動(`&self`,編譯期保證 —— 題目驗收的「資料不被修改」
    /// 由借用檢查器無償提供)。
    ///
    /// 實作提示:列用 [`row`](Matrix::row) 借出(模組外無法直接碰 `data`)。
    /// 題目提示的 `enumerate` + `filter` 形狀 —— 外層
    /// `(0..self.rows()).filter(|&r| r != row)`,內層對借出的列
    /// `iter().enumerate().filter(...)` 留下 `c != col` 的元素,雙層 `collect`
    /// 出 `Vec<Vec<f64>>` 後交給 [`from_rows`](Matrix::from_rows)。
    pub fn submatrix(&self, row: usize, col: usize) -> Result<Matrix, LinAlgError> {
        if row >= self.rows() {
            return Err(LinAlgError::IndexOutOfRange {
                index: row,
                len: self.rows(),
            });
        }
        if col >= self.cols() {
            return Err(LinAlgError::IndexOutOfRange {
                index: col,
                len: self.cols(),
            });
        }
        let data = (0..self.rows())
            .filter(|&r| r != row)
            .map(|r| {
                self.row(r)
                    .unwrap() // r 來自 0..rows() → 界內,unwrap 安全
                    .iter()
                    .enumerate()
                    .filter(|&(c, _)| c != col)
                    .map(|(_, &v)| v)
                    .collect()
            })
            .collect();
        Ok(Matrix::from_rows(data))
    }

    /// 行列式 —— **遞迴餘因子展開**(cofactor expansion,定義本身):沿第一列展開,
    ///
    /// det A = Σⱼ (−1)^(1+j) · a₁ⱼ · det A₁ⱼ(教材 1-based)
    ///       = Σⱼ (−1)^j · a\[0\]\[j\] · det(submatrix(0, j))(程式 0-based,
    ///         符號從 `+` 開始交替 —— 1-based 的 (−1)^(1+j) 在 j 從 0 數時恰是 (−1)^j)。
    ///
    /// **O(n!)**:每層展開 n 個 (n−1)×(n−1) 子問題 —— 這是「定義直譯」的教學版,
    /// 拿來建立直覺、給練習 4 的 Gaussian 版(O(n³),得正名 `determinant`)當
    /// 對帳基準。實用場合不要用它。
    ///
    /// **不收 epsilon**:純加減乘、無判零無消去 —— 本章三支裡唯一精確的
    /// (對比練 3 的三角判定、練 4 的 pivot 搜尋都要容差)。
    ///
    /// 非方陣 → [`LinAlgError::NotSquare`](帶實際形狀)—— `error.rs` 在
    /// `NotSquare` 的 doc 裡早就預言「未來的 `determinant` 同樣適用」,本方法兌現。
    ///
    /// **Base case(拍板延續「邊界全定義」)**:
    /// - 1×1 → `a₁₁`(題目驗收);
    /// - 0×0 → `1.0`(**空積**慣例)—— 這不只是邊界補丁:它讓 1×1 自己也能走
    ///   展開式(a₁₁ · det(0×0) = a₁₁ · 1),所以實作可以二選一 ——
    ///   (a) 教材式:base 寫在 1×1,0×0 另外特判;
    ///   (b) 極簡式:base **只寫 0×0 → 1.0**,讓 1×1 自然落入展開迴圈。
    ///   兩條路測試都收。
    ///
    /// 實作提示:第一列用 [`row`](Matrix::row)`(0)` 借出;符號用
    /// `if j % 2 == 0 { 1.0 } else { -1.0 }`(比 `powi` 直白);子矩陣
    /// `self.submatrix(0, j)` 的索引依建構合法、子矩陣仍是方陣 → 兩層
    /// `unwrap` 都安全(記得行內註解)。
    pub fn determinant_recursive(&self) -> Result<f64, LinAlgError> {
        if !self.is_square() {
            return Err(LinAlgError::NotSquare {
                rows: self.rows(),
                cols: self.cols(),
            });
        }
        if self.rows() == 0 {
            return Ok(1.0); // 0×0 → 空積:讓 1×1 自然落入下方展開迴圈
        }
        let first_row = self.row(0).unwrap(); // rows() > 0 已確立 → 列 0 界內,unwrap 安全
        let mut det = 0.0;
        for (j, &entry) in first_row.iter().enumerate() {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            // 索引依建構合法、子矩陣必為方陣 → 兩層 unwrap 都安全
            let sub_det = self
                .submatrix(0, j)
                .unwrap()
                .determinant_recursive()
                .unwrap();
            det += sign * entry * sub_det;
        }
        Ok(det)
    }

    /// 是否為**上三角矩陣**(upper triangular):主對角線**以下**(i > j)的
    /// 每一格量值都在 `epsilon` 內(算零)—— 非零元素只准住在對角線含以上。
    ///
    /// 只對方陣談三角形;**非方陣 → `false`**(述詞回答「是不是」,不是錯誤
    /// —— 沿 `report`「非方陣恆 false」的精神)。對角矩陣上下皆 true;
    /// 0×0 與 1×1 沒有 off-diagonal 可違規 → vacuous true。
    ///
    /// 實作提示(題目提示的直譯):走訪所有 i > j 的位置檢查
    /// `|v| <= epsilon` —— 雙層迭代配 `all`,或雙層 `for` 提前 `return false`。
    pub fn is_upper_triangular(&self, epsilon: f64) -> bool {
        if !self.is_square() {
            return false; // 非方陣不談三角形 → false(述詞回答「是不是」,不是錯誤)
        }
        for i in 0..self.rows() {
            for j in 0..i {
                if self.row(i).unwrap()[j].abs() > epsilon {
                    return false; // i > j 的位置有非零 → 不是上三角
                }
            }
        }
        true
    }

    /// 是否為**下三角矩陣**(lower triangular):主對角線**以上**(i < j)
    /// 全為零(量值 ≤ `epsilon`)—— [`is_upper_triangular`](Matrix::is_upper_triangular)
    /// 的鏡像,非方陣同樣 `false`。
    ///
    /// 實作提示:兩條路 —— (a) 委派 `self.transpose().is_upper_triangular(eps)`
    /// (一行,但 laws 的轉置對偶律會退化成恆真式、失去獨立驗證的牙齒);
    /// (b) **獨立寫對稱迴圈**(i < j),讓「lower(A) ⟺ upper(Aᵀ)」維持
    /// 兩條獨立路徑互相對帳。建議 (b) —— law 要有東西可咬。
    pub fn is_lower_triangular(&self, epsilon: f64) -> bool {
        if !self.is_square() {
            return false; // 非方陣不談三角形 → false(述詞回答「是不是」,不是錯誤)
        }
        for i in 0..self.rows() {
            for j in i + 1..self.cols() {
                if self.row(i).unwrap()[j].abs() > epsilon {
                    return false; // i < j 的位置有非零 → 不是下三角
                }
            }
        }
        true
    }

    /// 行列式 —— **三角矩陣快速路徑**(Theorem 3.2):上**或**下三角的方陣,
    /// det = 主對角線分量的乘積,O(n) 一條對角線掃完;不是三角(或非方陣)
    /// 回 `None`。
    ///
    /// 為什麼對:下三角沿第一**列**展開,活著的只有 a₁₁ 那項(其餘被 0 乘掉),
    /// det = a₁₁ · det(右下角子陣)—— 子陣仍下三角,歸納剝完恰是對角線乘積;
    /// 上三角同理(沿第一**行**展開,或用練 5 的 det Aᵀ = det A 鏡射過去)。
    ///
    /// **`Option` 而非 `Result`**:「不是三角」不是錯誤,是 fast path
    /// **不適用** —— 呼叫端拿 `None` 就 fallback 到一般算法(沿
    /// `unreachable_vector` 的 Option 語感)。對角線含 0 **不特判**:仍是
    /// 三角、乘積自然為 0(題目驗收)。0×0 → `Some(1.0)`(空積,與
    /// `determinant_recursive` 的 base 同一個慣例 —— 兩條路在邊界也對齊)。
    ///
    /// `epsilon`:三角判定的判零門檻(委派給兩支述詞)。
    ///
    /// 實作提示:先問兩支述詞(`||`),不是三角回 `None`;是 → 對角線
    /// `(0..n).map(|i| row(i)[i]).product()`(`product` 對空迭代器回 1.0,
    /// 0×0 的空積**免費**)。
    pub fn determinant_triangular(&self, epsilon: f64) -> Option<f64> {
        if self.is_upper_triangular(epsilon) || self.is_lower_triangular(epsilon) {
            Some((0..self.rows()).map(|i| self.row(i).unwrap()[i]).product())
        } else {
            None
        }
    }

    /// 行列式 —— **Gaussian 消去版**(Theorem 3.3),三支裡最實用的,**得正名
    /// `determinant`**(沿 `inverse` 的命名先例:正名給實用算法,演算法後綴給
    /// 教學版)。forward 消去化往上三角,但**只准兩種 ERO**:
    ///
    /// - **swap**(換列):det 變號 → 記翻號,最後乘 (−1)^r;
    /// - **add**(R_r += c·R_pivot):det 不變 —— 消去的主力;
    /// - **絕不 scale**(題目要求):scale 會把 det 乘走一個倍數。
    ///
    /// 化完 **det = (−1)^r × 對角線乘積** —— O(n³)。練 2 存證的 ERO 效果
    /// 三部曲就是這個演算法的正確性證明:每一步要嘛翻號、要嘛不變,帳全程平。
    ///
    /// 與 [`row_echelon_form`](Matrix::row_echelon_form) 的差異:那支不數
    /// swap(REF 不在乎正負號),所以這裡自備迴圈;pivot 沿用 crate 內部的
    /// `pivot_row_below`(partial pivoting —— 比教科書「必要時才換」多 swap
    /// 幾次無妨,每次**真** swap 都翻號,代數恆成立)。
    ///
    /// **方陣的簡化**(比 REF 好寫):某 column 在對角線以下找不到 pivot ⟹
    /// 奇異 ⟹ **early return 精確 `0.0`**(題目驗收);反之每 column 必有
    /// pivot,pivot 恰沿對角線走 —— 不需獨立的 pivot_row 游標,`col` 自己
    /// 就是 pivot row。
    ///
    /// **陷阱(測試有釘)**:partial pivoting 找到的 `p` 可能就是 `col` 自己
    /// —— `swap_rows(i, i)` 是無害 no-op,**不可翻號**,否則正負號隨機錯。
    ///
    /// 非方陣 → [`LinAlgError::NotSquare`];0×0 → `Ok(1.0)`(空積,三條路
    /// 連邊界都對齊)。`epsilon`:pivot 判零門檻。
    ///
    /// 效能(題目驗收):O(n³) vs O(n!) 是結構性差距 —— 10×10 下 1,000 步
    /// 對 3,628,800 條展開路徑;測試不做計時斷言(flaky),改釘「遞迴版
    /// 跑不動的尺寸 Gaussian 輕鬆跑完且值正確」(12×12 三對角)。
    ///
    /// 實作提示:`clone` 出 working、`sign: f64 = 1.0`;`for col in 0..n` ——
    /// `pivot_row_below(col, col, epsilon)` 的 `else` 分支 `return Ok(0.0)`;
    /// `p != col` 才 swap + `sign = -sign`;內層 `r in col+1..n` 消去
    /// (factor = working\[r\]\[col\] / pivot,`add_scaled_row(r, col, -factor)`);
    /// 收尾 `Ok(sign * 對角線乘積)`(乘積寫法同 fast path)。
    pub fn determinant(&self, epsilon: f64) -> Result<f64, LinAlgError> {
        if !self.is_square() {
            return Err(LinAlgError::NotSquare {
                rows: self.rows(),
                cols: self.cols(),
            });
        }
        let mut working = self.clone();
        let mut sign = 1.0;
        for col in 0..working.rows() {
            // let-else 拉平 happy path(沿 row_echelon_form 的寫法)
            let Some(p) = working.pivot_row_below(col, col, epsilon) else {
                return Ok(0.0); // 對角線以下找不到 pivot → 奇異 → det 精確為 0
            };
            if p != col {
                working.swap_rows(p, col).unwrap(); // 兩索引皆界內 → unwrap 安全
                sign = -sign; // 真 swap 才翻號;p == col 的自換是 no-op,不翻
            }
            // pivot 值在內層迴圈中不變,提出來算一次(沿 row_echelon_form 的
            // loop-invariant code motion)
            let pivot_val = working.row(col).unwrap()[col];
            for r in col + 1..working.rows() {
                let factor = working.row(r).unwrap()[col] / pivot_val;
                working.add_scaled_row(r, col, -factor).unwrap(); // r > col 必相異
            }
        }
        Ok(sign
            * (0..working.rows())
                .map(|i| working.row(i).unwrap()[i])
                .product::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use crate::{LinAlgError, Matrix};

    /// 題目範例:3×3 刪第 0 列、第 1 行 → [[4,6],[7,9]]。
    #[test]
    fn submatrix_removes_row_and_column() {
        let a = Matrix::from_rows(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        let sub = a.submatrix(0, 1).unwrap();
        assert!(sub.equals(&Matrix::from_rows(vec![vec![4.0, 6.0], vec![7.0, 9.0]])));
    }

    /// 刪最後一列、最後一行 → 留下左上 2×2 —— filter 的另一端邊界也掃過。
    #[test]
    fn submatrix_removes_last_row_and_column() {
        let a = Matrix::from_rows(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        let sub = a.submatrix(2, 2).unwrap();
        assert!(sub.equals(&Matrix::from_rows(vec![vec![1.0, 2.0], vec![4.0, 5.0]])));
    }

    /// 2×2 → 1×1:遞迴 det 真正會走到的最小一步(展開 2×2 時取的子矩陣)。
    #[test]
    fn submatrix_of_2x2_is_1x1() {
        let a = Matrix::from_rows(vec![vec![11.0, 12.0], vec![-8.0, -9.0]]);
        let sub = a.submatrix(0, 0).unwrap();
        assert!(sub.equals(&Matrix::from_rows(vec![vec![-9.0]])));
    }

    /// 1×1 → 0×0(拍板:邊界全定義,回空矩陣不回錯)—— det 遞迴永遠不會
    /// 走到這裡(base case 是 1×1),但邊界行為要釘死。
    #[test]
    fn submatrix_of_1x1_is_empty() {
        let a = Matrix::from_rows(vec![vec![42.0]]);
        let sub = a.submatrix(0, 0).unwrap();
        assert_eq!(sub.rows(), 0);
        assert_eq!(sub.cols(), 0);
    }

    /// 定義自然泛化到非方陣:2×3 刪 (0, 1) → 1×2。
    #[test]
    fn submatrix_works_on_non_square() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let sub = a.submatrix(0, 1).unwrap();
        assert!(sub.equals(&Matrix::from_rows(vec![vec![4.0, 6.0]])));
    }

    /// 列索引越界 → IndexOutOfRange,`len` 帶的是列數。
    #[test]
    fn submatrix_rejects_row_out_of_range() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(
            a.submatrix(2, 0).unwrap_err(),
            LinAlgError::IndexOutOfRange { index: 2, len: 2 }
        );
    }

    /// 行索引越界 → IndexOutOfRange,`len` 帶的是行數。
    #[test]
    fn submatrix_rejects_col_out_of_range() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0]]);
        assert_eq!(
            a.submatrix(0, 5).unwrap_err(),
            LinAlgError::IndexOutOfRange { index: 5, len: 3 }
        );
    }

    /// 兩個索引都越界 → 先報 `row`(檢查順序是公開契約,鎖住)。
    #[test]
    fn submatrix_checks_row_before_col() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(
            a.submatrix(9, 9).unwrap_err(),
            LinAlgError::IndexOutOfRange { index: 9, len: 2 }
        );
    }

    /// 題目驗收:原矩陣不被修改。`&self` 已是編譯期保證,這支把驗收寫成
    /// 可跑的斷言(也防未來簽名倒退成 `&mut self`)。
    #[test]
    fn submatrix_leaves_source_untouched() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let before = a.clone();
        let _ = a.submatrix(0, 0).unwrap();
        assert!(a.equals(&before));
    }

    /// Base case(題目驗收):1×1 的行列式就是唯一的分量 a₁₁。
    #[test]
    fn determinant_recursive_of_1x1_is_the_entry() {
        let a = Matrix::from_rows(vec![vec![7.0]]);
        assert_eq!(a.determinant_recursive().unwrap(), 7.0);
    }

    /// 0×0 → 1.0(空積慣例)—— 讓 1×1 自己也能寫成展開式(a₁₁ · 1)。
    #[test]
    fn determinant_recursive_of_0x0_is_one() {
        let a = Matrix::from_rows(vec![]);
        assert_eq!(a.determinant_recursive().unwrap(), 1.0);
    }

    /// 題目範例:2×2 的展開 = ad − bc。11·(−9) − 12·(−8) = −99 + 96 = −3。
    #[test]
    fn determinant_recursive_of_2x2_matches_ad_minus_bc() {
        let a = Matrix::from_rows(vec![vec![11.0, 12.0], vec![-8.0, -9.0]]);
        assert_eq!(a.determinant_recursive().unwrap(), -3.0);
    }

    /// 題目驗收:3×3 與手算一致。沿第一列展開:
    /// 1·det[[5,6],[8,10]] − 2·det[[4,6],[7,10]] + 3·det[[4,5],[7,8]]
    /// = 1·2 − 2·(−2) + 3·(−3) = −3。
    #[test]
    fn determinant_recursive_of_3x3_matches_hand_computation() {
        let a = Matrix::from_rows(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ]);
        assert_eq!(a.determinant_recursive().unwrap(), -3.0);
    }

    /// 奇異矩陣(第二列 = 第一列 × 2)→ det 精確為 0(整數算術,連容差都不用)。
    #[test]
    fn determinant_recursive_of_singular_is_zero() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert_eq!(a.determinant_recursive().unwrap(), 0.0);
    }

    /// 非方陣 → NotSquare(帶實際形狀)—— error.rs 預言的兌現。
    #[test]
    fn determinant_recursive_rejects_non_square() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(
            a.determinant_recursive().unwrap_err(),
            LinAlgError::NotSquare { rows: 2, cols: 3 }
        );
    }

    /// 題目範例:上三角偵測 —— 對角線以下全零、以上隨意。
    #[test]
    fn upper_triangular_is_detected() {
        let a = Matrix::from_rows(vec![
            vec![3.0, -4.0, -7.0],
            vec![0.0, 8.0, -2.0],
            vec![0.0, 0.0, 9.0],
        ]);
        assert!(a.is_upper_triangular(1e-9));
        assert!(!a.is_lower_triangular(1e-9));
    }

    /// 鏡像:下三角偵測。
    #[test]
    fn lower_triangular_is_detected() {
        let a = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![5.0, 8.0, 0.0],
            vec![1.0, -2.0, 9.0],
        ]);
        assert!(a.is_lower_triangular(1e-9));
        assert!(!a.is_upper_triangular(1e-9));
    }

    /// 對角矩陣同時是上三角**且**下三角(兩個述詞不互斥)。
    #[test]
    fn diagonal_is_both_upper_and_lower_triangular() {
        let d = Matrix::from_rows(vec![vec![2.0, 0.0], vec![0.0, 3.0]]);
        assert!(d.is_upper_triangular(1e-9));
        assert!(d.is_lower_triangular(1e-9));
    }

    /// 題目驗收:非三角矩陣要正確識別(兩側都有非零 → 兩個述詞都 false)。
    #[test]
    fn dense_matrix_is_not_triangular() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(!a.is_upper_triangular(1e-9));
        assert!(!a.is_lower_triangular(1e-9));
    }

    /// 非方陣不談三角形 → 兩個述詞恆 false(述詞回答「是不是」,不是錯誤)。
    #[test]
    fn non_square_is_not_triangular() {
        let a = Matrix::from_rows(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        assert!(!a.is_upper_triangular(1e-9));
        assert!(!a.is_lower_triangular(1e-9));
    }

    /// 浮點殘差場景:對角線以下殘留 1e-12(消去殘渣量級)→ 在 1e-9 門檻下
    /// 仍判上三角 —— epsilon 存在的理由。
    #[test]
    fn epsilon_tolerates_tiny_off_diagonal_residue() {
        let a = Matrix::from_rows(vec![vec![3.0, 5.0], vec![1e-12, 8.0]]);
        assert!(a.is_upper_triangular(1e-9));
        assert!(!a.is_upper_triangular(0.0)); // 精確模式下殘渣就是非零
    }

    /// 題目範例:3 × 8 × 9 = 216。
    #[test]
    fn determinant_triangular_multiplies_diagonal() {
        let a = Matrix::from_rows(vec![
            vec![3.0, -4.0, -7.0],
            vec![0.0, 8.0, -2.0],
            vec![0.0, 0.0, 9.0],
        ]);
        assert_eq!(a.determinant_triangular(1e-9), Some(216.0));
    }

    /// 題目驗收:對角線含 0 → 結果為 0(仍是三角,乘積自然歸零,不特判)。
    #[test]
    fn determinant_triangular_with_zero_on_diagonal_is_zero() {
        let a = Matrix::from_rows(vec![
            vec![3.0, -4.0, -7.0],
            vec![0.0, 0.0, -2.0],
            vec![0.0, 0.0, 9.0],
        ]);
        assert_eq!(a.determinant_triangular(1e-9), Some(0.0));
    }

    /// 題目驗收:非三角 → None(fast path 不適用,呼叫端自行 fallback)。
    #[test]
    fn determinant_triangular_of_dense_is_none() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(a.determinant_triangular(1e-9), None);
    }

    /// 非方陣 → None(連三角都談不上)。
    #[test]
    fn determinant_triangular_of_non_square_is_none() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(a.determinant_triangular(1e-9), None);
    }

    /// 0×0 → Some(1.0):空積慣例 —— 與 determinant_recursive 的 base 在
    /// 邊界上對齊(兩條路連退化案例都算同一個數)。
    #[test]
    fn determinant_triangular_of_0x0_is_one() {
        let a = Matrix::from_rows(vec![]);
        assert_eq!(a.determinant_triangular(1e-9), Some(1.0));
    }

    /// 題目範例:2×2 → −3。消去帶除法(factor = −8/11)→ 殘差,用容差比較
    /// (對比 determinant_recursive 的同一案例用精確 assert_eq —— 兩支算法
    /// 的「精確 vs 近似」性格從測試寫法就看得出來)。
    #[test]
    fn determinant_matches_recursive_on_2x2() {
        let a = Matrix::from_rows(vec![vec![11.0, 12.0], vec![-8.0, -9.0]]);
        assert!((a.determinant(1e-9).unwrap() - (-3.0)).abs() < 1e-9);
    }

    /// 與練 2 同一個 3×3(手算 −3)—— 同一個數,第二條算法。
    #[test]
    fn determinant_of_3x3_matches_hand_computation() {
        let a = Matrix::from_rows(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ]);
        assert!((a.determinant(1e-9).unwrap() - (-3.0)).abs() < 1e-9);
    }

    /// I₄ → 精確 1.0:pivot 全在位(無 swap)、factor 全零(消去是 no-op)
    /// —— 連殘差都沒有,可以精確比較。
    #[test]
    fn determinant_of_identity_is_exactly_one() {
        assert_eq!(Matrix::identity(4).determinant(1e-9).unwrap(), 1.0);
    }

    /// 題目驗收:奇異(第二列 = 第一列 × 2)→ **精確** 0.0 ——
    /// 消去後該 column 找不到 pivot,early return 字面值 0.0,不是殘渣。
    #[test]
    fn determinant_of_singular_is_exactly_zero() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert_eq!(a.determinant(1e-9).unwrap(), 0.0);
    }

    /// 換列翻號:permutation 矩陣 [[0,1],[1,0]] 必須 swap 一次 →
    /// det = (−1)¹ × 1 × 1 = −1(精確:swap 後就是 identity,零殘差)。
    #[test]
    fn determinant_flips_sign_on_real_swap() {
        let a = Matrix::from_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        assert_eq!(a.determinant(1e-9).unwrap(), -1.0);
    }

    /// 陷阱釘死:pivot 已在定位(partial pivoting 回 p == col)→ 自換是
    /// no-op,**不可翻號** —— 錯翻的話對角矩陣會算出 −6。
    #[test]
    fn determinant_does_not_flip_sign_on_self_swap() {
        let a = Matrix::from_rows(vec![vec![2.0, 0.0], vec![0.0, 3.0]]);
        assert_eq!(a.determinant(1e-9).unwrap(), 6.0);
    }

    /// 非方陣 → NotSquare(與 determinant_recursive 同一個錯誤面)。
    #[test]
    fn determinant_rejects_non_square() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(
            a.determinant(1e-9).unwrap_err(),
            LinAlgError::NotSquare { rows: 2, cols: 3 }
        );
    }

    /// 0×0 → 1.0:三條算法在退化邊界全數對齊(空積慣例)。
    #[test]
    fn determinant_of_0x0_is_one() {
        let a = Matrix::from_rows(vec![]);
        assert_eq!(a.determinant(1e-9).unwrap(), 1.0);
    }

    /// 效能驗收的可跑版:12×12 的遞迴版要 12! ≈ 4.8 億條展開路徑(不可行),
    /// Gaussian 版瞬間跑完 —— 這支測試能結束本身就是驗收。
    /// 用三對角矩陣(對角 2、鄰對角 1):det 滿足 continuant 遞推
    /// Dₙ = 2Dₙ₋₁ − Dₙ₋₂(D₁ = 2, D₂ = 3)⟹ Dₙ = n + 1,期望 13。
    #[test]
    fn determinant_scales_to_sizes_recursion_cannot() {
        let n = 12usize;
        let a = Matrix::from_rows(
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            if i == j {
                                2.0
                            } else if i.abs_diff(j) == 1 {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect(),
        );
        assert!((a.determinant(1e-9).unwrap() - 13.0).abs() < 1e-6);
    }
}

/// 行列式章的 property test —— 主軸是「同一個數,三種算法」,laws 隨練習
/// 推進累積成三路互相對帳的網;練 1 先鋪原料 submatrix 的結構律。
#[cfg(test)]
mod laws {
    use crate::Matrix;
    use proptest::prelude::*;

    /// 消去法判零門檻(整數輸入的殘差遠低於此,沿 composition laws 的 EPS)。
    const EPS: f64 = 1e-9;

    /// Gaussian 版 `determinant` 的**等式比較**容差:消去帶除法、殘差隨運算
    /// 累積 —— 沿 inverse / composition 章的雙容差慣例放寬到 1e-6
    /// (EPS 判零、EQ_EPS 比等式,各司其職)。
    const EQ_EPS: f64 = 1e-6;

    /// 固定 `rows×cols`、元素 [-10, 10] 整數(沿 composition 章慣例)。
    /// submatrix 純搬運、零運算 —— 整數配**精確** `equals`,連容差都不需要。
    fn int_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-10i64..=10, cols), rows).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            )
        })
    }

    /// 隨機形狀(1..=4 × 1..=4)的整數矩陣 —— 三角述詞的轉置對偶律要掃到
    /// 非方陣(兩個述詞對非方陣同回 false,轉置兩側仍要一致)。
    fn int_matrix_any_shape() -> impl Strategy<Value = Matrix> {
        (1usize..=4, 1usize..=4).prop_flat_map(|(rows, cols)| int_matrix(rows, cols))
    }

    /// 2..=5 × 2..=5 的矩陣連同一組界內 (row, col) —— 先抽形狀、再抽內容與
    /// 索引(**依建構**合法,沿「先抽形狀再抽內容」的兩階段慣例)。
    /// 下限 2:刪一列一行後仍至少 1×1,形狀律的「各減一」不被導出表示法的
    /// 退化(0 列 ⟹ `cols()` 回 0)干擾 —— 退化形狀由 example test 釘。
    fn matrix_with_index() -> impl Strategy<Value = (Matrix, usize, usize)> {
        (2usize..=5, 2usize..=5)
            .prop_flat_map(|(rows, cols)| (int_matrix(rows, cols), 0..rows, 0..cols))
    }

    /// 方陣連同一個界內列索引(scale 一列只需一個索引)。
    /// 上限 4:`determinant_recursive` 是 O(n!),4! = 24 條遞迴 × proptest
    /// 256 案例還跑得動;元素 ≤ 10 → det ≤ 4!·10⁴,f64 整數精確範圍(2⁵³)
    /// 綽綽有餘,laws 全用**精確**比較。
    fn square_with_row() -> impl Strategy<Value = (Matrix, usize)> {
        (1usize..=4).prop_flat_map(|n| (int_matrix(n, n), 0..n))
    }

    /// n ≥ 2 的方陣連同**兩個相異**列索引 —— modular shift 依建構保證
    /// i ≠ j(j = (i+1+offset) mod n,offset < n−1 走不滿一圈),
    /// 免 prop_assume 丟樣本。
    fn square_with_distinct_rows() -> impl Strategy<Value = (Matrix, usize, usize)> {
        (2usize..=4)
            .prop_flat_map(|n| (int_matrix(n, n), 0..n, 0..n - 1))
            .prop_map(|(m, i, offset)| {
                let n = m.rows();
                (m, i, (i + 1 + offset) % n)
            })
    }

    /// 把不滿足 `keep(i, j)` 的格子清零、其餘照搬 —— 模組外碰不到 `data`,
    /// 用 `row` + `from_rows` 重建(三角策略的共用積木)。
    fn zero_unless(m: &Matrix, keep: fn(usize, usize) -> bool) -> Matrix {
        Matrix::from_rows(
            (0..m.rows())
                .map(|i| {
                    m.row(i)
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(j, &v)| if keep(i, j) { v } else { 0.0 })
                        .collect()
                })
                .collect(),
        )
    }

    /// 隨機**上三角**整數方陣(1..=4):先抽滿矩陣,再把對角線以下清零 ——
    /// **依建構**必為上三角,免 prop_assume。
    fn upper_triangular_int_matrix() -> impl Strategy<Value = Matrix> {
        (1usize..=4)
            .prop_flat_map(|n| int_matrix(n, n))
            .prop_map(|m| zero_unless(&m, |i, j| i <= j))
    }

    /// 隨機**下三角**整數方陣(1..=4):鏡像,把對角線以上清零。
    fn lower_triangular_int_matrix() -> impl Strategy<Value = Matrix> {
        (1usize..=4)
            .prop_flat_map(|n| int_matrix(n, n))
            .prop_map(|m| zero_unless(&m, |i, j| i >= j))
    }

    /// 1..=4 的隨機整數方陣 —— 三路對帳的主食(上限 4:對帳對象
    /// `determinant_recursive` 是 O(n!));練 5 的 Theorem 3.4 三條也吃它。
    fn square_int_matrix() -> impl Strategy<Value = Matrix> {
        (1usize..=4).prop_flat_map(|n| int_matrix(n, n))
    }

    /// **同尺寸**的方陣對 —— 乘法積性(Theorem 3.4(b))要兩個可乘的方陣,
    /// n 在同一次 flat_map 共用,依建構保證可乘(沿 composable_pair 的招)。
    fn square_int_matrix_pair() -> impl Strategy<Value = (Matrix, Matrix)> {
        (1usize..=4).prop_flat_map(|n| (int_matrix(n, n), int_matrix(n, n)))
    }

    proptest! {
        // 形狀律:刪一列一行,兩維各減一 —— 「A₍ᵢⱼ₎ 是 (n−1)×(n−1)」的程式版。
        #[test]
        fn submatrix_shrinks_both_dimensions((m, i, j) in matrix_with_index()) {
            let sub = m.submatrix(i, j).unwrap();
            prop_assert_eq!(sub.rows(), m.rows() - 1);
            prop_assert_eq!(sub.cols(), m.cols() - 1);
        }

        // 內容律:sub[r][c] 必來自原矩陣「跳過第 i 列 / 第 j 行」的對應格 ——
        // 同一個定義的另一種寫法(索引算術 vs filter),兩條路必須同值。
        #[test]
        fn submatrix_entries_come_from_source((m, i, j) in matrix_with_index()) {
            let sub = m.submatrix(i, j).unwrap();
            for r in 0..sub.rows() {
                let src_r = if r < i { r } else { r + 1 };
                for c in 0..sub.cols() {
                    let src_c = if c < j { c } else { c + 1 };
                    prop_assert_eq!(sub.row(r).unwrap()[c], m.row(src_r).unwrap()[src_c]);
                }
            }
        }

        // 轉置對偶:刪列刪行與轉置交換 —— (Aᵀ)₍ⱼᵢ₎ = (A₍ᵢⱼ₎)ᵀ。
        // 練 5 的 det Aᵀ = det A(Theorem 3.4(c))在結構層的前奏。
        #[test]
        fn submatrix_commutes_with_transpose((m, i, j) in matrix_with_index()) {
            let lhs = m.transpose().submatrix(j, i).unwrap();
            let rhs = m.submatrix(i, j).unwrap().transpose();
            prop_assert!(lhs.equals(&rhs));
        }

        // det(Iₙ) = 1(任意 n):展開式沿第一列只有 a₁₁ = 1 那項活著,
        // 遞迴一路剝到 base —— 也是練 3「對角線乘積」在單位矩陣上的特例。
        #[test]
        fn determinant_recursive_of_identity_is_one(n in 1usize..=5) {
            prop_assert_eq!(Matrix::identity(n).determinant_recursive().unwrap(), 1.0);
        }

        // ERO 效果三部曲(一)交換兩列 → det 變號。
        // 練 4 Gaussian 的 (−1)^r 全靠這條 —— 先用「定義版」存證。
        #[test]
        fn swapping_rows_flips_determinant_sign((m, i, j) in square_with_distinct_rows()) {
            let mut swapped = m.clone();
            swapped.swap_rows(i, j).unwrap();
            prop_assert_eq!(
                swapped.determinant_recursive().unwrap(),
                -m.determinant_recursive().unwrap()
            );
        }

        // ERO 效果三部曲(二)某列乘 c → det 乘 c。
        // 這正是練 4 要求「不用 scaling」的理由:scaling 不保 det,
        // 消去過程只准 swap(變號)與 add(不變)。c 依建構非零。
        #[test]
        fn scaling_row_scales_determinant(
            (m, i) in square_with_row(),
            c in prop_oneof![-5i64..=-1, 1i64..=5],
        ) {
            let c = c as f64;
            let mut scaled = m.clone();
            scaled.scale_row(i, c).unwrap();
            prop_assert_eq!(
                scaled.determinant_recursive().unwrap(),
                c * m.determinant_recursive().unwrap()
            );
        }

        // ERO 效果三部曲(三)R_dst += c·R_src → det 不變。
        // Gaussian 消去能一路保持 det(只差正負號)的根據 ——
        // 練 4 的正確性整個站在這條上。c = 0(no-op)也涵蓋。
        #[test]
        fn adding_scaled_row_preserves_determinant(
            (m, dst, src) in square_with_distinct_rows(),
            c in -5i64..=5,
        ) {
            let mut added = m.clone();
            added.add_scaled_row(dst, src, c as f64).unwrap();
            prop_assert_eq!(
                added.determinant_recursive().unwrap(),
                m.determinant_recursive().unwrap()
            );
        }

        // 「同一個數」第一回合:上三角的 fast path(對角線乘積)必與定義版
        // (餘因子展開)同值 —— Theorem 3.2 不是另一個行列式,是同一個數的
        // 快捷算法。整數三角矩陣兩條路都精確,Some 包著精確相等。
        #[test]
        fn triangular_fast_path_agrees_with_recursive_on_upper(
            m in upper_triangular_int_matrix(),
        ) {
            prop_assert_eq!(
                m.determinant_triangular(0.0), // 整數零就是零,精確判定
                Some(m.determinant_recursive().unwrap())
            );
        }

        // 鏡像:下三角同樣對帳(展開沿第一列,下三角是「每層只活一項」
        // 最直接的那一側)。
        #[test]
        fn triangular_fast_path_agrees_with_recursive_on_lower(
            m in lower_triangular_int_matrix(),
        ) {
            prop_assert_eq!(
                m.determinant_triangular(0.0),
                Some(m.determinant_recursive().unwrap())
            );
        }

        // 轉置對偶(任意形狀,含非方陣):lower(A) ⟺ upper(Aᵀ)——
        // 「對角線以下」轉置後變「對角線以上」。兩支述詞獨立實作,
        // 這條 law 才有牙齒(委派 transpose 實作會讓它退化成恆真式);
        // 也是練 5 det Aᵀ = det A 在結構層的前奏。
        #[test]
        fn lower_triangular_iff_transpose_is_upper(m in int_matrix_any_shape()) {
            prop_assert_eq!(
                m.is_lower_triangular(0.0),
                m.transpose().is_upper_triangular(0.0)
            );
        }

        // 「同一個數」第二回合(主對帳):Gaussian(O(n³))與定義版(O(n!))
        // 在隨機方陣上必同值 —— 消去 vs 展開,兩條完全獨立的計算路徑。
        // 整數輸入下遞迴精確、Gaussian 帶除法殘差 → EQ_EPS 等式容差。
        #[test]
        fn gaussian_agrees_with_recursive(m in square_int_matrix()) {
            let gauss = m.determinant(EPS).unwrap();
            let exact = m.determinant_recursive().unwrap();
            prop_assert!((gauss - exact).abs() <= EQ_EPS);
        }

        // 第三回合,三角形閉合:Gaussian 與三角 fast path 在三角矩陣上同值
        // —— recursive ↔ triangular(練 3)、gaussian ↔ recursive(上一條)、
        // gaussian ↔ triangular(這條)三邊都直接驗過,任兩條算法不一致
        // 都會在某條 law 現形。依建構必是三角 → fast path 必 Some。
        #[test]
        fn gaussian_agrees_with_triangular_fast_path(
            m in prop_oneof![upper_triangular_int_matrix(), lower_triangular_int_matrix()],
        ) {
            let fast = m.determinant_triangular(EPS).unwrap();
            let gauss = m.determinant(EPS).unwrap();
            prop_assert!((gauss - fast).abs() <= EQ_EPS);
        }

        // ───── 練習 5:Theorem 3.4 三大性質(laws 收官,由使用者親手寫)─────

        // (a) A 可逆 ⟺ det A ≠ 0 —— 行列式存在的理由:把整套 IMT 壓縮成
        // 一個數的非零判定。本章與可逆矩陣章在此會師。
        // 左路 is_invertible(EPS)(rank 滿不滿),右路 |det| 是否 > EQ_EPS
        // (殘差下「非零」要用容差判;整數矩陣 det 非零時 |det| ≥ 1,安全)。
        #[test]
        fn invertible_iff_determinant_nonzero(m in square_int_matrix()) {
            let invertible = m.is_invertible(EPS);
            let det_nonzero = m.determinant(EPS).unwrap().abs() > EQ_EPS;
            prop_assert_eq!(invertible, det_nonzero);
        }

        // (b) det(AB) = det A · det B —— 乘法積性:與乘法章會師。隨機矩陣
        // 幾乎都非對稱(題目驗收涵蓋)。⚠ 容差陷阱:det(AB) 的量級可達
        // 4!·400⁴ ≈ 6×10¹¹,Gaussian 的「相對」殘差乘上去,絕對誤差會遠超
        // EQ_EPS —— 等式比較要用**相對容差**:
        //   |lhs − rhs| ≤ EQ_EPS · rhs.abs().max(1.0)
        // (「epsilon 由呼叫端視運算數量級指定」這條全 repo 慣例,在這裡
        // 從建議變成必須。)
        #[test]
        fn determinant_is_multiplicative((a, b) in square_int_matrix_pair()) {
            let ab = a.multiply(&b).unwrap();
            let det_ab = ab.determinant(EPS).unwrap();
            let det_a = a.determinant(EPS).unwrap();
            let det_b = b.determinant(EPS).unwrap();
            let rhs = det_a * det_b;
            prop_assert!((det_ab - rhs).abs() <= EQ_EPS * rhs.abs().max(1.0));
        }

        // (c) det Aᵀ = det A —— 與 transpose 會師:行與列在行列式眼中對稱
        // (這也回頭解釋練 2 為何「沿第一列展開」不失一般性)。兩邊量級
        // 相同(≤ 4!·10⁴),絕對容差 EQ_EPS 即可。
        #[test]
        fn determinant_of_transpose_is_unchanged(m in square_int_matrix()) {
            let det_t = m.transpose().determinant(EPS).unwrap();
            let det = m.determinant(EPS).unwrap();
            prop_assert!((det_t - det).abs() <= EQ_EPS);
        }
    }
}
