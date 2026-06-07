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
}

/// 行列式章的 property test —— 主軸是「同一個數,三種算法」,laws 隨練習
/// 推進累積成三路互相對帳的網;練 1 先鋪原料 submatrix 的結構律。
#[cfg(test)]
mod laws {
    use crate::Matrix;
    use proptest::prelude::*;

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

    /// 2..=5 × 2..=5 的矩陣連同一組界內 (row, col) —— 先抽形狀、再抽內容與
    /// 索引(**依建構**合法,沿「先抽形狀再抽內容」的兩階段慣例)。
    /// 下限 2:刪一列一行後仍至少 1×1,形狀律的「各減一」不被導出表示法的
    /// 退化(0 列 ⟹ `cols()` 回 0)干擾 —— 退化形狀由 example test 釘。
    fn matrix_with_index() -> impl Strategy<Value = (Matrix, usize, usize)> {
        (2usize..=5, 2usize..=5)
            .prop_flat_map(|(rows, cols)| (int_matrix(rows, cols), 0..rows, 0..cols))
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
    }
}
