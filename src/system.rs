//! System —— 線性方程組 `Ax = b` 的表示與操作。
//!
//! 對應原始 Go 專案 commit `46b6b36`
//! (`feat(system): add System type with augmented matrix conversion`)。

use crate::{LinAlgError, Matrix, Vector};

/// 一個線性方程組 `Ax = b`:係數矩陣 `A` 配上常數向量 `b`,`x` 是待解的未知數向量。
///
/// `System` **擁有**(move 進來)它的 `A` 與 `b`,欄位 private —— 這也讓
/// [`to_augmented_matrix`](System::to_augmented_matrix) 回傳的矩陣不可能與來源
/// aliasing(Go 需要額外測試防範這件事,Rust 由所有權與封裝在結構上保證)。
#[derive(Debug, Clone)]
// 欄位 `A` 刻意用大寫對應數學的係數矩陣 A(Ax = b);Rust 預設欄位要 snake_case,
// 故在此 opt-out `non_snake_case`(為數學可讀性付的代價,範圍只限這個型別)。
#[allow(non_snake_case)]
pub struct System {
    A: Matrix,
    b: Vector,
}

impl System {
    /// 用係數矩陣 `a` 與常數向量 `b` 建立線性方程組。
    ///
    /// 每條方程式(A 的一列)對應一個常數(b 的一格),故 `a.rows()` 必須等於
    /// `b.rows()`;不符回 `Err(LinAlgError::DimensionMismatch)`。注意「未知數個數」
    /// (`a.cols()`)不必等於方程式個數 —— 長方系統(超定 / 欠定)是允許的。
    pub fn new(a: Matrix, b: Vector) -> Result<System, LinAlgError> {
        if a.rows() != b.rows() {
            Err(LinAlgError::DimensionMismatch)
        } else {
            Ok(System { A: a, b })
        }
    }

    /// 轉成增廣矩陣 `[A | b]`:列數同 A、行數為 `A.cols() + 1`,最後一行放 b 的
    /// 各分量。這是高斯消去法(Gaussian elimination)操作的表示。
    ///
    /// 不會失敗(建構子已保證 `a.rows() == b.rows()`),故回 `Matrix`。
    pub fn to_augmented_matrix(&self) -> Matrix {
        // 每一列 = A 的第 i 列 ++ b[i]。用 (0..rows).map().collect() —— 與本 crate
        // 其他建構(transpose / identity)同一個迭代器慣用法,免去手動 with_capacity
        // 與 push 的命令式步驟。row(i) 在 i ∈ [0, rows) 內必為 Ok,故 unwrap 安全。
        let augmented_rows = (0..self.A.rows())
            .map(|i| {
                let mut row = self.A.row(i).unwrap().to_vec();
                row.push(self.b.entries()[i]);
                row
            })
            .collect();
        Matrix::from_rows(augmented_rows)
    }
}
