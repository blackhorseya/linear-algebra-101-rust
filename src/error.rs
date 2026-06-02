//! 線性代數運算共用的錯誤型別。
//!
//! `LinAlgError` 是整個 library 的**橫切關注點**:維度不合之類的失敗,matrix、
//! vector、未來的線性方程組求解都會用到,因此獨立成模組,不歸任何單一數學概念。
//! 對外仍透過 `lib.rs` 的 `pub use` re-export 成 `crate::LinAlgError`,呼叫端路徑不變
//! ——`matrix.rs` 與 `vector.rs` 都引用同一個型別。

use std::fmt;

/// 線性代數運算的錯誤型別。
///
/// 手刷 enum、不依賴外部 crate,呼叫端可用 `match` 精確區分錯誤種類 ——
/// 這是 Rust 相對於 Go「sentinel error + 字串」的型別安全版本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinAlgError {
    /// 兩個運算元(矩陣或向量)維度不一致,無法進行該運算。
    /// 對應 Go 版的 `ErrDimensionMismatch`。
    DimensionMismatch,
    /// 兩組輸入的數量不符(例:線性組合的純量數 ≠ 向量數)。
    CountMismatch,
    /// 運算需要至少一個輸入,卻收到空集合(例:對空向量集合做線性組合,
    /// 無從決定結果維度)。
    EmptyInput,
    /// 索引超出合法範圍 `[0, len)`(例:取第 `index` 個標準基底向量,
    /// 但 `index >= len`)。這是首個**帶資料**的 variant —— 把出錯的
    /// `index` 與容器長度 `len` 一起帶上,呼叫端與訊息都能拿到具體數值,
    /// 對比 Go 那種「`fmt.Errorf` 格成字串就丟失結構」的做法。
    IndexOutOfRange { index: usize, len: usize },
    /// 基本列運算試圖把某列乘以 0 —— 會抹掉整列、不可逆,因此不算 elementary。
    /// (見 [`Matrix::scale_row`](crate::Matrix::scale_row)。)
    ScaleByZero,
    /// 基本列運算的兩個列索引必須相異,卻收到相同的列 —— 把一列折進自己會塌成
    /// 純量縮放、在係數為 −1 時不可逆。(見
    /// [`Matrix::add_scaled_row`](crate::Matrix::add_scaled_row)。)
    SameRow,
}

impl fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinAlgError::DimensionMismatch => {
                write!(
                    f,
                    "dimension mismatch: operands must have compatible dimensions"
                )
            }
            LinAlgError::CountMismatch => {
                write!(
                    f,
                    "count mismatch: number of scalars must equal number of vectors"
                )
            }
            LinAlgError::EmptyInput => {
                write!(f, "empty input: operation requires at least one vector")
            }
            LinAlgError::IndexOutOfRange { index, len } => {
                write!(f, "index out of range: {index} is not in [0, {len})")
            }
            LinAlgError::ScaleByZero => {
                write!(f, "scale by zero: row operation must be invertible")
            }
            LinAlgError::SameRow => {
                write!(
                    f,
                    "rows must differ: add-scaled-row would not be invertible"
                )
            }
        }
    }
}

impl std::error::Error for LinAlgError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    /// `Display` 輸出是面向人類的公開契約 —— 用完整比對鎖定訊息文字。
    /// 任何措辭改動(如先前 matrices → operands 那次)都會逼著一起更新此斷言,
    /// 等於強制 review 訊息變更,而非悄悄改掉。
    #[test]
    fn display_spells_out_dimension_mismatch() {
        assert_eq!(
            LinAlgError::DimensionMismatch.to_string(),
            "dimension mismatch: operands must have compatible dimensions"
        );
    }

    #[test]
    fn display_spells_out_count_mismatch() {
        assert_eq!(
            LinAlgError::CountMismatch.to_string(),
            "count mismatch: number of scalars must equal number of vectors"
        );
    }

    #[test]
    fn display_spells_out_empty_input() {
        assert_eq!(
            LinAlgError::EmptyInput.to_string(),
            "empty input: operation requires at least one vector"
        );
    }

    /// 帶資料的 variant:除了鎖定措辭,也驗證 `index`/`len` 真的被插進訊息 ——
    /// 這正是選它(而非 unit variant)的理由,訊息要看得到具體數值。
    #[test]
    fn display_interpolates_index_out_of_range() {
        assert_eq!(
            LinAlgError::IndexOutOfRange { index: 3, len: 3 }.to_string(),
            "index out of range: 3 is not in [0, 3)"
        );
    }

    #[test]
    fn display_spells_out_scale_by_zero() {
        assert_eq!(
            LinAlgError::ScaleByZero.to_string(),
            "scale by zero: row operation must be invertible"
        );
    }

    #[test]
    fn display_spells_out_same_row() {
        assert_eq!(
            LinAlgError::SameRow.to_string(),
            "rows must differ: add-scaled-row would not be invertible"
        );
    }

    /// 驗證 `impl std::error::Error` 真的生效:能裝箱成 trait object
    /// (`Box<dyn Error>`,呼叫端串接 / 傳遞錯誤時的常見用法),且因為這是
    /// 葉節點錯誤(沒有更底層的成因),`source()` 回傳 `None`。
    #[test]
    fn usable_as_std_error_trait_object() {
        let err: Box<dyn Error> = Box::new(LinAlgError::DimensionMismatch);
        assert!(err.to_string().contains("dimension mismatch"));
        assert!(err.source().is_none());
    }
}
