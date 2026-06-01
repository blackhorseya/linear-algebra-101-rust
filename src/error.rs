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
