//! 線性轉換(Linear Transformation)—— 矩陣作為函數。
//!
//! 筆記「線性轉換與矩陣」章(單元 5-1,講義 2.7 前段):把 m×n 矩陣 A 從
//! 「運算工具」升格為「函數」—— A 誘導一個映射 **T_A: ℝⁿ → ℝᵐ,T_A(x) = Ax**。
//! 本模組依筆記的鋪陳分四步:
//!
//! 1. **[`Transformation`] 結構**:包住矩陣,回答「定義域 / 對應域是哪個空間」——
//!    n(行數)是輸入維度、m(列數)是輸出維度,方向容易搞反,用命名方法釘住。
//! 2. **apply**:T_A(x) = Ax,矩陣–向量乘法的「函數視角」。
//! 3. **verify_linearity**:線性轉換的兩大守恆 —— 加法 T(u+v) = T(u)+T(v)、
//!    純量乘 T(cu) = c·T(u)。
//! 4. **identity / zero**:最簡單的兩個線性轉換 I(x) = x、T₀(x) = 0。
//!
//! Theorem 2.7(**矩陣誘導的轉換必為線性**)以 `mod laws` 的 proptest 隨機驗證。
//!
//! 與 `inverse` 同款佈局:本模組跨在 `matrix` 模組外、碰不到 private 的 `data`
//! 欄位 —— 一律走 public API,再次驗證先前刻的公開介面足以表達新概念。

use crate::Matrix;

/// 由矩陣誘導的轉換(matrix transformation induced by A):
/// 把 m×n 矩陣 A 視為函數 **T_A: ℝⁿ → ℝᵐ**。
///
/// newtype 包裝的意義:`Matrix` 是「一張數字表」,`Transformation` 是「一個函數」——
/// 同一份資料、兩種視角。型別把視角的切換變成顯式動作(`Transformation::new`),
/// 而不是讀程式碼的人腦中的默契。
#[derive(Debug, Clone)]
pub struct Transformation {
    matrix: Matrix,
}

impl Transformation {
    /// 把矩陣升格為轉換:A ↦ T_A。
    ///
    /// 任何矩陣都誘導一個合法的映射(維度再小都有對應的 ℝⁿ → ℝᵐ),
    /// 所以建構不會失敗 —— 失敗的可能性留給真正會出錯的運算(如 apply 的維度檢查)。
    pub fn new(matrix: Matrix) -> Transformation {
        Transformation { matrix }
    }

    /// 定義域(domain)維度 n:輸入向量 x ∈ ℝⁿ。
    ///
    /// 練習 1 的核心陷阱:n 對應矩陣的「行數」還是「列數」?
    /// 想想 Ax 要怎麼乘 —— x 的長度必須等於 A 每一列的長度。
    pub fn domain_dim(&self) -> usize {
        self.matrix.cols()
    }

    /// 對應域(codomain)維度 m:輸出向量 y = T_A(x) ∈ ℝᵐ。
    pub fn codomain_dim(&self) -> usize {
        self.matrix.rows()
    }

    /// 回傳 `(n, m)`:此轉換從 ℝⁿ 映射到 ℝᵐ。
    ///
    /// 注意順序是「(定義域, 對應域)」—— 與矩陣慣稱的 m×n(列×行)恰好相反,
    /// 這正是題目要釘住的觀念:**矩陣大小唸作 m×n,映射方向卻是 ℝⁿ → ℝᵐ**。
    pub fn dimensions(&self) -> (usize, usize) {
        (self.domain_dim(), self.codomain_dim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 練習 1 題目原例:3×5 矩陣(3 列 5 行)誘導 T: ℝ⁵ → ℝ³。
    /// 列數 3 是「輸出」維度、行數 5 是「輸入」維度 —— 方向與 m×n 的唸法相反。
    #[test]
    fn dimensions_of_3x5_matrix_maps_r5_to_r3() {
        let t = Transformation::new(Matrix::new(3, 5));
        assert_eq!(t.domain_dim(), 5, "定義域 n = 行數(cols)");
        assert_eq!(t.codomain_dim(), 3, "對應域 m = 列數(rows)");
        assert_eq!(t.dimensions(), (5, 3), "(n, m):從 ℝ⁵ 映到 ℝ³");
    }

    /// 方陣是「不換空間」的轉換:ℝⁿ → ℝⁿ(如旋轉、剪切)。
    #[test]
    fn square_matrix_maps_within_same_space() {
        let t = Transformation::new(Matrix::identity(4));
        assert_eq!(t.dimensions(), (4, 4));
    }

    /// 1×n 的列矩陣把整個 ℝⁿ 壓到 ℝ¹(數線)—— 之後學 dot product 會再遇到它。
    #[test]
    fn row_matrix_collapses_to_r1() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0, 3.0]]));
        assert_eq!(t.dimensions(), (3, 1));
    }
}
