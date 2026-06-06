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

use crate::{LinAlgError, Matrix, Vector};

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

    /// 施作轉換:T_A(x) = Ax,把 x ∈ ℝⁿ 送到它的影像(image)y ∈ ℝᵐ。
    ///
    /// 這就是「矩陣–向量乘法的函數視角」—— 計算本身第四單元已經刻好
    /// ([`Matrix::multiply_vector`]),這裡的學習點是**認出它**:
    /// 不重寫演算法,單一真相只有一份。
    ///
    /// `x` 不在定義域(`x.rows() != n`)→ [`LinAlgError::DimensionMismatch`],
    /// 由乘法本身的維度檢查傳播上來 —— 驗證規則同樣只有單一真相。
    pub fn apply(&self, x: &Vector) -> Result<Vector, LinAlgError> {
        self.matrix.multiply_vector(x)
    }

    /// 單位轉換(identity transformation)I: ℝⁿ → ℝⁿ,**I(x) = x** ——
    /// 由單位矩陣 Iₙ 誘導,什麼都不動的轉換。
    ///
    /// 矩陣生成邏輯第二單元已刻好([`Matrix::identity`]),這裡是它的「函數視角」。
    pub fn identity(n: usize) -> Transformation {
        Transformation::new(Matrix::identity(n))
    }

    /// 零轉換(zero transformation)T₀: ℝⁿ → ℝᵐ,**T₀(x) = 0** ——
    /// 由 m×n 零矩陣誘導,把整個空間吸到原點的轉換。
    ///
    /// 注意零轉換**不必是方陣**:輸出的零向量 0 ∈ ℝᵐ(codomain),維度可與輸入不同
    /// —— 參數順序 (m, n) 沿矩陣慣稱「m×n = 列×行」。
    pub fn zero(m: usize, n: usize) -> Transformation {
        Transformation::new(Matrix::new(m, n))
    }
}

/// 線性性質驗證器:檢查映射 T 在**一組樣本** (u, v, c) 上是否滿足
/// 線性轉換定義的兩大守恆:
///
/// 1. 加法守恆:T(u + v) = T(u) + T(v)
/// 2. 純量乘守恆:T(c·u) = c·T(u)
///
/// **為什麼是 free function、不是 `Transformation` 的方法?** 線性是對
/// 「任意映射」的定義 —— T 不必由矩陣誘導。泛型 `F: Fn(&Vector) -> Vector`
/// 收任何「向量進、向量出」的函數(closure、fn pointer、包了 `apply` 的
/// closure⋯⋯),Theorem 2.7 才回頭說:**矩陣誘導的那種必過此檢查**。
/// 定義(這裡)與定理(`mod laws`)分開放,概念的依賴方向才對。
///
/// **見證與反例的不對稱**:通過一組樣本只是「見證」,不能證明線性
/// (那是 for all 命題,見 laws 的 proptest);但**失敗一組樣本就證明了非線性**
/// —— 這正是它能識破仿射轉換 T(x) = 2x + 3 的原因。
///
/// u、v 不在同一空間(維度不合),或 T 的輸出維度前後不一致 → `false`
/// (沿 [`Vector::is_parallel`] 的慣例:不在同一空間就談不上守恆)。
/// 浮點比較的容差 `epsilon` 由呼叫端視運算數量級指定。
pub fn verify_linearity<F>(t: F, u: &Vector, v: &Vector, c: f64, epsilon: f64) -> bool
where
    F: Fn(&Vector) -> Vector,
{
    if u.rows() != v.rows() {
        return false; // u、v 不在同一空間,談不上守恆
    }
    let tu = t(u);
    let tv = t(v);
    if tu.rows() != tv.rows() {
        return false; // T 的輸出維度前後不一致,談不上守恆
    }
    let left_add = t(&u.add(v).unwrap());
    let right_add = tu.add(&tv).unwrap();
    let left_scale = t(&u.scale(c));
    let right_scale = tu.scale(c);
    left_add.approx_equals(&right_add, epsilon) && left_scale.approx_equals(&right_scale, epsilon)
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

    /// 練習 2 題目原例:投影到 xy 平面 —— z 分量歸零、x 與 y 不動。
    /// 幾何上是「壓扁」:整個 ℝ³ 被拍到 z = 0 的平面上。
    #[test]
    fn apply_projects_onto_xy_plane() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]));
        let y = t.apply(&Vector::from_vec(vec![1.0, 2.0, 3.0])).unwrap();
        assert!(y.equals(&Vector::from_vec(vec![1.0, 2.0, 0.0])));
    }

    /// 剪切(shear)—— 驗收條件點名的變換:x 分量被 y 分量「推」k 倍、y 不動。
    /// [1, 1] 經 k = 2 的水平剪切 → [1 + 2·1, 1] = [3, 1]。
    #[test]
    fn apply_shears_along_x_axis() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![0.0, 1.0]]));
        let y = t.apply(&Vector::from_vec(vec![1.0, 1.0])).unwrap();
        assert!(y.equals(&Vector::from_vec(vec![3.0, 1.0])));
    }

    /// 非方陣換空間:2×3 矩陣把 ℝ³ 的向量送進 ℝ² —— 影像落在 codomain。
    #[test]
    fn apply_image_lands_in_codomain() {
        let t = Transformation::new(Matrix::from_rows(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ]));
        let y = t.apply(&Vector::from_vec(vec![7.0, 8.0, 9.0])).unwrap();
        assert_eq!(y.rows(), t.codomain_dim());
    }

    /// x ∉ ℝⁿ 就不在函數的定義範圍內 —— 錯誤是值(DimensionMismatch),不 panic。
    #[test]
    fn apply_rejects_vector_outside_domain() {
        let t = Transformation::new(Matrix::new(3, 5));
        let x = Vector::from_vec(vec![1.0, 2.0]); // ℝ²,但定義域是 ℝ⁵
        assert_eq!(t.apply(&x).unwrap_err(), LinAlgError::DimensionMismatch);
    }

    /// 練習 3:矩陣誘導的轉換(shear)通過線性檢查 —— Theorem 2.7 的單一見證,
    /// 全稱版見 mod laws。closure 包住 apply,把 Transformation 餵進泛型驗證器。
    #[test]
    fn verify_linearity_passes_matrix_transformation() {
        let t = Transformation::new(Matrix::from_rows(vec![vec![1.0, 2.0], vec![0.0, 1.0]]));
        let u = Vector::from_vec(vec![1.0, -2.0]);
        let v = Vector::from_vec(vec![3.0, 0.5]);
        assert!(verify_linearity(
            |x| t.apply(x).unwrap(),
            &u,
            &v,
            -1.5,
            1e-9
        ));
    }

    /// 驗收條件:識破仿射轉換 T(x) = 2x + 3。「線性」要求過原點,平移破壞它:
    /// T(u+v) = 2(u+v)+3,但 T(u)+T(v) = 2(u+v)+6 —— 常數項被加了兩次。
    #[test]
    fn verify_linearity_rejects_affine_map() {
        let affine = |x: &Vector| {
            let shift = Vector::from_vec(vec![3.0; x.rows()]);
            x.scale(2.0).add(&shift).unwrap()
        };
        let u = Vector::from_vec(vec![1.0, 2.0]);
        let v = Vector::from_vec(vec![0.0, -1.0]);
        assert!(!verify_linearity(affine, &u, &v, 2.0, 1e-9));
    }

    /// c = 1 時純量乘守恆對**任何**映射都退化成立(T(1·u) = 1·T(u) 是恆等式),
    /// 仿射映射只能靠加法守恆抓 —— 證明兩個條件缺一不可、不能只查其中一個。
    #[test]
    fn verify_linearity_catches_affine_by_additivity_when_scaling_degenerates() {
        let affine = |x: &Vector| {
            let shift = Vector::from_vec(vec![3.0; x.rows()]);
            x.scale(2.0).add(&shift).unwrap()
        };
        let u = Vector::from_vec(vec![1.0, 2.0]);
        let v = Vector::from_vec(vec![0.0, -1.0]);
        assert!(!verify_linearity(affine, &u, &v, 1.0, 1e-9));
    }

    /// u、v 不在同一空間 → u+v 根本不存在,談不上守恆 —— 沿 is_parallel 慣例回 false。
    #[test]
    fn verify_linearity_rejects_mismatched_spaces() {
        let id = |x: &Vector| x.clone();
        let u = Vector::from_vec(vec![1.0]);
        let v = Vector::from_vec(vec![1.0, 2.0]);
        assert!(!verify_linearity(id, &u, &v, 2.0, 1e-9));
    }

    /// 練習 4 驗收(一):單位轉換後向量**完全不變**(Iₙ·x 每項是 1·xᵢ + 0 的和,
    /// 浮點下精確,可用精確 equals)。
    #[test]
    fn identity_transformation_fixes_every_vector() {
        let i = Transformation::identity(3);
        let x = Vector::from_vec(vec![1.5, -2.0, 3.25]);
        assert!(i.apply(&x).unwrap().equals(&x));
        assert_eq!(i.dimensions(), (3, 3), "I 不換空間:ℝ³ → ℝ³");
    }

    /// 練習 4 驗收(二):零轉換的影像必為全零向量 —— 且落在 codomain ℝᵐ,
    /// 不是 domain ℝⁿ(2×4 的零矩陣把 ℝ⁴ 吸到 ℝ² 的原點)。
    #[test]
    fn zero_transformation_sends_everything_to_origin() {
        let z = Transformation::zero(2, 4);
        let y = z
            .apply(&Vector::from_vec(vec![1.0, -2.0, 3.0, 4.0]))
            .unwrap();
        assert!(y.is_zero());
        assert_eq!(y.rows(), 2, "零向量 0 ∈ ℝᵐ(codomain),維度與輸入不同");
    }

    /// identity 與 zero 都通過線性檢查 —— 它們是「最簡單的兩個線性轉換」,
    /// 也是 Theorem 2.7 的特例(各由 Iₙ、0ₘₓₙ 誘導)。
    #[test]
    fn identity_and_zero_transformations_are_linear() {
        let u = Vector::from_vec(vec![1.0, 2.0]);
        let v = Vector::from_vec(vec![-3.0, 0.5]);
        let i = Transformation::identity(2);
        let z = Transformation::zero(3, 2);
        assert!(verify_linearity(|x| i.apply(x).unwrap(), &u, &v, 2.5, 1e-9));
        assert!(verify_linearity(|x| z.apply(x).unwrap(), &u, &v, 2.5, 1e-9));
    }
}
