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
//! 單元 5-2(講義 2.7 後段)補上反向通道 **Theorem 2.9**:每個線性轉換
//! T: ℝⁿ → ℝᵐ 都由**唯一**的 m×n 矩陣誘導 —— [`standard_matrix`] 對 T 做
//! n 次標準基底取樣,把「函數」蒸餾回「資料」,與 `Transformation::new`
//! (資料 ↦ 函數)合起來構成一一對應。
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

    /// 此轉換的標準矩陣 A —— [`new`](Transformation::new)(A ↦ T_A)的反向讀取。
    ///
    /// 之所以是個**良定義**的數學操作,靠的是 Theorem 2.9:T ↦ A 唯一,
    /// 「轉換的標準矩陣」才有唯一答案可回。`range` 模組(單元 5-3)跨在本模組外,
    /// 經此通道把 A 餵給 rank / 行抽取等矩陣積木。借用而非拷貝:呼叫端只需讀。
    pub fn matrix(&self) -> &Matrix {
        &self.matrix
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

/// 標準矩陣(standard matrix)構造器:對映射 T 做 n 次標準基底取樣,
/// 蒸餾出誘導它的矩陣 A —— **A 的第 j 行(column)= T(eⱼ)**(Theorem 2.9)。
///
/// 與 [`verify_linearity`] 是一對:同樣收泛型 `F: Fn(&Vector) -> Vector`
/// (T 不必由矩陣誘導),一個**檢查**任意映射、一個**取樣**任意映射。
/// Theorem 2.9 的「若 T 線性」前提在這裡承重:非線性的 T 照樣能取樣出
/// 一個矩陣(構造器只看 eⱼ 上的值),但那個矩陣**重現不了** T ——
/// 見測試 `standard_matrix_of_nonlinear_map_builds_an_impostor`。
///
/// codomain 維度 m 不收參數、從 T 的輸出導出(維度從資料導出,不另存):
/// - `n == 0` → [`LinAlgError::EmptyInput`]:沒有基底可取樣,m 無從決定
///   (與 [`Vector::linear_combination`] 的空集合同款)。
/// - 各 T(eⱼ) 輸出維度不一致 → [`LinAlgError::DimensionMismatch`]
///   (T 根本不是到同一個 ℝᵐ 的函數)。
pub fn standard_matrix<F>(n: usize, t: F) -> Result<Matrix, LinAlgError>
where
    F: Fn(&Vector) -> Vector,
{
    // 沒有 e_j 可取樣,m 無從導出 —— 與 linear_combination 的空集合同款。
    if n == 0 {
        return Err(LinAlgError::EmptyInput);
    }
    // 取樣:images[j] = T(e_j)。j < n 是迴圈不變式,Vector::standard 的
    // Err(IndexOutOfRange)是被證明的死路 —— unwrap 安全(先守衛、再 unwrap)。
    let images: Vec<Vector> = (0..n)
        .map(|j| t(&Vector::standard(n, j).unwrap()))
        .collect();
    // m 從第一支影像導出(維度從資料導出,不另存);其餘影像必須住在同一個 ℝᵐ。
    let m = images[0].rows();
    if images.iter().any(|image| image.rows() != m) {
        return Err(LinAlgError::DimensionMismatch);
    }
    // 組裝(row-major 逐列收):第 i 列橫掃所有影像的第 i 個分量,
    // 等效於「T(e_j) 直放為 A 的第 j 行」—— A[i][j] = T(e_j)ᵢ,免去顯式 transpose。
    let rows: Vec<Vec<f64>> = (0..m)
        .map(|i| images.iter().map(|image| image.entries()[i]).collect())
        .collect();
    Ok(Matrix::from_rows(rows))
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

    /// 單元 5-2 練習 1 題目原例:T(x₁,x₂,x₃) = (3x₁−4x₂, 2x₁+x₃) 的標準矩陣。
    /// 注意 m = 2 沒被宣告 —— codomain 維度從 T 的輸出導出(維度從資料導出,
    /// 不另存),整數係數在 f64 下精確,可用精確 equals。
    #[test]
    fn standard_matrix_of_formula_example() {
        let t = |x: &Vector| {
            let e = x.entries();
            Vector::from_vec(vec![3.0 * e[0] - 4.0 * e[1], 2.0 * e[0] + e[2]])
        };
        let a = standard_matrix(3, t).unwrap();
        assert!(a.equals(&Matrix::from_rows(vec![
            vec![3.0, -4.0, 0.0],
            vec![2.0, 0.0, 1.0],
        ])));
        assert_eq!((a.rows(), a.cols()), (2, 3), "m×n = 2×3:ℝ³ → ℝ²");
    }

    /// 單一見證的 round-trip:矩陣 B 誘導的 T 取樣回來就是 B 自己 ——
    /// Theorem 2.9「唯一性」的具體一例,全稱版留給 mod laws(練習 3)。
    #[test]
    fn standard_matrix_recovers_inducing_matrix() {
        let b = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let t = Transformation::new(b.clone());
        let a = standard_matrix(2, |x| t.apply(x).unwrap()).unwrap();
        assert!(a.equals(&b));
    }

    /// 非線性 T 也能取樣出矩陣 —— 但它是冒牌貨,基底之外重現不了 T。
    /// 逐元素平方在 eⱼ 上說不了謊(0² = 0、1² = 1 → 取樣出 I),
    /// 但 T(2e₁) = 4e₁ ≠ I·(2e₁) = 2e₁:Theorem 2.9 的「若 T 線性」前提在承重。
    #[test]
    fn standard_matrix_of_nonlinear_map_builds_an_impostor() {
        let square = |x: &Vector| Vector::from_vec(x.entries().iter().map(|e| e * e).collect());
        let a = standard_matrix(2, square).unwrap();
        assert!(a.equals(&Matrix::identity(2)), "在基底上取樣看不出非線性");
        let x = Vector::from_vec(vec![2.0, 0.0]);
        let via_matrix = a.multiply_vector(&x).unwrap();
        assert!(!square(&x).equals(&via_matrix), "基底之外,冒牌貨就穿幫");
    }

    /// n = 0:沒有基底可取樣,codomain 維度無從導出 —— EmptyInput
    /// (沿 linear_combination 空集合的慣例:無從決定結果維度)。
    #[test]
    fn standard_matrix_rejects_empty_domain() {
        assert_eq!(
            standard_matrix(0, |x: &Vector| x.clone()).unwrap_err(),
            LinAlgError::EmptyInput
        );
    }

    /// 單元 5-2 練習 2:x 軸反射 —— 幾何規則經 standard_matrix 蒸餾成矩陣。
    /// 反射不進 core API:教學點是「幾何直觀 → 構造器 → 數值」這條工作流本身
    /// (寫**規則**,讓構造器去發現矩陣 —— 不是先想好矩陣再抄進去)。
    #[test]
    fn standard_matrix_of_x_axis_reflection() {
        // 「x 不動、y 翻號」((x, y) ↦ (x, −y)),矩陣留給構造器去發現。
        let reflect = |x: &Vector| -> Vector {
            let e = x.entries();
            Vector::from_vec(vec![e[0], -e[1]])
        };
        let a = standard_matrix(2, reflect).unwrap();
        assert!(
            a.equals(&Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, -1.0]])),
            "構造器應從規則發現 U(e₁) = e₁、U(e₂) = −e₂"
        );
        // 驗收:(3, 5) 反射後 (3, −5) —— 用蒸餾出的矩陣親自轉一次
        let y = a
            .multiply_vector(&Vector::from_vec(vec![3.0, 5.0]))
            .unwrap();
        assert!(y.equals(&Vector::from_vec(vec![3.0, -5.0])));
    }

    /// 單元 5-2 練習 4(一):單位轉換 I(x) = x 的標準矩陣就是 Iₙ。
    /// 題目要的 identity_matrix(n) 不必新刻 —— Matrix::identity 第二單元就有,
    /// 這裡對帳:構造器從「什麼都不動」的規則重新發現同一個矩陣(單一真相)。
    #[test]
    fn standard_matrix_of_identity_map_is_identity_matrix() {
        let i = |x: &Vector| x.clone();
        let a = standard_matrix(3, i).unwrap();
        assert!(a.equals(&Matrix::identity(3)));
    }

    /// 單元 5-2 練習 4(二):零轉換 T₀(x) = 0 的標準矩陣是零矩陣(= Matrix::new)。
    /// 陷阱:closure 只說「一切都到 ℝᵐ 的原點」—— m 由輸出導出、n 是取樣次數,
    /// 兩者解耦,挑 m ≠ n 的形狀(如 2×4)才驗得到「零轉換不必方陣」。
    #[test]
    fn standard_matrix_of_zero_map_is_zero_matrix() {
        let zero = |_: &Vector| Vector::new(2); // codomain 維度 m = 2,與 n = 4 解耦
        let a = standard_matrix(4, zero).unwrap();
        assert!(a.equals(&Matrix::new(2, 4)));
    }

    /// 輸出維度忽長忽短的「映射」根本不是到同一個 ℝᵐ 的函數 → DimensionMismatch。
    /// 這支 closure 對 e₁ 吐 ℝ¹、對 e₂ 吐 ℝ²(輸出長度 = 非零分量位置 + 1)。
    #[test]
    fn standard_matrix_rejects_inconsistent_codomain() {
        let ragged = |x: &Vector| {
            let j = x.entries().iter().position(|&e| e != 0.0).unwrap();
            Vector::new(j + 1)
        };
        assert_eq!(
            standard_matrix(2, ragged).unwrap_err(),
            LinAlgError::DimensionMismatch
        );
    }
}

/// Theorem 2.7 與 2.9 的全稱驗證。
///
/// `verify_linearity` 在單一樣本上只是「見證」;這裡用 proptest 升級成 for all
/// 形式:隨機產生矩陣 A 與樣本 (u, v, c),每一組都必須通過線性檢查 ——
/// 預設 256 組(比題目要求的 10 組多 25 倍),失敗會自動 shrink 成最小反例。
///
/// 單元 5-2 接力 **Theorem 2.9** 的兩個半邊:**唯一性**(round-trip:B 誘導的
/// T 取樣回來必是 B 自己)與**存在性**(T(v) = Av 對任意 v 成立)。這兩條的
/// **維度也隨機**(`prop_flat_map` 先抽形狀、再抽內容),涵蓋 ℝⁿ → ℝᵐ 的各種組合。
///
/// 沿 repo 慣例,兩種策略對應兩種比較:
/// - 整數策略:小整數在 f64 下加減乘**完全精確** → epsilon 可給 0.0(精確相等)。
/// - 真實浮點策略:捨入誤差使定律只在容差內成立 → `1e-9`。
#[cfg(test)]
mod laws {
    use super::*;
    use proptest::prelude::*;

    /// 固定 `rows×cols`、元素為 [-10, 10] 整數的矩陣(f64 下加減乘完全精確)。
    fn int_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-10i64..=10, cols), rows).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            )
        })
    }

    /// 長度 `n`、元素為 [-10, 10] 整數的向量。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()))
    }

    /// 固定 `rows×cols`、元素為 [-100, 100] 真實浮點的矩陣。
    fn real_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-100.0f64..100.0, cols), rows)
            .prop_map(Matrix::from_rows)
    }

    /// 長度 `n`、元素為 [-100, 100] 真實浮點的向量。
    fn real_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-100.0f64..100.0, n).prop_map(Vector::from_vec)
    }

    /// 隨機形狀(1..=4 × 1..=4)的整數矩陣 —— `prop_flat_map` 的教學點:
    /// tuple 策略的各分量**獨立**抽樣,但「先抽形狀、再抽該形狀的矩陣」是
    /// **有依賴**的兩步 —— flat_map 讓後面的策略吃到前面抽出的值。
    fn int_matrix_any_shape() -> impl Strategy<Value = Matrix> {
        (1usize..=4, 1usize..=4).prop_flat_map(|(m, n)| int_matrix(m, n))
    }

    /// 隨機形狀的真實浮點矩陣,連同一支**長度 = 矩陣行數**的向量 ——
    /// v 必須住在 T 的定義域 ℝⁿ,所以 n 得在同一次 flat_map 裡共用。
    fn real_matrix_with_vector() -> impl Strategy<Value = (Matrix, Vector)> {
        (1usize..=4, 1usize..=4).prop_flat_map(|(m, n)| (real_matrix(m, n), real_vector(n)))
    }

    proptest! {
        // Theorem 2.7(整數版,精確):任何 3×4 小整數矩陣誘導的 T_A 都通過
        // 線性檢查。整數運算在 f64 下完全精確 —— epsilon 給 0.0,一絲不差。
        #[test]
        fn theorem_2_7_matrix_transformations_are_linear_exact(
            a in int_matrix(3, 4),
            u in int_vector(4),
            v in int_vector(4),
            c in -10i64..=10,
        ) {
            let t = Transformation::new(a);
            prop_assert!(verify_linearity(
                |x| t.apply(x).unwrap(),
                &u,
                &v,
                c as f64,
                0.0
            ));
        }

        // Theorem 2.7(真實浮點版,容差):同一條定理在真實浮點下也成立,但
        // T(u+v) 與 T(u)+T(v) 的捨入路徑不同,只能在 1e-9 容差內相等 ——
        // 與整數版合起來,正是「整數配精確、浮點配近似」的慣例本身。
        #[test]
        fn theorem_2_7_holds_for_real_matrices_within_tolerance(
            a in real_matrix(3, 4),
            u in real_vector(4),
            v in real_vector(4),
            c in -100.0f64..100.0,
        ) {
            let t = Transformation::new(a);
            prop_assert!(verify_linearity(
                |x| t.apply(x).unwrap(),
                &u,
                &v,
                c,
                1e-9
            ));
        }

        // Theorem 2.9(唯一性半邊,整數精確):隨機形狀的 B 誘導 T_B,
        // standard_matrix 從 T_B 取樣重建的矩陣必須 == B 一絲不差 ——
        // 「線性轉換的標準矩陣唯一」寫成可跑的定理。
        //   提醒:laws 一律 prop_assert!;n = b.cols() ≥ 1,unwrap 有守衛。
        #[test]
        fn theorem_2_9_standard_matrix_recovers_inducing_matrix(b in int_matrix_any_shape()) {
            let t = Transformation::new(b.clone());
            let a = standard_matrix(t.domain_dim(), |x| t.apply(x).unwrap()).unwrap();
            prop_assert!(a.equals(&b));
        }

        // Theorem 2.9(存在性半邊,真實浮點):題目原話 —— 對任意隨機 v,
        // 「直接呼叫 T(v)」與「左乘標準矩陣 Av」在 1e-9 容差內相同;
        // 形狀隨機,涵蓋 ℝ² → ℝ³ 等各種維度的映射。
        //   比對 t.apply(&v) 與 a.multiply_vector(&v) 兩條路的結果。
        #[test]
        fn theorem_2_9_transformation_agrees_with_matrix_multiplication(
            (b, v) in real_matrix_with_vector(),
        ) {
            let t = Transformation::new(b.clone());
            let a = standard_matrix(t.domain_dim(), |x| t.apply(x).unwrap()).unwrap();
            let via_t = t.apply(&v).unwrap();
            let via_a = a.multiply_vector(&v).unwrap();
            prop_assert!(via_t.approx_equals(&via_a, 1e-9));
        }
    }
}
