//! 特徵值與特徵向量(Eigenvalues & Eigenvectors)—— 單元 8-1 – 8-3(講義 5.1–5.2)。
//!
//! 這是課程進入「進階主題」的第一章。前面所有轉換都把整個空間搬來搬去;這裡問一個
//! 更尖銳的問題:對一個運算子 T(方陣 A),**哪些非零向量 v 只被「伸縮」、方向不變**?
//! 即存在純量 λ 使
//!
//! > **A·v = λ·v**(v ≠ 0)
//!
//! 這樣的 v 叫**特徵向量(eigenvector)**、λ 叫對應的**特徵值(eigenvalue)**。把
//! A·v = λv 移項成 **(A − λI)·v = 0**,特徵向量就是 **Null(A − λI) 裡的非零向量** ——
//! 一句話把這整章接回子空間機器:**特徵空間 Eλ = Null(A − λI)**。
//!
//! ## 這章新增什麼(reuse 分析:5 題 → 2 新函式 + 1 新子空間積木 + 定理走 laws)
//!
//! 沿用本專案慣例(見 `prefer-reuse-and-laws-over-new-api`):只有「帶來新計算」的
//! 才進公開 API,定理寫成 `mod laws` 的隨機驗證。
//!
//! - **練習 1** [`is_eigenpair`]:`A·v = λv` 且 v ≠ 0 的定義性檢查 —— 全章的判準基石。
//! - **練習 3** [`characteristic_matrix`]:`A − λI` —— 把「找特徵向量」翻譯成「找零空間」的
//!   閘門矩陣。
//! - **練習 4** [`eigenspace_basis`]:`Eλ = Null(A − λI)` 的一組基底 —— 串接練習 3 與
//!   **本章真正的新演算法** [`Transformation::null_space_basis`](crate::Transformation::null_space_basis)
//!   (零空間基底萃取器)。
//!   筆記說「沿用 6-2 的 NullSpace 邏輯」,但本 repo 的 6-2 只刻了
//!   [`null_space_contains`](crate::Transformation::null_space_contains)(會員判定)與
//!   [`nullity`](crate::Matrix::nullity)(只數維度)—— **從未有基底萃取器**,故它在
//!   `subspace.rs` 補上,正好補齊「矩陣子空間基底三兄弟」:`range_basis`(Col A)、
//!   `row_space_basis`(Row A)、**`null_space_basis`(Null A)**。
//! - **練習 5** [`has_real_eigenvalues_2x2`]:2×2 特徵多項式判別式 —— 90° 旋轉沒有實特徵值
//!   的幾何直觀(正式算法在 5.2 節)。
//! - **練習 2**(運算子版特徵檢查)**不進 API**:運算子的特徵向量 = 其標準矩陣的特徵向量,
//!   這是定理不是新計算 —— 經 [`standard_matrix`](crate::standard_matrix) 橋接 closure 與
//!   矩陣,寫成 `mod laws` 的 `operator_eigen_agrees_with_standard_matrix`。
//!
//! ## 單元 8-2:特徵多項式 1/2(講義 5.2 前段)—— 6 題 → 3 新計算 + 3 題吃老本
//!
//! 8-1 給了「驗證」的工具(給你 λ,我驗它是不是特徵值);8-2 回答「怎麼**找**」:
//! `(A − tI)·v = 0` 有非零解 ⟺ `A − tI` 奇異 ⟺ **`det(A − tI) = 0`(特徵方程)**。
//! `det(A − tI)` 是 t 的 n 次多項式(**特徵多項式**),特徵值就是它的根。
//!
//! - **練習 1**(`A − tI` 構造)= 8-1 的 [`characteristic_matrix`],**零新碼** ——
//!   補筆記原例的具體測試(stub 階段就綠,reuse 的見證)。
//! - **練習 2**(2×2 特徵多項式係數 `t² − tr(A)·t + det(A)`)→ 新積木 [`Matrix::trace`]。
//!   「係數包裝函式」只是重述公式、不進 API;law
//!   `char_poly_2x2_expands_via_trace_and_det` 直接隨機驗**展開本身**(比回傳係數更強)。
//! - **練習 3** [`triangular_eigenvalues`]:三角矩陣特徵值 = 主對角線直讀(Property 3)——
//!   真正的新計算,配新錯誤 [`LinAlgError::NotTriangular`]。
//! - **練習 4**(相似矩陣共享特徵多項式)是定理不是計算 → law
//!   `similar_matrices_share_characteristic_polynomial`(`B = P⁻¹AP`,隨機 t 對帳兩邊 det)。
//! - **練習 5** [`algebraic_multiplicity`]:代數重數計數器 —— 練習 3 的輸出是它的輸入,
//!   下一單元的對角化判準(代數 vs 幾何重數)是它的下游。
//! - **練習 6**(90° 旋轉無實根)= 8-1 的 [`has_real_eigenvalues_2x2`],**零新碼** →
//!   law 對帳兩條判別式:`tr² − 4·det`(係數路)≡ `(a−d)² + 4bc`(8-1 路)。
//!
//! ## 單元 8-3:特徵多項式 2/2(講義 5.2 後段)—— 5 題 → 1 新函式 + 1 報告 struct
//!
//! 主角是兩種「重數」的張力:**代數重數**(λ 作為特徵多項式根的重疊次數,8-2 已刻)vs
//! **幾何重數**(`dim Eλ = nullity(A − λI)`,λ 實際提供多少個線性獨立特徵向量),以及
//! **Theorem 5.1:幾何重數 ≤ 代數重數**。幾何 < 代數的特徵值稱**虧損(deficient)**——
//! 特徵多項式承諾了 k 份、特徵空間卻交不出 k 個獨立向量,是下一單元「不可對角化」的元凶。
//!
//! - **練習 1**(代數重數)= 8-2 的 [`algebraic_multiplicity`],**零新碼**(連原例
//!   `[−1, 3, 3] → 2` 都是同一題,測試已在)。
//! - **練習 2** [`geometric_multiplicity`]:計算是舊的(8-1 law
//!   `eigenspace_dimension_equals_nullity` 早證了 nullity 路 = 基底路),但概念是本單元
//!   主角、練習 4 與下一單元的對角化判準都是下游 —— 值得一個名字。
//! - **練習 3**(Theorem 5.1 驗證器)是定理不是計算 → law
//!   `theorem_5_1_geometric_at_most_algebraic`(三角家族 —— 代數重數可直讀,誠實對應
//!   repo 尚無一般求根器);外加對照律 `diagonal_matrices_are_never_deficient`
//!   (對角矩陣幾何 = 代數,預告「對角 ⟹ 可對角化」)。
//! - **練習 4** [`EigenReport`] / [`analyze_eigen_properties`]:整合報告 —— 沿
//!   [`TransformationReport`](crate::TransformationReport) 先例(pub 欄位的值物件,
//!   「報告與獨立計算器一致」是 laws 要證的,不鎖建構子)。
//! - **練習 5**(複數特徵值偵測)**零新碼**:2×2 有複數特徵值 ⟺ 判別式 < 0 ⟺
//!   [`has_real_eigenvalues_2x2`] 的**逆否**,8-1 旋轉測試 + 8-2 判別式 law 已全覆蓋,
//!   再包一層函式只是在 API 裡放一個 `!`。
//!
//! 與 `transformation` / `subspace` 同款佈局:本模組碰不到 private 的 `data`,一律走
//! public API。

use crate::{LinAlgError, Matrix, Transformation, Vector};

/// **練習 1 —— 特徵對(eigenpair)驗證器**:v 是否為 A 對應特徵值 λ 的特徵向量,即
/// **A·v = λv** 且 **v ≠ 0**(容差 `epsilon` 內)。
///
/// 兩個非顯而易見的判準,缺一不可:
/// - **v 必須非零**:依定義特徵向量不能是零向量(否則 A·0 = λ·0 對**任何** λ 都成立,
///   λ 就失去意義)。零向量 → `false`。
/// - **維度 / 形狀不合 → `false`**:A 非方陣、或 v 不在 A 的定義域(`A·v` 與 `λv` 維度對不上)
///   時,談不上特徵對(沿 [`null_space_contains`](crate::Transformation::null_space_contains)
///   / [`verify_linearity`](crate::verify_linearity) 的 bool 述詞慣例 —— 不在同一空間就回
///   `false`,不 panic)。
///
/// 實作提示:[`Matrix::multiply_vector`] 算 `A·v`(回 `Result`,維度檢查隨之繼承 ——
/// `Err` 恰好折成 `false`);λv 是 [`Vector::scale`];兩者用
/// [`Vector::approx_equals`]`(_, epsilon)` 比;非零用 [`Vector::is_approx_zero`]`(epsilon)`
/// 的反面。先擋零向量,再算、再比。
pub fn is_eigenpair(epsilon: f64, a: &Matrix, v: &Vector, lambda: f64) -> bool {
    if v.is_approx_zero(epsilon) {
        return false;
    }
    match a.multiply_vector(v) {
        Ok(av) => av.approx_equals(&v.scale(lambda), epsilon),
        Err(_) => false,
    }
}

/// **練習 3 —— 特徵閘門矩陣**:回傳 **M = A − λI**(`I` 為同階單位矩陣)。
///
/// 這是全章的轉軸:`A·v = λv ⟺ (A − λI)·v = 0`,於是「找特徵向量」變成「找
/// `M = A − λI` 的零空間非零向量」。**只動主對角線**(每個 `A[i][i]` 減 λ),其餘元素照搬。
///
/// 回 `Result`:`A − λI` 只對**方陣**有定義(λI 要同階),非方陣 → [`LinAlgError::NotSquare`]
/// (沿 [`Matrix::power`](crate::Matrix::power) 的先例:同樣「需要方陣」的運算)。
///
/// 實作提示:守住 [`Matrix::is_square`] 後,`λI` = [`Matrix::identity`]`(n)` 再
/// [`Matrix::scalar_multiply`]`(λ)`;`A − λI` 可寫成 `A + (−λ)I`,用 [`Matrix::add`]
/// (兩者皆 n×n,守門後相加不會失敗 —— 同 `power` 的 `expect` 手法,別假裝它會錯)。
pub fn characteristic_matrix(a: &Matrix, lambda: f64) -> Result<Matrix, LinAlgError> {
    if !a.is_square() {
        return Err(LinAlgError::NotSquare {
            rows: a.rows(),
            cols: a.cols(),
        });
    }
    let n = a.rows();
    let identity = Matrix::identity(n).scalar_multiply(lambda);
    Ok(a.add(&identity.scalar_multiply(-1.0))
        .expect("characteristic_matrix: A, λI 同階,加法不會失敗;若失敗,是程式 bug"))
}

/// **練習 4 —— 特徵空間基底**:`Eλ = Null(A − λI)` 的一組基底向量。
///
/// 特徵空間是「對應特徵值 λ 的所有特徵向量,再加上零向量」——它恰好是
/// `M = A − λI` 的零空間(故是子空間)。回傳的每個向量都滿足 `A·v = λv`、且彼此線性獨立
/// (由零空間基底演算法保證);其**個數 = `Eλ` 的維度 = λ 的幾何重數 = nullity(A − λI)**。
///
/// λ **不是** A 的特徵值時,`Eλ = {0}` → **回空 `Vec`**(無非零特徵向量,維度 0)。
///
/// 實作提示:這題幾乎是一行的合縫 —— [`characteristic_matrix`]`(a, lambda)?` 取得
/// `M`,包成 [`Transformation::new`](crate::Transformation::new),呼叫本章新增的
/// [`null_space_basis`](crate::Transformation::null_space_basis)`(epsilon)`。
/// `Result` 沿 `characteristic_matrix` 的非方陣失敗一路傳上來(`?`)。
pub fn eigenspace_basis(epsilon: f64, a: &Matrix, lambda: f64) -> Result<Vec<Vector>, LinAlgError> {
    let m = characteristic_matrix(a, lambda)?;
    let t = Transformation::new(m);
    Ok(t.null_space_basis(epsilon))
}

/// **練習 5 —— 2×2 是否有實特徵值**:用特徵多項式 `t² − tr(A)·t + det(A) = 0` 的**判別式**
/// 判定(預習 5.2 的幾何直觀)。
///
/// 對 `A = [[a, b], [c, d]]`,判別式 `Δ = tr² − 4·det = (a − d)² + 4bc`。**Δ ≥ 0 ⟺ 有實特徵值**
/// (Δ < 0 時兩特徵值是共軛複數 —— 如 90° 旋轉把每個向量轉開、沒有方向被保留)。
///
/// 驗收觀念:**對稱矩陣**(b = c)必有實特徵值 —— 此時 `Δ = (a − d)² + 4b² ≥ 0` 恆成立
/// (見 laws `symmetric_2x2_always_has_real_eigenvalues`)。
///
/// 非 2×2 輸入 → `false`(本判別式只對 2×2 成立,沿 bool 述詞「形狀不合即 false」慣例;
/// 不用 epsilon —— 旋轉的 Δ 明顯為負、對稱的明顯 ≥ 0,邊界 `Δ == 0` 視為有實根)。
///
/// 實作提示:守住 `rows() == 2 && cols() == 2` 後,用 [`Matrix::row`] 取出 a, b, c, d,
/// 直接算 `(a − d)² + 4bc >= 0`(比 `tr² − 4det` 少一次相減、數值更穩)。
pub fn has_real_eigenvalues_2x2(a: &Matrix) -> bool {
    if a.rows() != 2 || a.cols() != 2 {
        return false;
    }
    let row0 = a.row(0).unwrap();
    let row1 = a.row(1).unwrap();
    let a = row0[0];
    let b = row0[1];
    let c = row1[0];
    let d = row1[1];
    (a - d).powi(2) + 4.0 * b * c >= 0.0
}

// ───────────────────── 單元 8-2:特徵多項式 1/2(講義 5.2 前段) ─────────────────────

impl Matrix {
    /// **8-2 練習 2 的新積木 —— 跡(trace)**:主對角線元素之和 `tr(A) = Σ aᵢᵢ`。
    ///
    /// 在特徵多項式裡它是一次項係數:對 2×2,`det(A − tI) = t² − tr(A)·t + det(A)`
    /// (law `char_poly_2x2_expands_via_trace_and_det` 隨機驗證這個展開)。與 det 同為
    /// **相似不變量**(`P⁻¹AP` 不改變 trace)—— 5.2 後段「trace = Σ 特徵值」會回頭用它。
    ///
    /// 非方陣的「主對角線」走不到底、談不上跡 → [`LinAlgError::NotSquare`]
    /// (沿 [`characteristic_matrix`] / [`Matrix::power`](crate::Matrix::power) 的先例)。
    ///
    /// 實作提示:守住 [`Matrix::is_square`] 後,`(0..n)` 逐 i 取 [`Matrix::row`]`(i)[i]`
    /// 加總(guard 後 `row` 不會失敗,`expect` 即可;本模組碰不到 private `data`)。
    pub fn trace(&self) -> Result<f64, LinAlgError> {
        if !self.is_square() {
            return Err(LinAlgError::NotSquare {
                rows: self.rows(),
                cols: self.cols(),
            });
        }
        let n = self.rows();
        let mut sum = 0.0;
        for i in 0..n {
            let row = self
                .row(i)
                .expect("trace: is_square guard 後 row 不會失敗;若失敗,是程式 bug");
            sum += row[i];
        }
        Ok(sum)
    }
}

/// **練習 3 —— 三角矩陣特徵值直讀**:上或下三角矩陣的特徵值就是**主對角線元素**
/// (Property 3)。回傳照對角線順序、**保留重複** —— 重複次數正是該特徵值的代數重數
/// (交給 [`algebraic_multiplicity`] 數)。
///
/// 為什麼成立:A 三角 ⟹ `A − tI` 也三角 ⟹ `det(A − tI) = Π (aᵢᵢ − t)`
/// (Theorem 3.2,同 [`Matrix::determinant_triangular`](crate::Matrix::determinant_triangular)
/// 的對角線乘積),根一字排開就是各 `aᵢᵢ`。
///
/// 兩層失敗各自精確(沿 [`Matrix::inverse`](crate::Matrix::inverse) 的
/// NotSquare / NotInvertible 先例):
/// - 非方陣 → [`LinAlgError::NotSquare`](談不上特徵值);
/// - 方陣但非三角 → [`LinAlgError::NotTriangular`](直讀失效 —— 一般矩陣要走特徵方程)。
///
/// 實作提示:守住 [`Matrix::is_square`] 後,用 5-5 的
/// [`is_upper_triangular`](crate::Matrix::is_upper_triangular) /
/// [`is_lower_triangular`](crate::Matrix::is_lower_triangular)`(epsilon)`(兩者其一即過),
/// 再 `(0..n)` 收集 `row(i)[i]`。
pub fn triangular_eigenvalues(epsilon: f64, a: &Matrix) -> Result<Vec<f64>, LinAlgError> {
    if !a.is_square() {
        return Err(LinAlgError::NotSquare {
            rows: a.rows(),
            cols: a.cols(),
        });
    }
    if !a.is_upper_triangular(epsilon) && !a.is_lower_triangular(epsilon) {
        return Err(LinAlgError::NotTriangular);
    }
    let n = a.rows();
    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        let row = a
            .row(i)
            .expect("triangular_eigenvalues: is_square guard 後 row 不會失敗;若失敗,是程式 bug");
        eigenvalues.push(row[i]);
    }
    Ok(eigenvalues)
}

/// **練習 5 —— 代數重數(algebraic multiplicity)**:λ 在特徵值列表(特徵多項式的根,
/// 含重複)中出現的次數。
///
/// 列表哪來?本章唯一的來源是 [`triangular_eigenvalues`](重根天然入列,如 `[−1, 3, 3]`
/// 裡 3 的代數重數 = 2);5.2 後段的一般求根會接上同一個計數器。與**幾何重數**
/// (`dim Eλ` = [`eigenspace_basis`] 的向量個數)相對 —— 下一單元的對角化判準就是比較
/// 這兩個數(見測試 `algebraic_multiplicity_can_exceed_geometric`:剪切的 3 是
/// 代數 2、幾何 1)。
///
/// 浮點的「同一個根」用量值 ≤ `epsilon` 判(數值殘差不該拆散重根);找不到 → 0
/// (「λ 不是特徵值」是合法答案,不是錯誤)。
///
/// 實作提示:`iter().filter(…).count()` 一條函數式管線就夠。
pub fn algebraic_multiplicity(epsilon: f64, eigenvalues: &[f64], lambda: f64) -> usize {
    eigenvalues
        .iter()
        .filter(|&&ev| (ev - lambda).abs() <= epsilon)
        .count()
}

// ───────────────────── 單元 8-3:特徵多項式 2/2(講義 5.2 後段) ─────────────────────

/// **練習 2 —— 幾何重數(geometric multiplicity)**:`dim Eλ = nullity(A − λI)`,
/// λ 實際提供多少個線性獨立的特徵向量。
///
/// 與代數重數的張力是本單元主角:代數重數是特徵多項式「承諾」的份數,幾何重數是
/// 特徵空間「交得出」的獨立向量數。**Theorem 5.1:幾何 ≤ 代數**(law 見證);
/// 幾何 < 代數的 λ 稱**虧損(deficient)**,是「不可對角化」的元凶(下一單元)。
///
/// 非方陣 → [`LinAlgError::NotSquare`](沿 [`characteristic_matrix`] 一路傳上來);
/// λ 不是特徵值 → `0`(`Eλ = {0}`,維度 0 —— 合法答案,不是錯誤)。
///
/// 實作提示 —— 兩條等價路(8-1 law `eigenspace_dimension_equals_nullity` 已證相等),
/// 挑一條、在 body 裡留一句為什麼:
/// - **數維度**:[`characteristic_matrix`]`(a, λ)?` 接
///   [`Matrix::nullity`](crate::Matrix::nullity)`(ε)` —— 只算 rank,不必造出基底;
/// - **數基底**:[`eigenspace_basis`]`(ε, a, λ)?` 取 `.len()` —— 重用 8-1 全鏈,
///   多做了「把基底真的造出來」的工。
pub fn geometric_multiplicity(epsilon: f64, a: &Matrix, lambda: f64) -> Result<usize, LinAlgError> {
    let m = characteristic_matrix(a, lambda)?;
    Ok(m.nullity(epsilon))
}

/// **練習 4 —— 單一特徵值的重數報告**:代數 / 幾何重數並排,虧損與否一眼可判。
///
/// 沿 [`TransformationReport`](crate::TransformationReport) 先例:pub 欄位的值物件,
/// 欄位各自獨立計算 ——「報告與獨立計算器一致」「幾何 ≤ 代數」是 laws 要證的定理,
/// 不鎖進建構子(鎖了反而蓋掉值得證明的東西)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EigenReport {
    /// 特徵值 λ。
    pub value: f64,
    /// 代數重數:λ 在特徵多項式根列表中的出現次數(見 [`algebraic_multiplicity`])。
    pub algebraic: usize,
    /// 幾何重數:`dim Eλ = nullity(A − λI)`(見 [`geometric_multiplicity`])。
    pub geometric: usize,
    /// 虧損(deficient):幾何 < 代數 —— λ 交不出足額的獨立特徵向量;
    /// 下一單元將看到:**任何一個 λ 虧損 ⟹ A 不可對角化**。
    pub is_deficient: bool,
}

/// **練習 4 —— 特徵值性質綜合報告**:對 `eigenvalues`(特徵多項式的根,**含重複**,
/// 如 [`triangular_eigenvalues`] 的輸出)裡每個**相異**特徵值產出一份 [`EigenReport`]。
///
/// 設計約定:
/// - **每個相異 λ 恰一份報告**(重根合併 —— 重複的資訊已收進 `algebraic` 欄位),
///   順序照**首次出現**;「相異」用量值 ≤ `epsilon` 判(與 [`algebraic_multiplicity`]
///   同一把尺)。
/// - 非方陣 → [`LinAlgError::NotSquare`](幾何重數那條路傳上來);空列表 → 空報告。
///
/// 實作提示:走訪 `eigenvalues`,拿「已出報告的 λ」清單跳過容差內重複(`Vec<f64>` +
/// `iter().any(…)` 即可);每個新 λ 各叫一次 [`algebraic_multiplicity`] 與
/// [`geometric_multiplicity`](`?` 傳播錯誤),`is_deficient` 就是 `幾何 < 代數`,
/// struct literal 收尾。
pub fn analyze_eigen_properties(
    epsilon: f64,
    a: &Matrix,
    eigenvalues: &[f64],
) -> Result<Vec<EigenReport>, LinAlgError> {
    // 型別註記必要:下方 `r.value` 的欄位存取先於 push,推斷走不到那裡。
    let mut reports: Vec<EigenReport> = Vec::new();
    for &lambda in eigenvalues {
        if reports.iter().any(|r| (r.value - lambda).abs() <= epsilon) {
            continue; // 已有報告,跳過
        }
        let algebraic = algebraic_multiplicity(epsilon, eigenvalues, lambda);
        let geometric = geometric_multiplicity(epsilon, a, lambda)?;
        let is_deficient = geometric < algebraic;
        reports.push(EigenReport {
            value: lambda,
            algebraic,
            geometric,
            is_deficient,
        });
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // ---- 練習 1:is_eigenpair ----

    /// 筆記題目原例:A·v = 4v —— v 是 λ = 4 的特徵向量。
    #[test]
    fn is_eigenpair_accepts_textbook_example() {
        let a = Matrix::from_rows(vec![
            vec![5.0, 2.0, 1.0],
            vec![-2.0, 1.0, -1.0],
            vec![2.0, 2.0, 4.0],
        ]);
        let v = Vector::from_vec(vec![1.0, -1.0, 1.0]);
        // A·v = [5−2+1, −2−1−1, 2−2+4] = [4, −4, 4] = 4·v
        assert!(is_eigenpair(EPS, &a, &v, 4.0));
    }

    /// 同一個 v、錯的 λ → 不是特徵對。
    #[test]
    fn is_eigenpair_rejects_wrong_lambda() {
        let a = Matrix::from_rows(vec![
            vec![5.0, 2.0, 1.0],
            vec![-2.0, 1.0, -1.0],
            vec![2.0, 2.0, 4.0],
        ]);
        let v = Vector::from_vec(vec![1.0, -1.0, 1.0]);
        assert!(!is_eigenpair(EPS, &a, &v, 3.0), "A·v = 4v ≠ 3v");
    }

    /// 零向量永遠不是特徵向量(定義排除)—— 即使 A·0 = λ·0 對任何 λ 成立。
    #[test]
    fn is_eigenpair_rejects_zero_vector() {
        let a = Matrix::identity(3);
        assert!(!is_eigenpair(EPS, &a, &Vector::new(3), 1.0));
    }

    /// 維度不合(v 不在 domain)→ false,不 panic。
    #[test]
    fn is_eigenpair_rejects_dimension_mismatch() {
        let a = Matrix::identity(3);
        assert!(!is_eigenpair(
            EPS,
            &a,
            &Vector::from_vec(vec![1.0, 0.0]),
            1.0
        ));
    }

    // ---- 練習 3:characteristic_matrix ----

    /// 筆記題目原例:A − 3I —— 只有主對角線被減 3。
    #[test]
    fn characteristic_matrix_subtracts_lambda_on_diagonal() {
        let a = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        let m = characteristic_matrix(&a, 3.0).unwrap();
        assert!(m.equals(&Matrix::from_rows(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, -2.0, 2.0],
            vec![0.0, 2.0, -2.0],
        ])));
    }

    /// 非方陣沒有 A − λI(λI 無從同階)→ NotSquare。
    #[test]
    fn characteristic_matrix_rejects_non_square() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]); // 2×3
        assert_eq!(
            characteristic_matrix(&a, 1.0).unwrap_err(),
            LinAlgError::NotSquare { rows: 2, cols: 3 }
        );
    }

    // ---- 練習 4:eigenspace_basis ----

    /// 筆記題目原例:B 對 λ = 3 的特徵空間是二維(幾何重數 2)。
    /// 不寫死「哪兩個向量」(基底表示不唯一),改驗:個數 = 2、每個都是 λ = 3 的特徵對。
    #[test]
    fn eigenspace_basis_recovers_two_dimensional_eigenspace() {
        let b = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        let basis = eigenspace_basis(EPS, &b, 3.0).unwrap();
        assert_eq!(basis.len(), 2, "E₃ 是二維(λ = 3 幾何重數 2)");
        for v in &basis {
            assert!(
                is_eigenpair(EPS, &b, v, 3.0),
                "基底向量必為 λ = 3 的特徵向量"
            );
        }
    }

    /// λ 不是特徵值 → 特徵空間只有零向量 → 空基底。
    #[test]
    fn eigenspace_basis_is_empty_for_non_eigenvalue() {
        let b = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        assert!(
            eigenspace_basis(EPS, &b, 7.0).unwrap().is_empty(),
            "7 不是 B 的特徵值,E₇ = {{0}}"
        );
    }

    // ---- 練習 5:has_real_eigenvalues_2x2 ----

    /// 90° 旋轉沒有實特徵值(把每個方向都轉開)—— Δ = (0−0)² + 4·(−1)·1 = −4 < 0。
    #[test]
    fn rotation_90_has_no_real_eigenvalues() {
        let rot = Matrix::from_rows(vec![vec![0.0, -1.0], vec![1.0, 0.0]]);
        assert!(!has_real_eigenvalues_2x2(&rot));
    }

    /// 對稱矩陣必有實特徵值;非均勻縮放(對角)也有(Δ ≥ 0)。
    #[test]
    fn symmetric_and_diagonal_have_real_eigenvalues() {
        let symmetric = Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        let scaling = Matrix::from_rows(vec![vec![2.0, 0.0], vec![0.0, 3.0]]);
        assert!(has_real_eigenvalues_2x2(&symmetric));
        assert!(has_real_eigenvalues_2x2(&scaling));
    }

    /// 非 2×2 → false(判別式只對 2×2 成立)。
    #[test]
    fn has_real_eigenvalues_2x2_rejects_non_2x2() {
        assert!(!has_real_eigenvalues_2x2(&Matrix::identity(3)));
    }

    // ═══════════════ 單元 8-2:特徵多項式 1/2 ═══════════════

    // ---- 練習 1:characteristic_matrix 重用(零新碼,stub 階段就綠)----

    /// 8-2 筆記原例:A − 5I。t = 5 不是隨便選的 —— det(A − 5I) = (−9)(1) − (−3)(3) = 0,
    /// 5 正是 A 的特徵值(見 `char_poly_coefficients_factor_textbook_example`:
    /// (t+3)(t−5) 的根之一)。
    #[test]
    fn characteristic_matrix_matches_unit_8_2_example() {
        let a = Matrix::from_rows(vec![vec![-4.0, -3.0], vec![3.0, 6.0]]);
        let m = characteristic_matrix(&a, 5.0).unwrap();
        assert!(m.equals(&Matrix::from_rows(vec![vec![-9.0, -3.0], vec![3.0, 1.0]])));
    }

    // ---- 練習 2:trace(+ 特徵多項式係數的具體對帳)----

    /// 8-2 筆記原例:tr([[−4,−3],[3,6]]) = −4 + 6 = 2。
    #[test]
    fn trace_sums_the_main_diagonal() {
        let a = Matrix::from_rows(vec![vec![-4.0, -3.0], vec![3.0, 6.0]]);
        assert_eq!(a.trace().unwrap(), 2.0);
    }

    /// 非方陣的主對角線走不到底 → NotSquare(沿 characteristic_matrix / power 先例)。
    #[test]
    fn trace_rejects_non_square() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(
            a.trace().unwrap_err(),
            LinAlgError::NotSquare { rows: 2, cols: 3 }
        );
    }

    /// 筆記原例整條串起來:A = [[−4,−3],[3,6]] 的特徵多項式
    /// t² − tr·t + det = t² − 2t − 15 = (t+3)(t−5) —— 兩根 −3、5 都真的是特徵值
    /// (Eλ 非平凡),非根就不是。「解特徵方程」與「零空間非平凡」在具體數字上會合。
    #[test]
    fn char_poly_coefficients_factor_textbook_example() {
        let a = Matrix::from_rows(vec![vec![-4.0, -3.0], vec![3.0, 6.0]]);
        assert_eq!(a.trace().unwrap(), 2.0);
        assert_eq!(a.determinant_recursive().unwrap(), -15.0);
        assert!(
            !eigenspace_basis(EPS, &a, 5.0).unwrap().is_empty(),
            "根 5 是特徵值"
        );
        assert!(
            !eigenspace_basis(EPS, &a, -3.0).unwrap().is_empty(),
            "根 −3 是特徵值"
        );
        assert!(
            eigenspace_basis(EPS, &a, 1.0).unwrap().is_empty(),
            "1 不是根,E₁ = {{0}}"
        );
    }

    // ---- 練習 3:triangular_eigenvalues ----

    /// 8-2 筆記原例:上三角 → 特徵值 = 對角線直讀,重根照留
    /// ([−1, 3, 3]:3 的代數重數 2)。
    #[test]
    fn triangular_eigenvalues_read_the_diagonal_with_repeats() {
        let a = Matrix::from_rows(vec![
            vec![-1.0, 0.0, 0.0],
            vec![0.0, 3.0, 1.0],
            vec![0.0, 0.0, 3.0],
        ]);
        assert_eq!(
            triangular_eigenvalues(EPS, &a).unwrap(),
            vec![-1.0, 3.0, 3.0]
        );
    }

    /// 下三角走 is_lower_triangular 那條路,同樣直讀。
    #[test]
    fn triangular_eigenvalues_accept_lower_triangular() {
        let a = Matrix::from_rows(vec![vec![2.0, 0.0], vec![7.0, -5.0]]);
        assert_eq!(triangular_eigenvalues(EPS, &a).unwrap(), vec![2.0, -5.0]);
    }

    /// 8-2 筆記原例(反例):對稱但非三角 → 直讀不適用,NotTriangular
    /// (它的特徵值是 3, 3, −1,對角線卻是 3, 1, 1 —— 亂讀就錯)。
    #[test]
    fn triangular_eigenvalues_reject_non_triangular() {
        let b = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        assert_eq!(
            triangular_eigenvalues(EPS, &b).unwrap_err(),
            LinAlgError::NotTriangular
        );
    }

    /// 兩層失敗各自精確:非方陣談不上特徵值 → NotSquare(不是 NotTriangular)。
    #[test]
    fn triangular_eigenvalues_reject_non_square() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(
            triangular_eigenvalues(EPS, &a).unwrap_err(),
            LinAlgError::NotSquare { rows: 2, cols: 3 }
        );
    }

    // ---- 練習 5:algebraic_multiplicity ----

    /// 8-2 筆記原例:[−1, 3, 3] 中 3 出現 2 次 —— λ = 3 的代數重數 2。
    #[test]
    fn algebraic_multiplicity_counts_repeated_roots() {
        let eigs = [-1.0, 3.0, 3.0];
        assert_eq!(algebraic_multiplicity(EPS, &eigs, 3.0), 2);
        assert_eq!(algebraic_multiplicity(EPS, &eigs, -1.0), 1);
    }

    /// 不在列表 → 0(「λ 不是特徵值」是合法答案,不是錯誤)。
    #[test]
    fn algebraic_multiplicity_is_zero_for_absent_value() {
        assert_eq!(algebraic_multiplicity(EPS, &[-1.0, 3.0, 3.0], 7.0), 0);
    }

    /// 浮點容差:量值 ≤ epsilon 算同一個根(數值求根的殘差不該拆散重根);
    /// 精確模式(epsilon = 0)下殘差就是不同(沿 is_upper_triangular 的精確模式測試)。
    #[test]
    fn algebraic_multiplicity_merges_roots_within_epsilon() {
        let eigs = [3.0 + 5e-10, 3.0 - 5e-10];
        assert_eq!(algebraic_multiplicity(EPS, &eigs, 3.0), 2);
        assert_eq!(algebraic_multiplicity(0.0, &eigs, 3.0), 0);
    }

    /// 預告下一單元的張力:剪切 [[3,1],[0,3]] 的 λ = 3 **代數重數 2**(對角線兩個 3)、
    /// **幾何重數卻只有 1**(E₃ 一維)—— 兩個重數可以不相等,這正是「可否對角化」的關鍵。
    #[test]
    fn algebraic_multiplicity_can_exceed_geometric() {
        let shear = Matrix::from_rows(vec![vec![3.0, 1.0], vec![0.0, 3.0]]);
        let eigs = triangular_eigenvalues(EPS, &shear).unwrap();
        assert_eq!(algebraic_multiplicity(EPS, &eigs, 3.0), 2, "代數重數");
        assert_eq!(
            eigenspace_basis(EPS, &shear, 3.0).unwrap().len(),
            1,
            "幾何重數"
        );
    }

    // ═══════════════ 單元 8-3:特徵多項式 2/2 ═══════════════

    // ---- 練習 2:geometric_multiplicity ----

    /// 講義矩陣 C(= 8-2 三角原例):λ = 3 幾何重數 1 —— rank(C − 3I) = 2、
    /// nullity = 3 − 2 = 1,對照代數重數 2(對角線兩個 3):**虧損**。λ = −1 則足額。
    #[test]
    fn geometric_multiplicity_detects_deficiency_of_c() {
        let c = Matrix::from_rows(vec![
            vec![-1.0, 0.0, 0.0],
            vec![0.0, 3.0, 1.0],
            vec![0.0, 0.0, 3.0],
        ]);
        assert_eq!(geometric_multiplicity(EPS, &c, 3.0).unwrap(), 1);
        assert_eq!(geometric_multiplicity(EPS, &c, -1.0).unwrap(), 1);
    }

    /// 對照組:講義矩陣 B(8-1 原例)的 λ = 3 幾何重數 2 = 代數重數 —— 足額不虧損。
    #[test]
    fn geometric_multiplicity_of_full_eigenvalue_matches_algebraic() {
        let b = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        assert_eq!(geometric_multiplicity(EPS, &b, 3.0).unwrap(), 2);
    }

    /// λ 不是特徵值 → Eλ = {0} → 幾何重數 0(合法答案,不是錯誤)。
    #[test]
    fn geometric_multiplicity_is_zero_for_non_eigenvalue() {
        let b = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        assert_eq!(geometric_multiplicity(EPS, &b, 7.0).unwrap(), 0);
    }

    /// 非方陣 → NotSquare(characteristic_matrix 那層傳上來)。
    #[test]
    fn geometric_multiplicity_rejects_non_square() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(
            geometric_multiplicity(EPS, &a, 1.0).unwrap_err(),
            LinAlgError::NotSquare { rows: 2, cols: 3 }
        );
    }

    // ---- 練習 4:analyze_eigen_properties ----

    /// 驗收條件:報告必須識別出講義矩陣 C 的特徵值 3 **虧損**(幾何 1 < 代數 2),
    /// −1 足額;重根合併成一份、順序照首次出現。
    #[test]
    fn analyze_identifies_deficient_eigenvalue_of_c() {
        let c = Matrix::from_rows(vec![
            vec![-1.0, 0.0, 0.0],
            vec![0.0, 3.0, 1.0],
            vec![0.0, 0.0, 3.0],
        ]);
        let eigs = triangular_eigenvalues(EPS, &c).unwrap(); // [−1, 3, 3]
        let reports = analyze_eigen_properties(EPS, &c, &eigs).unwrap();
        assert_eq!(
            reports,
            vec![
                EigenReport {
                    value: -1.0,
                    algebraic: 1,
                    geometric: 1,
                    is_deficient: false,
                },
                EigenReport {
                    value: 3.0,
                    algebraic: 2,
                    geometric: 1,
                    is_deficient: true,
                },
            ]
        );
    }

    /// 對照組:矩陣 B 的兩個特徵值都足額(幾何 = 代數)—— 全員不虧損。
    /// B 非三角,特徵值列表 [3, 3, −1] 由講義給(det(B − tI) = (3−t)²(−1−t))。
    #[test]
    fn analyze_reports_no_deficiency_for_b() {
        let b = Matrix::from_rows(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 1.0],
        ]);
        let reports = analyze_eigen_properties(EPS, &b, &[3.0, 3.0, -1.0]).unwrap();
        assert_eq!(
            reports,
            vec![
                EigenReport {
                    value: 3.0,
                    algebraic: 2,
                    geometric: 2,
                    is_deficient: false,
                },
                EigenReport {
                    value: -1.0,
                    algebraic: 1,
                    geometric: 1,
                    is_deficient: false,
                },
            ]
        );
    }

    /// 空列表 → 空報告(沒有根就沒有報告,不是錯誤)。
    #[test]
    fn analyze_of_empty_root_list_is_empty() {
        let b = Matrix::identity(2);
        assert!(analyze_eigen_properties(EPS, &b, &[]).unwrap().is_empty());
    }
}

/// 教材定理的隨機驗證(「for all」形式)。整數策略配精確 / 1e-9 容差(整數加減乘在 f64 下
/// 精確;唯有 `eigenspace_basis` 經 RREF 帶除法,容差吸收捨入殘差)。
#[cfg(test)]
mod laws {
    use super::*;
    use crate::{is_linearly_independent, standard_matrix};
    use proptest::prelude::*;

    const EPS: f64 = 1e-9;

    /// 固定 n×n、元素為小整數的方陣。
    fn int_square_matrix(n: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-5i64..=5, n), n).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            )
        })
    }

    /// 隨機階數(1..=4)的整數方陣。
    fn int_square_any() -> impl Strategy<Value = Matrix> {
        (1usize..=4).prop_flat_map(int_square_matrix)
    }

    /// 對角矩陣 diag(d₀…dₙ₋₁):eᵢ 是已知的特徵向量(特徵值 dᵢ),laws 用來「植入」特徵對。
    fn diagonal_matrix(diag: &[f64]) -> Matrix {
        let n = diag.len();
        Matrix::from_rows(
            (0..n)
                .map(|i| (0..n).map(|j| if i == j { diag[i] } else { 0.0 }).collect())
                .collect(),
        )
    }

    /// 對角線(小整數)連同一個合法索引 `i < n` —— 植入「eᵢ 是 λ = dᵢ 的特徵向量」。
    fn diagonal_with_index() -> impl Strategy<Value = (Vec<f64>, usize)> {
        (1usize..=4).prop_flat_map(|n| {
            (
                prop::collection::vec((-5i64..=5).prop_map(|v| v as f64), n),
                0..n,
            )
        })
    }

    /// 對稱 2×2 矩陣 [[a, b], [b, d]](小整數)。
    fn symmetric_2x2() -> impl Strategy<Value = Matrix> {
        (-9i64..=9, -9i64..=9, -9i64..=9).prop_map(|(a, b, d)| {
            Matrix::from_rows(vec![vec![a as f64, b as f64], vec![b as f64, d as f64]])
        })
    }

    /// 上三角整數方陣(i > j 全填 0)—— 8-2 練習 3「對角線直讀」的原生棲地。
    fn int_upper_triangular() -> impl Strategy<Value = Matrix> {
        (1usize..=4).prop_flat_map(|n| {
            prop::collection::vec(prop::collection::vec(-5i64..=5, n), n).prop_map(|grid| {
                Matrix::from_rows(
                    grid.into_iter()
                        .enumerate()
                        .map(|(i, row)| {
                            row.into_iter()
                                .enumerate()
                                .map(|(j, v)| if i > j { 0.0 } else { v as f64 })
                                .collect()
                        })
                        .collect(),
                )
            })
        })
    }

    /// 同階整數方陣對 (A, P) —— 相似性 law 要兩枚同階(沿 operator.rs 的 paired 策略)。
    fn int_square_pair() -> impl Strategy<Value = (Matrix, Matrix)> {
        (1usize..=3).prop_flat_map(|n| (int_square_matrix(n), int_square_matrix(n)))
    }

    proptest! {
        // 練習 1(植入版):對角 D 的 eᵢ 必是 λ = dᵢ 的特徵向量(D·eᵢ = dᵢ·eᵢ 精確成立)。
        #[test]
        fn diagonal_standard_vectors_are_eigenpairs((diag, i) in diagonal_with_index()) {
            let d = diagonal_matrix(&diag);
            let ei = Vector::standard(diag.len(), i).unwrap();
            prop_assert!(is_eigenpair(EPS, &d, &ei, diag[i]));
        }

        // 練習 1(否定):零向量對**任何** A、任何 λ 都不是特徵對。
        #[test]
        fn zero_vector_is_never_an_eigenvector(a in int_square_any(), lambda in -5i64..=5) {
            let n = a.cols();
            prop_assert!(!is_eigenpair(EPS, &a, &Vector::new(n), lambda as f64));
        }

        // 練習 2(運算子 = 標準矩陣特徵,經 Theorem 2.9 橋接):把對角 D 當運算子 closure,
        // standard_matrix 取樣重建出 A == D,於是「運算子的特徵對」與「其標準矩陣的特徵對」
        // 是同一回事 —— 運算子版檢查不必另立函式。
        #[test]
        fn operator_eigen_agrees_with_standard_matrix((diag, i) in diagonal_with_index()) {
            let d = diagonal_matrix(&diag);
            let a = standard_matrix(diag.len(), |x| d.multiply_vector(x).unwrap()).unwrap();
            prop_assert!(a.approx_equals(&d, EPS), "Thm 2.9:取樣重建運算子的標準矩陣");
            let ei = Vector::standard(diag.len(), i).unwrap();
            prop_assert!(is_eigenpair(EPS, &a, &ei, diag[i]));
        }

        // 練習 3:characteristic_matrix 只動主對角線 —— 對角格減 λ、其餘照搬。
        #[test]
        fn characteristic_matrix_touches_only_diagonal(a in int_square_any(), lambda in -5i64..=5) {
            let n = a.cols();
            let m = characteristic_matrix(&a, lambda as f64).unwrap();
            prop_assert_eq!((m.rows(), m.cols()), (n, n), "形狀不變");
            for i in 0..n {
                for j in 0..n {
                    let expected = a.row(i).unwrap()[j] - if i == j { lambda as f64 } else { 0.0 };
                    prop_assert!((m.row(i).unwrap()[j] - expected).abs() <= EPS);
                }
            }
        }

        // 練習 4(驗收條件):特徵空間基底的每個向量都是該 λ 的特徵向量、且彼此獨立。
        // 用對角 D + λ = dᵢ 植入,保證 Eλ 非平凡。
        #[test]
        fn eigenspace_basis_vectors_are_independent_eigenvectors(
            (diag, i) in diagonal_with_index(),
        ) {
            let d = diagonal_matrix(&diag);
            let lambda = diag[i];
            let basis = eigenspace_basis(EPS, &d, lambda).unwrap();
            prop_assert!(!basis.is_empty(), "λ = dᵢ 是特徵值,Eλ 非平凡");
            prop_assert!(is_linearly_independent(EPS, &basis), "基底必線性獨立");
            for v in &basis {
                prop_assert!(is_eigenpair(EPS, &d, v, lambda), "基底向量必為特徵向量");
            }
        }

        // 練習 4(幾何重數):dim Eλ = nullity(A − λI)。對任意 λ 成立(多半兩邊皆 0)——
        // 把特徵空間維度接回零空間維度,Eλ = Null(A − λI) 的直接後果。
        #[test]
        fn eigenspace_dimension_equals_nullity(a in int_square_any(), lambda in -5i64..=5) {
            const SOLVE_EPS: f64 = 1e-7; // RREF 除法殘差
            let m = characteristic_matrix(&a, lambda as f64).unwrap();
            let basis = eigenspace_basis(SOLVE_EPS, &a, lambda as f64).unwrap();
            prop_assert_eq!(basis.len(), m.nullity(SOLVE_EPS), "dim Eλ ≠ nullity(A − λI)");
        }

        // 練習 5(驗收條件):對稱 2×2 必有實特徵值 —— Δ = (a−d)² + 4b² ≥ 0 恆成立。
        #[test]
        fn symmetric_2x2_always_has_real_eigenvalues(a in symmetric_2x2()) {
            prop_assert!(has_real_eigenvalues_2x2(&a));
        }

        // ═══════════════ 單元 8-2:特徵多項式 1/2 ═══════════════

        // 8-2 練習 2:2×2 特徵多項式展開 det(A − tI) = t² − tr(A)·t + det(A)。
        // 「係數函式」只會重述這條公式 → 不進 API,直接驗展開本身。
        // determinant_recursive(餘因子展開,純多項式無除法)對整數精確 → 兩邊可精確比。
        #[test]
        fn char_poly_2x2_expands_via_trace_and_det(a in int_square_matrix(2), t in -5i64..=5) {
            let t = t as f64;
            let lhs = characteristic_matrix(&a, t)
                .unwrap()
                .determinant_recursive()
                .unwrap();
            let rhs = t * t - a.trace().unwrap() * t + a.determinant_recursive().unwrap();
            prop_assert_eq!(lhs, rhs, "det(A−tI) 應精確展開為 t² − tr·t + det");
        }

        // 8-2 練習 3:三角矩陣特徵值 = 主對角線(照序、含重複);
        // 下三角 = 上三角的轉置,同一條律(轉置不動對角線)。
        #[test]
        fn triangular_eigenvalues_read_off_the_diagonal(a in int_upper_triangular()) {
            let n = a.rows();
            let diag: Vec<f64> = (0..n).map(|i| a.row(i).unwrap()[i]).collect();
            prop_assert_eq!(triangular_eigenvalues(EPS, &a).unwrap(), diag.clone(), "上三角");
            prop_assert_eq!(triangular_eigenvalues(EPS, &a.transpose()).unwrap(), diag, "下三角");
        }

        // 8-2 練習 3(接回 8-1):直讀出的每個 λ 真的是特徵值 —— Eλ 非平凡,基底全是特徵對
        // (A 三角 ⟹ A − λI 三角且對角線見 0 ⟹ det = 0 ⟹ 零空間非平凡)。
        #[test]
        fn triangular_eigenvalues_have_nontrivial_eigenspace(a in int_upper_triangular()) {
            const SOLVE_EPS: f64 = 1e-7; // RREF 除法殘差
            for lambda in triangular_eigenvalues(EPS, &a).unwrap() {
                let basis = eigenspace_basis(SOLVE_EPS, &a, lambda).unwrap();
                prop_assert!(!basis.is_empty(), "對角線元素 {} 必是特徵值", lambda);
                for v in &basis {
                    prop_assert!(is_eigenpair(SOLVE_EPS, &a, v, lambda));
                }
            }
        }

        // 8-2 練習 4:相似矩陣共享特徵多項式 —— B = P⁻¹AP 時 det(A − tI) = det(B − tI)
        // 對所有 t(講義證明:det(P⁻¹(A − tI)P) = det(P⁻¹)·det(A − tI)·det(P))。
        // P⁻¹ 帶除法 → SOLVE/COMPARE 兩檔容差,沿 operator.rs 的相似 laws 慣例。
        #[test]
        fn similar_matrices_share_characteristic_polynomial(
            (a, p) in int_square_pair(), t in -5i64..=5,
        ) {
            const SOLVE_EPS: f64 = 1e-9;
            const COMPARE_EPS: f64 = 1e-6;
            prop_assume!(p.is_invertible(SOLVE_EPS)); // 奇異 P 不見證相似,略過

            let t = t as f64;
            let p_inv = p.inverse(SOLVE_EPS).expect("可逆已由 prop_assume 保證");
            let b = p_inv.multiply(&a).unwrap().multiply(&p).unwrap();

            let lhs = characteristic_matrix(&a, t).unwrap().determinant(SOLVE_EPS).unwrap();
            let rhs = characteristic_matrix(&b, t).unwrap().determinant(SOLVE_EPS).unwrap();
            prop_assert!(
                (lhs - rhs).abs() <= COMPARE_EPS,
                "det(A−tI) = {},det(P⁻¹AP − tI) = {}",
                lhs,
                rhs
            );
        }

        // 8-2 練習 5:重數會計 —— 三角矩陣 n 個對角元全數入列,
        // 相異特徵值的代數重數總和 = n(特徵多項式 n 次,根含重數恰 n 個)。
        #[test]
        fn algebraic_multiplicities_sum_to_dimension(a in int_upper_triangular()) {
            let eigs = triangular_eigenvalues(EPS, &a).unwrap();
            let mut distinct = eigs.clone();
            distinct.sort_by(f64::total_cmp);
            distinct.dedup(); // 整數精確,dedup 不受浮點殘差干擾
            let total: usize = distinct
                .iter()
                .map(|&lambda| algebraic_multiplicity(EPS, &eigs, lambda))
                .sum();
            prop_assert_eq!(total, a.rows());
        }

        // 8-2 練習 6:判別式殊途同歸 —— 係數路 Δ = tr² − 4·det 與 8-1 的
        // (a−d)² + 4bc 是代數上同一個數,判定必一致(整數精確,連 Δ = 0 邊界都準)。
        #[test]
        fn discriminant_via_coefficients_agrees_with_has_real(a in int_square_matrix(2)) {
            let delta = a.trace().unwrap().powi(2) - 4.0 * a.determinant_recursive().unwrap();
            prop_assert_eq!(delta >= 0.0, has_real_eigenvalues_2x2(&a));
        }

        // ═══════════════ 單元 8-3:特徵多項式 2/2 ═══════════════

        // 8-3 練習 2(驗收):rank-nullity 兩路對帳 —— 幾何重數 = n − rank(A − λI)。
        // 「數自由行」(nullity / special solutions)與「數 pivot 行」(rank)互為餘數,
        // 對任意 λ 成立(λ 非特徵值時兩邊皆給 0)。
        #[test]
        fn geometric_multiplicity_is_dimension_minus_rank(
            a in int_square_any(), lambda in -5i64..=5,
        ) {
            const SOLVE_EPS: f64 = 1e-7; // RREF 除法殘差
            let m = characteristic_matrix(&a, lambda as f64).unwrap();
            let geo = geometric_multiplicity(SOLVE_EPS, &a, lambda as f64).unwrap();
            prop_assert_eq!(geo, a.rows() - m.rank(SOLVE_EPS));
        }

        // 8-3 練習 3(Theorem 5.1):真特徵值的幾何重數夾在 1 與代數重數之間 ——
        // 1 ≤ 幾何 ≤ 代數。三角家族:代數重數可直讀(repo 尚無一般求根器,誠實驗在
        // 能精確算兩邊的家族上)。
        #[test]
        fn theorem_5_1_geometric_at_most_algebraic(a in int_upper_triangular()) {
            const SOLVE_EPS: f64 = 1e-7;
            let eigs = triangular_eigenvalues(EPS, &a).unwrap();
            let mut distinct = eigs.clone();
            distinct.sort_by(f64::total_cmp);
            distinct.dedup();
            for lambda in distinct {
                let alg = algebraic_multiplicity(EPS, &eigs, lambda);
                let geo = geometric_multiplicity(SOLVE_EPS, &a, lambda).unwrap();
                prop_assert!(
                    (1..=alg).contains(&geo),
                    "Theorem 5.1 破功:λ = {},幾何 {} ∉ [1, 代數 {}]",
                    lambda,
                    geo,
                    alg
                );
            }
        }

        // 8-3 對照律(預告對角化):對角矩陣**永不虧損** —— 每個特徵值幾何 = 代數
        // (dᵢ 在對角線出現 k 次 ⟺ D − dᵢI 恰有 k 個零對角 ⟺ nullity = k)。
        // 下一單元的「可對角化 ⟺ 每個 λ 都足額」在最簡家族上先看見。
        #[test]
        fn diagonal_matrices_are_never_deficient((diag, _) in diagonal_with_index()) {
            const SOLVE_EPS: f64 = 1e-7;
            let d = diagonal_matrix(&diag);
            let mut distinct = diag.clone();
            distinct.sort_by(f64::total_cmp);
            distinct.dedup();
            for lambda in distinct {
                prop_assert_eq!(
                    geometric_multiplicity(SOLVE_EPS, &d, lambda).unwrap(),
                    algebraic_multiplicity(EPS, &diag, lambda),
                    "對角矩陣的 λ = {} 不該虧損",
                    lambda
                );
            }
        }

        // 8-3 練習 4:報告與獨立計算器逐項對帳 —— 每個相異 λ 恰一份、algebraic /
        // geometric / is_deficient 與 standalone 函式一致、Σ 代數重數 = n(8-2 會計律
        // 經報告路重現)。
        #[test]
        fn eigen_reports_agree_with_standalone_calculators(a in int_upper_triangular()) {
            const SOLVE_EPS: f64 = 1e-7;
            let eigs = triangular_eigenvalues(EPS, &a).unwrap();
            let reports = analyze_eigen_properties(SOLVE_EPS, &a, &eigs).unwrap();

            let mut distinct = eigs.clone();
            distinct.sort_by(f64::total_cmp);
            distinct.dedup();
            prop_assert_eq!(reports.len(), distinct.len(), "每個相異 λ 恰一份報告");

            let mut total_algebraic = 0;
            for r in &reports {
                prop_assert_eq!(r.algebraic, algebraic_multiplicity(SOLVE_EPS, &eigs, r.value));
                prop_assert_eq!(
                    r.geometric,
                    geometric_multiplicity(SOLVE_EPS, &a, r.value).unwrap()
                );
                prop_assert_eq!(r.is_deficient, r.geometric < r.algebraic);
                total_algebraic += r.algebraic;
            }
            prop_assert_eq!(total_algebraic, a.rows(), "Σ 代數重數 = n");
        }
    }
}
