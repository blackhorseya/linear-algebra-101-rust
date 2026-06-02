//! System —— 線性方程組 `Ax = b` 的表示與操作。
//!
//! 對應原始 Go 專案 commit `46b6b36`
//! (`feat(system): add System type with augmented matrix conversion`)。

use crate::{LinAlgError, Matrix, Vector};
use std::fmt;

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

    /// 把增廣矩陣 `[A | b]` 拆回方程組 —— [`to_augmented_matrix`](System::to_augmented_matrix)
    /// 的逆:最後一行剝成常數向量 `b`,其餘各行組回係數矩陣 `A`。
    ///
    /// 依 from-constructor 慣例**取走** `aug` 的所有權(對齊 [`Matrix::from_rows`] /
    /// [`Vector::from_vec`])。注意「取走」在這裡不省記憶體:本函式只能走 `Matrix` 的
    /// public API(`column` / `row`),兩者都回拷貝 —— 所有權的意義在語意,不在效率。
    /// Go 版需手動 defensive copy 並測試結果不與來源 aliasing;Rust 由模組封裝在結構上
    /// 保證,該測試無從失敗,故不移植。
    ///
    /// `aug.cols() == 0` 時沒有最後一行可剝成常數 → `Err(LinAlgError::EmptyMatrix)`。
    /// (對應原始 Go 專案 commit `a49be2e`。)
    pub fn from_augmented_matrix(aug: Matrix) -> Result<System, LinAlgError> {
        if aug.cols() == 0 {
            return Err(LinAlgError::EmptyMatrix);
        }
        let n = aug.cols() - 1;
        let b = aug.column(n)?; // 最後一行就是常數向量 b
        let a_rows = (0..aug.rows())
            .map(|i| aug.row(i).map(|row| row[..n].to_vec())) // 各列去掉最後一格 → A 的列
            .collect::<Result<Vec<_>, _>>()?;
        System::new(Matrix::from_rows(a_rows), b)
    }

    /// 驗證候選向量是否為本系統的解,即 `A·candidate` 是否等於常數向量 `b`。
    ///
    /// 元素是 `f64`,故用容差 `epsilon` 比較而非精確相等:傳 `0.0` 要求精確,
    /// 傳如 `1e-9` 吸收矩陣-向量乘積累積的捨入誤差。
    ///
    /// 候選長度錯誤時回 `Err(LinAlgError::DimensionMismatch)` —— 長度 ≠ 未知數
    /// 個數(A 的行數)不是「答錯」,而是「問題本身 ill-formed」,無從判定,
    /// 故回錯誤而非靜默的 `false`(由 `multiply_vector` 把關、用 `?` 上拋)。
    pub fn is_solution(&self, candidate: &Vector, epsilon: f64) -> Result<bool, LinAlgError> {
        let ax = self.A.multiply_vector(candidate)?; // 長度不符 → ? 自動上拋 Err
        Ok(ax.approx_equals(&self.b, epsilon))
    }

    /// 依第 `i` 條方程式在增廣矩陣 `[A | b]` 裡那一列的形態,判定它的 [`RowKind`]。
    ///
    /// 係數或常數是否算「零」在容差 `epsilon` 內判斷(傳 `0.0` 即精確)。這是
    /// **preliminary(初步)** 檢查:只照當下的列字面讀,看不到要列化簡後才浮現的
    /// 矛盾或冗餘 —— 例如 `[1 1 | 2]` 與 `[1 1 | 3]` 各自 [`RowKind::Normal`],合起來
    /// 卻矛盾(見測試 `has_contradictory_row_misses_hidden_contradiction`)。
    ///
    /// `i` 越界(`>= 方程式數`)→ `Err(LinAlgError::IndexOutOfRange)`,由 `A.row(i)?`
    /// 代為把關(不必另寫索引檢查)。負索引在 `usize` 下不可表示,故無需 runtime 檢查。
    /// (對應原始 Go 專案 commit `c0a294a`。)
    pub fn classify_row(&self, i: usize, epsilon: f64) -> Result<RowKind, LinAlgError> {
        let coeffs = self.A.row(i)?; // 越界自動上拋 IndexOutOfRange
        // 係數有任一非零 → 真約束;否則係數全零,由常數決定是 0 = 0 還是 0 = c。
        let kind = if coeffs.iter().any(|&c| c.abs() > epsilon) {
            RowKind::Normal
        } else if self.b.entries()[i].abs() <= epsilon {
            RowKind::Redundant
        } else {
            RowKind::Contradictory
        };
        Ok(kind)
    }

    /// 是否**有任何**方程式是矛盾的(`0 = c`,c ≠ 0)。`true` 代表系統**確定無解**;
    /// `false` 什麼都不能斷定 —— 同 [`classify_row`](System::classify_row) 的初步限制,
    /// 隱藏的矛盾可能要列化簡後才現形。
    pub fn has_contradictory_row(&self, epsilon: f64) -> bool {
        // i 必在 [0, rows) 內,故 classify_row 不會 Err;用 matches! 免 unwrap。
        (0..self.A.rows())
            .any(|i| matches!(self.classify_row(i, epsilon), Ok(RowKind::Contradictory)))
    }

    /// 系統 `Ax = b` 是否**至少有一個解**(相容)。依一致性定理(Theorem 1.5)的條件 (d):
    /// `rank(A) == rank([A | b])` —— 增廣 b 至多讓 rank 多 1,故兩者相等 ⟺ b 沒帶進新的
    /// pivot ⟺ b 落在 A 的 column space 裡。
    ///
    /// 這是 [`has_contradictory_row`](System::has_contradictory_row) 的**完整版**:它會
    /// 化簡系統,故能抓到要化簡後才現形的隱藏矛盾(`x+y=2, x+y=3`),逐列檢查抓不到。
    /// (對應原始 Go 專案 commit `7a4739e`。)
    pub fn is_consistent(&self, epsilon: f64) -> bool {
        // Theorem 1.5 條件 (d):b 沒帶進新 pivot ⟺ rank 不變 ⟺ 系統有解。
        self.A.rank(epsilon) == self.to_augmented_matrix().rank(epsilon)
    }

    /// 求解線性系統:把增廣矩陣 `[A | b]` 化成 RREF,再讀出三種結局之一([`Solution`])。
    /// 真正的工都在化簡;`solve` 只是解讀那座階梯。量值在 `epsilon` 內算零(傳 `0.0` 即精確)。
    ///
    /// 這一步終於能抓到 [`classify_row`](System::classify_row) 抓不到的**隱藏矛盾**:
    /// `x+y=2, x+y=3` 化簡後冒出 `[0 0 | 1]` → [`Solution::Inconsistent`]。
    /// (對應原始 Go 專案 commit `8839879`。)
    pub fn solve(&self, epsilon: f64) -> Solution {
        let n = self.A.cols(); // 未知數個數;增廣矩陣最後一行(index n)是常數行
        let rref = self.to_augmented_matrix().reduced_row_echelon_form(epsilon);
        let mut coords = vec![0.0; n]; // 解座標,自由變數留 0
        let mut rank = 0;
        for i in 0..rref.rows() {
            let Some(pc) = rref.pivot_col(i, epsilon) else {
                continue; // 零列不帶約束
            };
            if pc == n {
                return Solution::Inconsistent; // pivot 落在常數行 → 0 = 1,矛盾
            }
            coords[pc] = rref.row(i).unwrap()[n]; // 此 pivot 把第 pc 個未知數釘成那個常數
            rank += 1;
        }
        if rank < n {
            Solution::Infinite // 還有自由變數
        } else {
            Solution::Unique(Vector::from_vec(coords))
        }
    }
}

/// 一條方程式(增廣矩陣的一列)的形態分類 —— [`System::classify_row`] 的回傳。
///
/// Rust 的 enum 是封閉和:`RowKind` 只可能是這三者之一,Go 那種 `RowKind(99)` 未定義值
/// 無法表示,故不需要(也無法寫)fallback 分支。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// 至少有一個非零係數 —— 一條真正的約束。
    Normal,
    /// `0 = 0`:係數與常數全為零,方程式恆成立、不帶資訊(冗餘)。
    Redundant,
    /// `0 = c`(c ≠ 0):係數全零但常數非零,方程式永不成立 —— 整個系統因此無解。
    Contradictory,
}

impl fmt::Display for RowKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display 是人類可讀字串(對照 Go 的 `String()`);括號內標出對應的列形態,
        // 比 Debug 的純 variant 名多帶一層數學意義。
        match self {
            RowKind::Normal => write!(f, "normal"),
            RowKind::Redundant => write!(f, "redundant (0 = 0)"),
            RowKind::Contradictory => write!(f, "contradictory (0 = c)"),
        }
    }
}

/// 求解線性系統的三種結局 —— [`System::solve`] 的回傳。
///
/// 解向量只長在 `Unique` 那一支:它**存在 ⟺ 系統有唯一解**,由型別保證(對比 Go 用
/// 「種類 + 可能為 nil 的 vector 欄位」,得靠不變式約束、還要測 nil)。呼叫端被 `match`
/// 逼著面對三種結局,而解向量只有在 `Unique(v)` 的 arm 裡才拿得到。
#[derive(Debug)]
pub enum Solution {
    /// 無解:系統矛盾(化簡後出現 `0 = c`,c ≠ 0 的列)。
    Inconsistent,
    /// 唯一解:恰好一個向量滿足整個系統,解就在此。
    Unique(Vector),
    /// 無限多解:系統相容,但有自由變數(pivot 數 < 未知數個數)。
    Infinite,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_matching_dimensions() {
        // 方陣 2×2:兩式、兩未知數
        let sys = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
            Vector::from_vec(vec![5.0, 6.0]),
        )
        .expect("A 列數與 b 長度相符應成立");
        assert!(
            sys.A
                .equals(&Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]))
        );
        assert!(sys.b.equals(&Vector::from_vec(vec![5.0, 6.0])));

        // 長方(超定):3 式、2 未知數 —— 未知數個數不必等於方程式個數,
        // 只要 A 的列數 == b 長度
        let over = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]),
            Vector::from_vec(vec![7.0, 8.0, 9.0]),
        );
        assert!(over.is_ok(), "超定系統(3 列、b 長 3)應成立");
    }

    #[test]
    fn new_rejects_dimension_mismatch() {
        // 2 式,但 b 有 3 格:第三個常數沒有方程式可歸屬
        let sys = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
            Vector::from_vec(vec![5.0, 6.0, 7.0]),
        );
        assert_eq!(sys.unwrap_err(), LinAlgError::DimensionMismatch);
    }

    #[test]
    fn to_augmented_matrix_appends_constants_column() {
        // 方陣:尾端多一行常數 → 形狀 rows×(cols+1)
        let sys = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
            Vector::from_vec(vec![5.0, 6.0]),
        )
        .unwrap();
        let aug = sys.to_augmented_matrix();
        assert_eq!(
            (aug.rows(), aug.cols()),
            (2, 3),
            "增廣矩陣形狀應為 rows×(cols+1)"
        );
        assert!(aug.equals(&Matrix::from_rows(vec![
            vec![1.0, 2.0, 5.0],
            vec![3.0, 4.0, 6.0],
        ])));

        // 長方:保留多出來的列
        let rect = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]),
            Vector::from_vec(vec![7.0, 8.0, 9.0]),
        )
        .unwrap();
        assert!(rect.to_augmented_matrix().equals(&Matrix::from_rows(vec![
            vec![1.0, 2.0, 7.0],
            vec![3.0, 4.0, 8.0],
            vec![5.0, 6.0, 9.0],
        ])));

        // 單一方程式 2x + 3y = 4 → 一列增廣矩陣 [2 3 | 4]
        let single = System::new(
            Matrix::from_rows(vec![vec![2.0, 3.0]]),
            Vector::from_vec(vec![4.0]),
        )
        .unwrap();
        assert!(
            single
                .to_augmented_matrix()
                .equals(&Matrix::from_rows(vec![vec![2.0, 3.0, 4.0]]))
        );
    }

    #[test]
    fn from_augmented_matrix_splits_off_last_column() {
        // 方陣:[1 2 | 5; 3 4 | 6] → A = [1 2; 3 4]、b = [5, 6]
        let sys = System::from_augmented_matrix(Matrix::from_rows(vec![
            vec![1.0, 2.0, 5.0],
            vec![3.0, 4.0, 6.0],
        ]))
        .expect("有 column 的增廣矩陣應能拆解");
        assert!(
            sys.A
                .equals(&Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]))
        );
        assert!(sys.b.equals(&Vector::from_vec(vec![5.0, 6.0])));

        // 單一方程式 2x + 3y = 4:[2 3 | 4] → A = [2 3]、b = [4]
        let single =
            System::from_augmented_matrix(Matrix::from_rows(vec![vec![2.0, 3.0, 4.0]])).unwrap();
        assert!(single.A.equals(&Matrix::from_rows(vec![vec![2.0, 3.0]])));
        assert!(single.b.equals(&Vector::from_vec(vec![4.0])));
    }

    #[test]
    fn from_augmented_matrix_rejects_no_columns() {
        // 0 行的矩陣沒有最後一行可剝成常數 → EmptyMatrix
        // (Go 還測「結果為 nil」,Rust 由 Result 型別保證錯誤時不存在 System,無需測)
        assert_eq!(
            System::from_augmented_matrix(Matrix::new(2, 0)).unwrap_err(),
            LinAlgError::EmptyMatrix
        );
    }

    /// `from_augmented_matrix` 是 `to_augmented_matrix` 的逆:建出 `[A | b]` 再拆回,
    /// 應原封不動拿回 A 與 b。兩者是一對逆函數。
    #[test]
    fn augmented_matrix_round_trip() {
        // 長方(3 式 2 未知數),確保不只方陣成立
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let b = Vector::from_vec(vec![7.0, 8.0, 9.0]);
        let sys = System::new(a.clone(), b.clone()).unwrap();

        let recovered =
            System::from_augmented_matrix(sys.to_augmented_matrix()).expect("round-trip 不應失敗");
        assert!(recovered.A.equals(&a), "round-trip 後的 A 應與原本相同");
        assert!(recovered.b.equals(&b), "round-trip 後的 b 應與原本相同");
    }

    #[test]
    fn is_solution_verifies_candidate_against_ax_eq_b() {
        // x + y = 3, x - y = 1  ⇒  唯一解 (2, 1)
        let sys = System::new(
            Matrix::from_rows(vec![vec![1.0, 1.0], vec![1.0, -1.0]]),
            Vector::from_vec(vec![3.0, 1.0]),
        )
        .unwrap();

        // 精確解
        assert!(
            sys.is_solution(&Vector::from_vec(vec![2.0, 1.0]), 0.0)
                .unwrap()
        );
        // 錯的候選:A·[1,1] = [2, 0] ≠ [3, 1]
        assert!(
            !sys.is_solution(&Vector::from_vec(vec![1.0, 1.0]), 0.0)
                .unwrap()
        );
        // 近似解:[2, 1+1e-12] 在 1e-9 容差內算解;精確檢查(eps=0)則否 —— 這正是
        // is_solution 收容差的理由
        let near = Vector::from_vec(vec![2.0, 1.0 + 1e-12]);
        assert!(sys.is_solution(&near, 1e-9).unwrap());
        assert!(!sys.is_solution(&near, 0.0).unwrap());
    }

    #[test]
    fn is_solution_accepts_consistent_over_determined_system() {
        // 超定:3 式、2 未知數,(1, 1) 同時滿足 x=1、y=1、x+y=2。
        // 「驗證」只是一次 O(mn) 乘積;「求解」這種系統卻要消去 / 最小平方。
        let sys = System::new(
            Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]),
            Vector::from_vec(vec![1.0, 1.0, 2.0]),
        )
        .unwrap();
        assert!(
            sys.is_solution(&Vector::from_vec(vec![1.0, 1.0]), 0.0)
                .unwrap()
        );
    }

    #[test]
    fn is_solution_rejects_candidate_length_mismatch() {
        // 2 未知數,但候選長度 3:A·candidate 無法成形 → 問題 ill-formed → Err
        let sys = System::new(
            Matrix::from_rows(vec![vec![1.0, 1.0], vec![1.0, -1.0]]),
            Vector::from_vec(vec![3.0, 1.0]),
        )
        .unwrap();
        assert_eq!(
            sys.is_solution(&Vector::from_vec(vec![1.0, 2.0, 3.0]), 0.0)
                .unwrap_err(),
            LinAlgError::DimensionMismatch
        );
    }

    /// 建單列系統的 helper:`[coeffs | constant]` 一條方程式,方便逐列分類測試。
    fn one_row(coeffs: Vec<f64>, constant: f64) -> System {
        System::new(
            Matrix::from_rows(vec![coeffs]),
            Vector::from_vec(vec![constant]),
        )
        .unwrap()
    }

    #[test]
    fn classify_row_labels_each_kind() {
        // 有非零係數 → 真正的約束(Normal)
        assert_eq!(
            one_row(vec![1.0, 2.0], 3.0).classify_row(0, 0.0).unwrap(),
            RowKind::Normal
        );
        // 係數與常數全零 → 0 = 0(Redundant)
        assert_eq!(
            one_row(vec![0.0, 0.0], 0.0).classify_row(0, 0.0).unwrap(),
            RowKind::Redundant
        );
        // 係數全零、常數非零 → 0 = c(Contradictory)
        assert_eq!(
            one_row(vec![0.0, 0.0], 5.0).classify_row(0, 0.0).unwrap(),
            RowKind::Contradictory
        );
        // 看的是大小不是正負:負係數仍是真約束、負常數一樣矛盾
        assert_eq!(
            one_row(vec![-1.0, 0.0], 0.0).classify_row(0, 0.0).unwrap(),
            RowKind::Normal
        );
        assert_eq!(
            one_row(vec![0.0, 0.0], -5.0).classify_row(0, 0.0).unwrap(),
            RowKind::Contradictory
        );
    }

    #[test]
    fn classify_row_judges_zero_within_epsilon() {
        // 係數側:1e-12 在 1e-9 容差內算零 → Redundant;精確檢查(eps=0)則算非零 → Normal
        let tiny_coeff = one_row(vec![1e-12, 0.0], 0.0);
        assert_eq!(
            tiny_coeff.classify_row(0, 1e-9).unwrap(),
            RowKind::Redundant
        );
        assert_eq!(tiny_coeff.classify_row(0, 0.0).unwrap(), RowKind::Normal);

        // 常數側:係數全零、常數 1e-12。1e-9 容差內常數算零 → Redundant;
        // 精確檢查下常數非零 → Contradictory
        let tiny_const = one_row(vec![0.0, 0.0], 1e-12);
        assert_eq!(
            tiny_const.classify_row(0, 1e-9).unwrap(),
            RowKind::Redundant
        );
        assert_eq!(
            tiny_const.classify_row(0, 0.0).unwrap(),
            RowKind::Contradictory
        );
    }

    #[test]
    fn classify_row_rejects_out_of_range() {
        // 1 條方程式,問第 5 列 → 越界(由 A.row(i)? 把關,回 IndexOutOfRange)
        // (Go 還測「負索引」,Rust 的 usize 在編譯期就排除,無需 runtime 測)
        assert_eq!(
            one_row(vec![1.0, 2.0], 3.0)
                .classify_row(5, 0.0)
                .unwrap_err(),
            LinAlgError::IndexOutOfRange { index: 5, len: 1 }
        );
    }

    #[test]
    fn has_contradictory_row_detects_explicit_contradiction() {
        // 第 1 列 [0 0 | 5] 是明寫的 0 = 5 → 偵測到矛盾
        let with_contradiction = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![0.0, 0.0]]),
            Vector::from_vec(vec![3.0, 5.0]),
        )
        .unwrap();
        assert!(with_contradiction.has_contradictory_row(0.0));

        // 全 Normal 的系統:沒有矛盾列
        let all_normal = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
            Vector::from_vec(vec![5.0, 6.0]),
        )
        .unwrap();
        assert!(!all_normal.has_contradictory_row(0.0));

        // 第 1 列 [0 0 | 0] 是冗餘(Redundant)而非矛盾 → false
        let redundant = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![0.0, 0.0]]),
            Vector::from_vec(vec![3.0, 0.0]),
        )
        .unwrap();
        assert!(!redundant.has_contradictory_row(0.0));
    }

    /// 為何這個偵測只是 **preliminary** 的見證:`x+y=2, x+y=3` 顯然無解(x+y 不能同時
    /// 是 2 和 3),但兩列都不是 `[0…0 | c]` 形態,逐列檢查回 `false`。矛盾要列化簡後
    /// (`R1 − R0 = [0 0 | 1]`)才現形 —— 完整偵測得等消去法。
    #[test]
    fn has_contradictory_row_misses_hidden_contradiction() {
        let hidden = System::new(
            Matrix::from_rows(vec![vec![1.0, 1.0], vec![1.0, 1.0]]),
            Vector::from_vec(vec![2.0, 3.0]),
        )
        .unwrap();
        assert!(
            !hidden.has_contradictory_row(0.0),
            "矛盾在列化簡前是隱藏的,初步偵測此處應回 false"
        );
    }

    #[test]
    fn row_kind_display_spells_out_each_kind() {
        // Display 是公開契約,完整比對鎖定字串(同 LinAlgError 的 Display 測試精神)
        assert_eq!(RowKind::Normal.to_string(), "normal");
        assert_eq!(RowKind::Redundant.to_string(), "redundant (0 = 0)");
        assert_eq!(RowKind::Contradictory.to_string(), "contradictory (0 = c)");
    }

    /// 解整數系統時的容差:消去法的除法會引入捨入,讀解時用容差比較。
    const SOLVE_EPSILON: f64 = 1e-9;

    #[test]
    fn solve_finds_unique_solution() {
        // 2x + y = 5, x + y = 3 ⇒ (2, 1)
        let s = System::new(
            Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 1.0]]),
            Vector::from_vec(vec![5.0, 3.0]),
        )
        .unwrap();
        let Solution::Unique(x) = s.solve(SOLVE_EPSILON) else {
            panic!("方陣可逆系統應有唯一解");
        };
        assert!(x.approx_equals(&Vector::from_vec(vec![2.0, 1.0]), SOLVE_EPSILON));

        // 3×3 可逆:A·[1,1,1]
        let s = System::new(
            Matrix::from_rows(vec![
                vec![1.0, 2.0, 3.0],
                vec![0.0, 1.0, 4.0],
                vec![5.0, 6.0, 0.0],
            ]),
            Vector::from_vec(vec![6.0, 5.0, 11.0]),
        )
        .unwrap();
        let Solution::Unique(x) = s.solve(SOLVE_EPSILON) else {
            panic!("3×3 可逆系統應有唯一解");
        };
        assert!(x.approx_equals(&Vector::from_vec(vec![1.0, 1.0, 1.0]), SOLVE_EPSILON));

        // 超定但相容:3 式 2 未知數,仍唯一
        let s = System::new(
            Matrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]),
            Vector::from_vec(vec![2.0, 3.0, 5.0]),
        )
        .unwrap();
        let Solution::Unique(x) = s.solve(SOLVE_EPSILON) else {
            panic!("相容的超定系統應有唯一解");
        };
        assert!(x.approx_equals(&Vector::from_vec(vec![2.0, 3.0]), SOLVE_EPSILON));
    }

    #[test]
    fn solve_reports_inconsistent() {
        // 隱藏矛盾:x+y=2, x+y=3 —— classify_row 抓不到,化簡後冒出 [0 0 | 1]
        let hidden = System::new(
            Matrix::from_rows(vec![vec![1.0, 1.0], vec![1.0, 1.0]]),
            Vector::from_vec(vec![2.0, 3.0]),
        )
        .unwrap();
        assert!(matches!(
            hidden.solve(SOLVE_EPSILON),
            Solution::Inconsistent
        ));

        // 係數成比例、常數不相容
        let proportional = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]),
            Vector::from_vec(vec![1.0, 5.0]),
        )
        .unwrap();
        assert!(matches!(
            proportional.solve(SOLVE_EPSILON),
            Solution::Inconsistent
        ));
    }

    #[test]
    fn solve_reports_infinite() {
        // 第二式 = 2×第一式、常數也相容 → 一條方程式、兩未知數
        let redundant = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]),
            Vector::from_vec(vec![3.0, 6.0]),
        )
        .unwrap();
        assert!(matches!(redundant.solve(SOLVE_EPSILON), Solution::Infinite));

        // 欠定:一式兩未知數 → 一整條線的解
        let under = System::new(
            Matrix::from_rows(vec![vec![1.0, 1.0]]),
            Vector::from_vec(vec![2.0]),
        )
        .unwrap();
        assert!(matches!(under.solve(SOLVE_EPSILON), Solution::Infinite));
    }

    #[test]
    fn is_consistent_classifies_systems() {
        // 唯一解 → 相容
        let unique = System::new(
            Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 1.0]]),
            Vector::from_vec(vec![5.0, 3.0]),
        )
        .unwrap();
        assert!(unique.is_consistent(SOLVE_EPSILON));

        // 無限多解 → 仍相容
        let infinite = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]),
            Vector::from_vec(vec![3.0, 6.0]),
        )
        .unwrap();
        assert!(infinite.is_consistent(SOLVE_EPSILON));

        // 隱藏矛盾 x+y=2, x+y=3:rank A = 1 但 rank[A|b] = 2 → 不相容
        let hidden = System::new(
            Matrix::from_rows(vec![vec![1.0, 1.0], vec![1.0, 1.0]]),
            Vector::from_vec(vec![2.0, 3.0]),
        )
        .unwrap();
        assert!(!hidden.is_consistent(SOLVE_EPSILON));

        // 係數成比例、常數不相容 → 不相容
        let proportional = System::new(
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]),
            Vector::from_vec(vec![1.0, 5.0]),
        )
        .unwrap();
        assert!(!proportional.is_consistent(SOLVE_EPSILON));
    }

    /// preliminary → complete 的收尾:`x+y=2, x+y=3` 騙得過逐列的 `has_contradictory_row`
    /// (沒有任何一列字面是 `[0…0 | c]`),但 `is_consistent` 會化簡系統、正確判定無解。
    #[test]
    fn is_consistent_catches_hidden_contradiction() {
        let hidden = System::new(
            Matrix::from_rows(vec![vec![1.0, 1.0], vec![1.0, 1.0]]),
            Vector::from_vec(vec![2.0, 3.0]),
        )
        .unwrap();
        assert!(
            !hidden.has_contradictory_row(0.0),
            "逐列檢查應被隱藏矛盾騙過"
        );
        assert!(
            !hidden.is_consistent(SOLVE_EPSILON),
            "完整判準應抓到隱藏矛盾"
        );
    }
}

/// 等價律的 property test —— 驗證高斯消去法的定理基礎:對增廣矩陣 `[A | b]` 施作
/// 任意基本列運算(ERO)序列,拆回的方程組與原系統**同解**。這是 `matrix.rs` 裡
/// ERO 可逆律在 System 層的後果(一次 ERO = 左乘一個可逆 elementary matrix E,
/// `Ax = b` 變成等價的 `EAx = Eb`)。
///
/// **陷阱(為何不能隨機取 x 比 `is_solution`):** 解集是 Rⁿ 中零測度的仿射子空間,
/// 隨機實數 x 幾乎必非解 —— 兩個系統都回 `false`,測試對「根本不等價」的系統也會通過。
/// 對策是**植入已知解**:取 A 與 x*,令 `b := A·x*`,x* 依建構即為 S 的解。經隨機 ERO
/// 序列得 S' 後,驗 x* 仍是解、且 S 與 S' 對每個探針給出相同判定。
///
/// ERO 純量取小的非零整數,讓 A'、b' 全程保持整數,植入的整數解精確留存,可用
/// `epsilon = 0` 精確比較(同 `matrix.rs` laws 的整數策略)。
#[cfg(test)]
mod laws {
    use super::*;
    use proptest::prelude::*;

    /// 產生 `rows×cols`、元素為 [-10, 10] 整數的矩陣(f64 下精確)。
    fn int_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-10i64..=10, cols), rows).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            )
        })
    }

    /// 產生長度 `n`、元素為 [-10, 10] 整數的向量(f64 下精確)。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()))
    }

    /// 產生隨機形狀(1..=5 × 1..=5)的整數矩陣 A 與長度 = A 列數的整數向量 b。b 與 A
    /// **獨立**(非植入解),故相容/不相容兩類系統都會出現,適合驗一致性判準。
    fn system_parts() -> impl Strategy<Value = (Matrix, Vector)> {
        (1usize..=5, 1usize..=5).prop_flat_map(|(rows, cols)| {
            let a = prop::collection::vec(prop::collection::vec(-5i64..=5, cols), rows).prop_map(
                |grid| {
                    Matrix::from_rows(
                        grid.into_iter()
                            .map(|row| row.into_iter().map(|v| v as f64).collect())
                            .collect(),
                    )
                },
            );
            let b = prop::collection::vec(-5i64..=5, rows)
                .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()));
            (a, b)
        })
    }

    /// 一個基本列運算的描述子 —— 把「做哪個 ERO、參數多少」當資料生成,再於測試裡施作。
    /// 參數**依建構即滿足各運算的不變式**(scale 的 c 非零、add 的 dst ≠ src),
    /// 故施作時 `unwrap` 必不 panic。
    #[derive(Debug, Clone)]
    enum Ero {
        Swap(usize, usize),
        Scale(usize, f64),
        AddScaled(usize, usize, f64),
    }

    /// 產生作用在 `rows`(須 ≥ 2)列矩陣上的合法 ERO。
    fn ero(rows: usize) -> impl Strategy<Value = Ero> {
        // 非零整數純量(避開 ScaleByZero):從 {-3..=-1} ∪ {1..=3} 取。
        let nonzero = prop_oneof![-3i64..=-1, 1i64..=3].prop_map(|c| c as f64);
        prop_oneof![
            (0..rows, 0..rows).prop_map(|(i, j)| Ero::Swap(i, j)),
            (0..rows, nonzero.clone()).prop_map(|(i, c)| Ero::Scale(i, c)),
            // dst ≠ src 依建構成立:src = (dst + step) mod rows,step ∈ [1, rows)
            (0..rows, 1..rows, nonzero).prop_map(move |(dst, step, c)| Ero::AddScaled(
                dst,
                (dst + step) % rows,
                c
            )),
        ]
    }

    /// 把一串 ERO 依序原地施作在 `m` 上。所有參數合法 → `unwrap` 安全。
    fn apply_eros(mut m: Matrix, ops: &[Ero]) -> Matrix {
        for op in ops {
            match *op {
                Ero::Swap(i, j) => m.swap_rows(i, j).unwrap(),
                Ero::Scale(i, c) => m.scale_row(i, c).unwrap(),
                Ero::AddScaled(dst, src, c) => m.add_scaled_row(dst, src, c).unwrap(),
            }
        }
        m
    }

    proptest! {
        // 對 [A | b] 做任意 ERO 序列 → 解集不變(高斯消去法的定理基礎)。
        #[test]
        fn row_ops_preserve_solution_set(
            a in int_matrix(3, 3),
            x_star in int_vector(3),
            ops in prop::collection::vec(ero(3), 0..8),
            extra_probes in prop::collection::vec(int_vector(3), 0..5),
        ) {
            // 植入:b := A·x*,使 x* 依建構即為 S 的解。
            let b = a.multiply_vector(&x_star).unwrap();
            let s = System::new(a, b).unwrap();
            prop_assert!(s.is_solution(&x_star, 0.0).unwrap(), "植入的 x* 應為 S 的解");

            // 對 [A | b] 跑隨機 ERO 序列,再拆回 S'。
            let aug = apply_eros(s.to_augmented_matrix(), &ops);
            let s_prime = System::from_augmented_matrix(aug).unwrap();

            // (1) 正向:植入的解必須在變換後存活。
            prop_assert!(
                s_prime.is_solution(&x_star, 0.0).unwrap(),
                "x* 應在 ERO 序列後仍是 S' 的解"
            );

            // (2) 一致:S 與 S' 同解集 → 對每個探針給出相同判定。探針 = x*(兩邊 true)
            //     + 隨機整數向量(幾乎必為兩邊 false;偶爾命中解也仍須兩邊一致)。
            let probes = std::iter::once(x_star.clone()).chain(extra_probes);
            for y in probes {
                let in_s = s.is_solution(&y, 0.0).unwrap();
                let in_s_prime = s_prime.is_solution(&y, 0.0).unwrap();
                prop_assert_eq!(in_s, in_s_prime, "S 與 S' 對探針判定不一致 → ERO 改變了解集");
            }
        }

        // 交叉驗證 solve:植入已知解 x*(b := A·x*),系統必相容 → solve 不該回 Inconsistent。
        // 回 Unique 時那個向量必須 == x*、且通過獨立的 is_solution —— 一條路徑靠化 RREF、
        // 另一條靠算 A·x,兩條無關路徑得出同答案是兩者皆正確的強證據。
        #[test]
        fn solve_agrees_with_is_solution_on_planted_systems(
            a in int_matrix(3, 3),
            x_star in int_vector(3),
        ) {
            const EPS: f64 = 1e-7; // 化簡引入捨入,用容差
            let b = a.multiply_vector(&x_star).unwrap();
            let s = System::new(a, b).unwrap();
            match s.solve(EPS) {
                Solution::Inconsistent => {
                    prop_assert!(false, "植入解的系統不該無解\n x*={x_star:?}");
                }
                Solution::Unique(x) => {
                    prop_assert!(
                        x.approx_equals(&x_star, EPS),
                        "唯一解應為植入的 x*\n got={x:?}\n x*={x_star:?}"
                    );
                    prop_assert!(s.is_solution(&x, EPS).unwrap(), "solve 的答案應通過 is_solution");
                }
                // 奇異 A:x* 只是無限多解之一,無從進一步比對
                Solution::Infinite => {}
            }
        }

        // Theorem 1.5 化為可執行斷言:跨隨機系統,三個判準 —— (a) solve 找得到解、
        // (c) [A|b] 的 RREF 常數行無 pivot、(d) rank(A) == rank([A|b]) —— 必須給出同一個
        // 判定,且 is_consistent 與三者皆一致。隨機 b 在各種形狀(含高瘦 A)下會混出
        // 相容與不相容兩類系統。
        #[test]
        fn consistency_theorem_conditions_agree((a, b) in system_parts()) {
            const EPS: f64 = 1e-9;
            let cols = a.cols();
            let s = System::new(a.clone(), b.clone()).unwrap();
            let aug = s.to_augmented_matrix();

            let cond_a = !matches!(s.solve(EPS), Solution::Inconsistent); // (a)
            let rref = aug.reduced_row_echelon_form(EPS); // (c)
            let cond_c = (0..rref.rows()).all(|i| rref.pivot_col(i, EPS) != Some(cols));
            let cond_d = a.rank(EPS) == aug.rank(EPS); // (d)
            let got = s.is_consistent(EPS);

            prop_assert!(
                got == cond_a && got == cond_c && got == cond_d,
                "一致性判準不一致: is_consistent={got} (a)={cond_a} (c)={cond_c} (d)={cond_d}\n a={a:?}\n b={b:?}"
            );
        }

        // (c)⟺(d) 背後的結構事實:接一行至多讓 rank 多 1,故 rank[A|b] 只可能是 rank A
        // (b 沒帶進 pivot → 相容)或 rank A + 1(b 帶進 pivot → 不相容),別無其他。
        #[test]
        fn augmented_rank_is_at_most_one_more((a, b) in system_parts()) {
            const EPS: f64 = 1e-9;
            let rank_a = a.rank(EPS);
            let rank_aug = System::new(a.clone(), b)
                .unwrap()
                .to_augmented_matrix()
                .rank(EPS);
            prop_assert!(
                rank_aug == rank_a || rank_aug == rank_a + 1,
                "rank[A|b]={rank_aug},應為 rank A({rank_a})或 +1\n a={a:?}"
            );
        }
    }
}
