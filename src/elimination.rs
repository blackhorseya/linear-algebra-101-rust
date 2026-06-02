//! Gaussian elimination —— 高斯消去法。
//!
//! 對應原始 Go 專案的 `elimination.go`。目前:forward pass(把矩陣化簡成 REF)。
//!
//! 方法掛在 [`Matrix`] 上(`impl Matrix`),但本模組跨在 `matrix` 模組外、碰不到 private
//! 的 `data` 欄位 —— 一律走 `Matrix` 的 **public API**(`clone` / `row` / `swap_rows` /
//! `add_scaled_row`)。這也順帶驗證:先前刻的公開介面,足以在模組外實作整個消去法。

use crate::Matrix;

impl Matrix {
    /// 私有輔助:**partial pivoting** —— 在 `start_row` 及以下,回 column `col` 量值最大的
    /// 列索引;整段都在 `epsilon` 內(沒有可用 pivot)回 `None`。
    ///
    /// 挑「量值最大」是數值穩定性的關鍵:forward pass 會除以 pivot,小 pivot 會放大消去
    /// 因子與捨入誤差;挑最大的讓每個 factor 量值 ≤ 1。哨兵 `-1` → `Option<usize>`,
    /// 「沒有 pivot」與真實列索引在型別上分開。
    fn pivot_row_below(&self, col: usize, start_row: usize, epsilon: f64) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_mag = epsilon; // 量值要超過 epsilon 才算可用 pivot
        for r in start_row..self.rows() {
            let mag = self.row(r).unwrap()[col].abs(); // r、col 皆合法 → unwrap 安全
            if mag > best_mag {
                best = Some(r);
                best_mag = mag;
            }
        }
        best
    }

    /// 高斯消去法的 **forward pass**:回傳一個化簡成 row echelon form(REF)的**新**矩陣。
    /// 逐 column 用 partial pivoting 選 pivot、換到定位、清掉其**下方**各格。**不**把 pivot
    /// 正規化成 1、也**不**清上方 —— 那是往 RREF 的 backward pass。呼叫端的矩陣不被更動
    /// (`&self`,編譯期保證)。
    ///
    /// `epsilon` 是搜尋 pivot 時「算零」的門檻(傳 `0.0` 即精確)。
    /// (對應原始 Go 專案 commit `98a1ffe`。)
    pub fn row_echelon_form(&self, epsilon: f64) -> Matrix {
        let mut result = self.clone(); // clone 一次,之後全部就地改
        let mut pivot_row = 0; // 下一條要安放 pivot 的列
        for col in 0..result.cols() {
            if pivot_row >= result.rows() {
                break; // 列用完了,剩下的 column 不會再有 pivot
            }
            let Some(p) = result.pivot_row_below(col, pivot_row, epsilon) else {
                continue; // 這 column 沒 pivot → 跳過,pivot 因此可跨 column
            };
            result.swap_rows(pivot_row, p).unwrap(); // 把 pivot 換到定位

            // pivot 值在內層迴圈中不變(只改它下方的列),故提到迴圈外算一次 ——
            // loop-invariant code motion,順手省掉每圈重讀一次 pivot 列。
            let pivot_val = result.row(pivot_row).unwrap()[col];
            for r in (pivot_row + 1)..result.rows() {
                let factor = result.row(r).unwrap()[col] / pivot_val;
                result.add_scaled_row(r, pivot_row, -factor).unwrap(); // r != pivot_row 必成立
            }
            pivot_row += 1; // 只有真的放了 pivot 才前進
        }
        result
    }

    /// 高斯消去法的完整化簡:回傳一個 **reduced row echelon form(RREF)** 的新矩陣。
    /// 先跑 forward pass 到 REF,再 backward pass:由最底的 pivot 往上,把每個 pivot
    /// 正規化成 1、清掉它**上方**的格子。與 REF 不同,一個矩陣的 **RREF 是唯一的**。
    /// 呼叫端的矩陣不被更動。
    ///
    /// `epsilon` 是「算零」的門檻(傳 `0.0` 即精確)。
    /// (對應原始 Go 專案 commit `125d40d`。)
    pub fn reduced_row_echelon_form(&self, epsilon: f64) -> Matrix {
        let mut result = self.row_echelon_form(epsilon); // forward pass → REF(獨立 owned)
        // backward pass:由下而上,每個 pivot 正規化成 1 並清掉其上方。由下而上才能單趟
        // 搞定 —— 用某 pivot 列往上清時,它在「更右邊 pivot column」的位置已是零,不擾動已完成的。
        for row in (0..result.rows()).rev() {
            let Some(pc) = result.pivot_col(row, epsilon) else {
                continue; // 零列沒有 pivot,跳過
            };
            let pivot_val = result.row(row).unwrap()[pc];
            result.scale_row(row, 1.0 / pivot_val).unwrap(); // pivot → 1(|pivot| > ε,不除以零)
            for r in 0..row {
                let factor = result.row(r).unwrap()[pc]; // pivot 已是 1,factor 就是那一格
                result.add_scaled_row(r, row, -factor).unwrap(); // r < row → r != row
            }
        }
        result
    }

    /// 把矩陣化成 REF,回傳 **pivot 行(基本變數)** 的遞增索引;其個數即 [`rank`](Matrix::rank)。
    /// 量值在 `epsilon` 內算零(傳 `0.0` 即精確)。
    /// (對應原始 Go 專案 commit `3b982c5`。)
    pub fn pivot_columns(&self, epsilon: f64) -> Vec<usize> {
        // 化 REF,逐列取 pivot_col(回 Option):None(零列)被丟掉、Some(pc) 留下。
        // REF 的階梯保證收集出來的索引遞增。collect 的型別由回傳型別推斷,免 turbofish。
        let ref_matrix = self.row_echelon_form(epsilon);
        (0..ref_matrix.rows())
            .filter_map(|i| ref_matrix.pivot_col(i, epsilon))
            .collect()
    }

    /// 矩陣的 **rank(秩)**:pivot 數 —— 即 column space 的維度、獨立約束的數量。
    pub fn rank(&self, epsilon: f64) -> usize {
        self.pivot_columns(epsilon).len()
    }

    /// 矩陣的 **nullity(零化度)**:null space 的維度、自由變數的個數。由 rank-nullity
    /// 定理等於 `cols - rank`(`rank ≤ cols`,不會 underflow)。
    pub fn nullity(&self, epsilon: f64) -> usize {
        self.cols() - self.rank(epsilon)
    }

    /// **free 行(自由變數)** 的遞增索引 —— 非 pivot 行。與 [`pivot_columns`](Matrix::pivot_columns)
    /// 一起恰好分割 `[0, cols)`:每個變數非基本(被 pivot 釘住)即自由。其個數即 nullity。
    pub fn free_columns(&self, epsilon: f64) -> Vec<usize> {
        // pivot 的補集:不在 pivot 行裡的每一行都是自由變數。
        // contains 在小矩陣下的 O(rank) 可接受(學習庫不追效能)。
        let pivot_cols = self.pivot_columns(epsilon);
        (0..self.cols())
            .filter(|c| !pivot_cols.contains(c))
            .collect()
    }
}

/// 消去法會除以 pivot,在「應為零」的格子留下捨入殘差;`REF_EPSILON` 把它吸收掉,
/// 讓 pivot 下方的格子仍判讀為零。小整數輸入的殘差遠低於此。
#[cfg(test)]
const REF_EPSILON: f64 = 1e-9;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_echelon_form_reduces_representative_shapes() {
        // 斷言「輸出形態」(在 REF + 維度不變)而非手算值 —— 獨立於 pivot 選擇。
        let cases = vec![
            // 已是 REF
            Matrix::from_rows(vec![
                vec![2.0, 1.0, 3.0],
                vec![0.0, 5.0, 4.0],
                vec![0.0, 0.0, 0.0],
            ]),
            // 第一個 pivot 要靠換列才找得到
            Matrix::from_rows(vec![
                vec![0.0, 2.0, 1.0],
                vec![4.0, 1.0, 0.0],
                vec![2.0, 1.0, 1.0],
            ]),
            // 零 column 讓 pivot 跳過
            Matrix::from_rows(vec![vec![0.0, 1.0], vec![0.0, 2.0]]),
            // rank deficient(第二列是第一列的兩倍)
            Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]]),
            // 全零
            Matrix::new(3, 3),
            // 單列、單行、寬、高
            Matrix::from_rows(vec![vec![0.0, 0.0, 3.0]]),
            Matrix::from_rows(vec![vec![2.0], vec![4.0], vec![6.0]]),
            Matrix::from_rows(vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]),
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]),
        ];
        for m in cases {
            let reduced = m.row_echelon_form(REF_EPSILON);
            assert!(
                reduced.is_row_echelon_form(REF_EPSILON),
                "輸出應在 REF\n m={:?}\n reduced={:?}",
                m,
                reduced
            );
            assert_eq!(
                (reduced.rows(), reduced.cols()),
                (m.rows(), m.cols()),
                "維度應保持"
            );
        }
    }

    #[test]
    fn reduced_row_echelon_form_reduces_representative_shapes() {
        let cases = vec![
            // 已是 RREF
            Matrix::from_rows(vec![
                vec![1.0, 0.0, 3.0],
                vec![0.0, 1.0, 4.0],
                vec![0.0, 0.0, 0.0],
            ]),
            // 需要正規化 pivot + 清上方
            Matrix::from_rows(vec![vec![2.0, 4.0, 2.0], vec![0.0, 3.0, 6.0]]),
            // rank deficient
            Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]]),
            // 零 column 讓 pivot 跳過
            Matrix::from_rows(vec![vec![0.0, 2.0], vec![0.0, 4.0]]),
            // 全零、單列、高
            Matrix::new(3, 3),
            Matrix::from_rows(vec![vec![0.0, 0.0, 3.0]]),
            Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]),
        ];
        for m in cases {
            let rref = m.reduced_row_echelon_form(REF_EPSILON);
            assert!(
                rref.is_reduced_row_echelon_form(REF_EPSILON),
                "輸出應在 RREF\n m={:?}\n rref={:?}",
                m,
                rref
            );
            assert_eq!(
                (rref.rows(), rref.cols()),
                (m.rows(), m.cols()),
                "維度應保持"
            );
        }
    }

    #[test]
    fn reduced_row_echelon_form_of_invertible_is_identity() {
        // 可逆矩陣一路化簡到單位矩陣 —— 這正是「可逆 A 的 Ax=b 有唯一解 x=A⁻¹b」的根據。
        let invertibles = vec![
            Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 1.0]]),
            Matrix::from_rows(vec![
                vec![1.0, 2.0, 3.0],
                vec![0.0, 1.0, 4.0],
                vec![5.0, 6.0, 0.0],
            ]),
        ];
        for m in invertibles {
            let rref = m.reduced_row_echelon_form(REF_EPSILON);
            assert!(
                rref.approx_equals(&Matrix::identity(m.rows()), REF_EPSILON),
                "可逆矩陣的 RREF 應為單位矩陣\n m={:?}\n rref={:?}",
                m,
                rref
            );
        }
    }

    #[test]
    fn rank_nullity_and_columns_characterize_structure() {
        // (matrix, want_rank, want_pivot_columns, want_free_columns)
        let cases: Vec<(Matrix, usize, Vec<usize>, Vec<usize>)> = vec![
            // 單位矩陣:滿秩、無自由變數
            (Matrix::identity(3), 3, vec![0, 1, 2], vec![]),
            // 零矩陣:rank 0、全自由
            (Matrix::new(2, 3), 0, vec![], vec![0, 1, 2]),
            // pivot 之間夾著自由行
            (
                Matrix::from_rows(vec![vec![1.0, 2.0, 0.0, 3.0], vec![0.0, 0.0, 1.0, 4.0]]),
                2,
                vec![0, 2],
                vec![1, 3],
            ),
            // rank deficient(第二列 = 2×第一列)
            (
                Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]),
                1,
                vec![0],
                vec![1],
            ),
            // 單列、兩個自由行
            (
                Matrix::from_rows(vec![vec![1.0, 2.0, 3.0]]),
                1,
                vec![0],
                vec![1, 2],
            ),
            // rank 要化簡才看得出來
            (
                Matrix::from_rows(vec![vec![1.0, 1.0], vec![1.0, 1.0]]),
                1,
                vec![0],
                vec![1],
            ),
        ];
        for (m, want_rank, want_pivots, want_free) in cases {
            assert_eq!(m.rank(REF_EPSILON), want_rank, "rank\n m={m:?}");
            assert_eq!(
                m.nullity(REF_EPSILON),
                m.cols() - want_rank,
                "nullity\n m={m:?}"
            );
            assert_eq!(
                m.pivot_columns(REF_EPSILON),
                want_pivots,
                "pivot_columns\n m={m:?}"
            );
            assert_eq!(
                m.free_columns(REF_EPSILON),
                want_free,
                "free_columns\n m={m:?}"
            );
        }
    }
}

/// 消去法的 property test —— 斷言形態與「保持解集」,而非手算值。
#[cfg(test)]
mod laws {
    use super::*;
    use crate::{Solution, System, Vector};
    use proptest::prelude::*;

    /// 隨機形狀(1..=4 × 1..=4)、實數元素的矩陣。先選形狀再生資料(`prop_flat_map`)。
    fn any_real_matrix() -> impl Strategy<Value = Matrix> {
        (1usize..=4, 1usize..=4).prop_flat_map(|(rows, cols)| {
            prop::collection::vec(prop::collection::vec(-10.0f64..10.0, cols), rows)
                .prop_map(Matrix::from_rows)
        })
    }

    /// 固定 `rows×cols`、小整數元素的矩陣(保持條件數溫和,殘差遠低於 epsilon)。
    fn int_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-5i64..=5, cols), rows).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|v| v as f64).collect())
                    .collect(),
            )
        })
    }

    /// 長度 `n`、小整數元素的向量。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-5i64..=5, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|v| v as f64).collect()))
    }

    /// 一個 ERO 的描述子 —— 用來生成隨機的 row-equivalent 矩陣(給 canonical 測試)。
    /// 參數依建構即合法(scale 的 c 非零、add 的 dst ≠ src),施作時 `unwrap` 安全。
    #[derive(Debug, Clone)]
    enum Ero {
        Swap(usize, usize),
        Scale(usize, f64),
        AddScaled(usize, usize, f64),
    }

    /// 產生作用在 `rows`(須 ≥ 2)列矩陣上的合法 ERO,純量為小的非零整數。
    fn ero(rows: usize) -> impl Strategy<Value = Ero> {
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
        // 招牌性質:無論形狀或秩,forward pass 都落在 REF(且維度不變)。
        #[test]
        fn row_echelon_form_produces_ref(m in any_real_matrix()) {
            let reduced = m.row_echelon_form(REF_EPSILON);
            prop_assert!(
                reduced.is_row_echelon_form(REF_EPSILON),
                "forward pass 未產生 REF\n m={m:?}\n reduced={reduced:?}"
            );
            prop_assert_eq!((reduced.rows(), reduced.cols()), (m.rows(), m.cols()), "維度應保持");
        }

        // 求解該在意的正確性:forward pass 只由 EROs 組成,故把 [A|b] 化成 REF 不改變解集。
        // 植入已知解 x*(b := A·x*),化簡、拆回 S',確認 x* 仍滿足化簡後的系統。
        #[test]
        fn forward_elimination_preserves_solution_set(
            a in int_matrix(3, 3),
            x_star in int_vector(3),
        ) {
            let b = a.multiply_vector(&x_star).unwrap();
            let s = System::new(a.clone(), b).unwrap();

            let reduced = s.to_augmented_matrix().row_echelon_form(REF_EPSILON);
            let s_prime = System::from_augmented_matrix(reduced).unwrap();

            prop_assert!(
                s_prime.is_solution(&x_star, REF_EPSILON).unwrap(),
                "x* 經 forward elimination 後不再是解\n a={a:?}\n x*={x_star:?}"
            );
        }

        // RREF 版的招牌性質:無論形狀或秩,完整 pass 都落在 RREF(且維度不變)。
        #[test]
        fn reduced_row_echelon_form_produces_rref(m in any_real_matrix()) {
            let rref = m.reduced_row_echelon_form(REF_EPSILON);
            prop_assert!(
                rref.is_reduced_row_echelon_form(REF_EPSILON),
                "full pass 未產生 RREF\n m={m:?}\n rref={rref:?}"
            );
            prop_assert_eq!((rref.rows(), rref.cols()), (m.rows(), m.cols()), "維度應保持");
        }

        // backward pass 也只由 EROs 組成:化簡 [A|b] 到 RREF 仍保持解集。
        #[test]
        fn reduced_elimination_preserves_solution_set(
            a in int_matrix(3, 3),
            x_star in int_vector(3),
        ) {
            let b = a.multiply_vector(&x_star).unwrap();
            let s = System::new(a.clone(), b).unwrap();

            let reduced = s.to_augmented_matrix().reduced_row_echelon_form(REF_EPSILON);
            let s_prime = System::from_augmented_matrix(reduced).unwrap();

            prop_assert!(
                s_prime.is_solution(&x_star, REF_EPSILON).unwrap(),
                "x* 經 full elimination 後不再是解\n a={a:?}\n x*={x_star:?}"
            );
        }

        // 整個 arc 的句點:RREF 唯一。兩個 row-equivalent 矩陣(一個是另一個跑隨機 ERO
        // 序列得到)化簡到同一個 RREF —— sampling 等價實驗的精確、確定性版本。
        #[test]
        fn rref_is_canonical(m in int_matrix(3, 3), ops in prop::collection::vec(ero(3), 0..8)) {
            // 兩條路徑捨入殘差略不同,容差比 REF_EPSILON 寬。
            const CANONICAL_EPSILON: f64 = 1e-7;
            let equivalent = apply_eros(m.clone(), &ops); // 與 m row-equivalent
            let from_m = m.reduced_row_echelon_form(REF_EPSILON);
            let from_equivalent = equivalent.reduced_row_echelon_form(REF_EPSILON);
            prop_assert!(
                from_m.approx_equals(&from_equivalent, CANONICAL_EPSILON),
                "row-equivalent 矩陣的 RREF 不同\n m={m:?}\n RREF(m)={from_m:?}\n RREF(equiv)={from_equivalent:?}"
            );
        }

        // rank-nullity 定理:rank + nullity = cols。每個變數非基本即自由,兩數必和為行數。
        #[test]
        fn rank_nullity_theorem(m in any_real_matrix()) {
            prop_assert_eq!(
                m.rank(REF_EPSILON) + m.nullity(REF_EPSILON),
                m.cols(),
                "rank + nullity ≠ cols\n m={:?}", m
            );
        }

        // pivot 行與 free 行恰好分割 [0, cols):每一行剛好被涵蓋一次。
        #[test]
        fn pivot_and_free_columns_partition(m in any_real_matrix()) {
            let mut covered = vec![0u32; m.cols()];
            for c in m.pivot_columns(REF_EPSILON) {
                covered[c] += 1;
            }
            for c in m.free_columns(REF_EPSILON) {
                covered[c] += 1;
            }
            prop_assert!(
                covered.iter().all(|&n| n == 1),
                "每一行應剛好被涵蓋一次\n covered={covered:?}\n m={m:?}"
            );
        }

        // rank 在 ERO 下不變:矩陣與其 row-equivalent 版本 pivot 數相同。
        #[test]
        fn rank_is_invariant_under_eros(
            m in int_matrix(4, 4),
            ops in prop::collection::vec(ero(4), 0..8),
        ) {
            let equivalent = apply_eros(m.clone(), &ops);
            prop_assert_eq!(
                m.rank(REF_EPSILON),
                equivalent.rank(REF_EPSILON),
                "rank 在 ERO 下改變了\n m={:?}", m
            );
        }

        // nullity 與 solve 一致:植入解(系統相容)→ nullity 0 ⟺ Unique、> 0 ⟺ Infinite。
        #[test]
        fn nullity_agrees_with_solve(a in int_matrix(3, 3), x_star in int_vector(3)) {
            const EPS: f64 = 1e-7;
            let nullity = a.nullity(EPS);
            let b = a.multiply_vector(&x_star).unwrap();
            let s = System::new(a, b).unwrap();
            match s.solve(EPS) {
                Solution::Unique(_) => prop_assert_eq!(nullity, 0, "唯一解應對應 nullity 0"),
                Solution::Infinite => prop_assert!(nullity > 0, "無限多解應對應 nullity > 0"),
                Solution::Inconsistent => prop_assert!(false, "植入解的系統不該無解"),
            }
        }
    }
}
