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
}

/// 消去法的 property test —— 斷言形態與「保持解集」,而非手算值。
#[cfg(test)]
mod laws {
    use super::*;
    use crate::{System, Vector};
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
    }
}
