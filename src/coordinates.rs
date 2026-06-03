//! 座標:一個向量在某組**有序基底**下的唯一權重 —— 向量空間理論章的收尾。
//!
//! 對應原始 Go 專案 commit `216d87a`。
//!
//! x 相對於有序基底 B = (b₁,…,b_d) 的**座標** `[x]_B = (c₁,…,c_d)` 是唯一一組權重,使
//!
//! ```text
//! x = c₁b₁ + … + c_db_d.
//! ```
//!
//! 它對**每個** x 都存在且唯一,正因為 B 是一組基底 ——「基底」的兩半,恰好各補上
//! 「座標良好定義」的一半:
//!
//! - **生成(spanning)** ⇒ 至少存在一組權重(x 碰得到 —— **存在性**)
//! - **獨立(independent)** ⇒ 至多一組權重(沒有鬆動 —— **唯一性**)
//!
//! 所以「基底」恰好是那個把 [`Span::combination`] 的三選一答案(無解 / 唯一 / 無限多)
//! 收束成唯一保證的「恰好一組」的前置條件 —— 這才讓我們能談 x 的「**那組**座標」。映射
//! x ↦ [x]_B 於是是個 bijection ℝ^d → ℝ^d:[`coordinates`] 是正向,[`from_coordinates`]
//! 是它的逆。

use crate::{LinAlgError, Solution, Span, Vector, is_basis};

/// 回傳 `[x]_B` —— x 在有序基底 `basis` 下的唯一權重向量。當 `basis` **不是** x 所在空間
/// 的一組基底時回 [`LinAlgError::NotABasis`],因為在那個前置條件之外座標並未定義:不生成的
/// 清單可能根本碰不到 x,相依的清單則讓 x 有多種權重、無正則可選者。
///
/// 前置條件用 `is_basis(epsilon, dim, basis)` 對 **ambient 維度** `dim = x.rows()` 檢查,
/// 而非讀 `combination` 的結局。`is_basis` 同時查兩半 —— 生成(x 碰得到)與獨立(權重唯一)
/// —— 其中**生成**那半是 `combination` 單看不見的:它只在 `span(basis)` 內部推理,於是一個
/// 「獨立但不生成」的清單(如 ℝ² 裡的 {(1,0)})會以它自己較小 span 的唯一解溜過去。對 `dim`
/// 檢查 spanning 才補上這個破口。
pub fn coordinates(epsilon: f64, x: &Vector, basis: &[Vector]) -> Result<Vector, LinAlgError> {
    let dim = x.rows();
    if !is_basis(epsilon, dim, basis) {
        return Err(LinAlgError::NotABasis { dim });
    }

    // 前置條件已成立 ⇒ Combination 保證回 Unique:生成讓 x 碰得到(存在)、獨立讓權重唯一
    // (no slack)。其餘兩支邏輯上不可達 —— Inconsistent 需「不在 span」(與生成矛盾)、
    // Infinite 需「相依」(與獨立矛盾)—— 故以 unreachable! 把這個定理當不變式編進型別。
    match Span::new(epsilon, basis.to_vec()).combination(x) {
        Solution::Unique(coords) => Ok(coords),
        other => unreachable!("已驗證為基底,Combination 必為 Unique,卻得 {other:?}"),
    }
}

/// 從座標向量重建 x:`x = coords₀·b₀ + … + coords_{d-1}·b_{d-1}`。它是 [`coordinates`] 的
/// **逆**,所以對任意 x 與任意基底 B,`from_coordinates(coordinates(x, B), B)` 還原回 x。
///
/// 重建不過就是「基底向量以座標為權重的線性組合」,於是直接委派給
/// [`Vector::linear_combination`] —— 並承襲它的錯誤:座標個數與基底向量數不符、或基底為空時。
pub fn from_coordinates(coords: &Vector, basis: &[Vector]) -> Result<Vector, LinAlgError> {
    Vector::linear_combination(coords.entries(), basis)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RREF 與 `is_basis` 背後的 rank 檢查都會引入捨入,故座標以小容差判定,不做精確比對。
    const COORD_EPS: f64 = 1e-9;

    fn v(data: Vec<f64>) -> Vector {
        Vector::from_vec(data)
    }

    /// 把座標映射釘在手算過的基底上。標準基底必須重現 x 自身的分量;傾斜或重縮放的基底必須
    /// 報出在**那個 frame** 下重建 x 的權重;而任何不是基底的清單都必須被拒,因為沒有基底就
    /// 沒有座標。通過的案例再走一次 round-trip,把逆律 `from_coordinates(coordinates(x)) = x`
    /// 縮小驗一遍。
    #[test]
    fn coordinates_known_cases() {
        struct Case {
            name: &'static str,
            x: Vector,
            basis: Vec<Vector>,
            want: Option<Vector>, // None ⇒ 預期回 NotABasis
        }

        let cases = vec![
            Case {
                // 標準基底 e₀,e₁:座標映射就是 identity —— [x]_B 即 x。
                name: "standard basis is the identity coordinate map",
                x: v(vec![3.0, 5.0]),
                basis: vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0])],
                want: Some(v(vec![3.0, 5.0])),
            },
            Case {
                // 傾斜基底 (1,1),(1,-1):解 c₁+c₂=3、c₁−c₂=5 ⇒ c=(4,−1)。同一點 x,換個
                // frame,座標就不同。
                name: "tilted basis gives different coordinates for the same point",
                x: v(vec![3.0, 5.0]),
                basis: vec![v(vec![1.0, 1.0]), v(vec![1.0, -1.0])],
                want: Some(v(vec![4.0, -1.0])),
            },
            Case {
                // 重縮放的軸 (2,0),(0,5):每個座標是 x 的分量除以軸長 —— (6,5) 變 (3,1)。
                name: "rescaled axes rescale the coordinates",
                x: v(vec![6.0, 5.0]),
                basis: vec![v(vec![2.0, 0.0]), v(vec![0.0, 5.0])],
                want: Some(v(vec![3.0, 1.0])),
            },
            Case {
                // 相依清單 —— span{(1,1),(2,2)} 只是一條線,不是 ℝ² 的基底。(3,3) 雖在線上
                // 卻有多種權重,座標未定義:回錯誤,不是幸運命中。
                name: "dependent list is not a basis",
                x: v(vec![3.0, 3.0]),
                basis: vec![v(vec![1.0, 1.0]), v(vec![2.0, 2.0])],
                want: None,
            },
            Case {
                // 向量太少、生成不了 ℝ²:單一軸不是基底,即使該點剛好落在它上面。前置條件敗在
                // 生成,不是「碰不到」。
                name: "non-spanning list is not a basis",
                x: v(vec![2.0, 0.0]),
                basis: vec![v(vec![1.0, 0.0])],
                want: None,
            },
        ];

        for case in cases {
            let got = coordinates(COORD_EPS, &case.x, &case.basis);
            match &case.want {
                None => {
                    // Vector 刻意不實作 PartialEq(浮點相等須帶容差),故不能對
                    // Result<Vector, _> 用 assert_eq!;只比錯誤側 —— LinAlgError 有 PartialEq。
                    assert_eq!(
                        got.unwrap_err(),
                        LinAlgError::NotABasis { dim: case.x.rows() },
                        "{}: 應拒非基底",
                        case.name
                    );
                }
                Some(want) => {
                    let coords =
                        got.unwrap_or_else(|e| panic!("{}: 對真正的基底不該出錯: {e}", case.name));
                    assert!(
                        coords.approx_equals(want, COORD_EPS),
                        "{}: 座標 = {coords:?},want {want:?}",
                        case.name
                    );

                    // round-trip:把座標餵回基底必須重建 x —— 逆律的縮小版。
                    let rebuilt = from_coordinates(&coords, &case.basis)
                        .unwrap_or_else(|e| panic!("{}: from_coordinates 出錯: {e}", case.name));
                    assert!(
                        rebuilt.approx_equals(&case.x, COORD_EPS),
                        "{}: round-trip 從座標 {coords:?} 重建出 {rebuilt:?},want {:?}",
                        case.name,
                        case.x
                    );
                }
            }
        }
    }

    /// 非基底時委派出去的 `from_coordinates` 仍可獨立呼叫 —— 它不查基底,純粹做線性組合,
    /// 故與 `coordinates` 解耦:座標個數與基底向量數不符時,承襲 `linear_combination` 的
    /// `CountMismatch`。
    #[test]
    fn from_coordinates_inherits_linear_combination_errors() {
        let coords = v(vec![1.0, 2.0, 3.0]); // 3 個座標
        let basis = vec![v(vec![1.0, 0.0]), v(vec![0.0, 1.0])]; // 但只有 2 個向量
        assert_eq!(
            from_coordinates(&coords, &basis).unwrap_err(),
            LinAlgError::CountMismatch
        );
    }
}

#[cfg(test)]
mod laws {
    use super::*;
    use crate::Matrix;
    use proptest::prelude::*;

    /// 產生 `n×n`、元素為 [-10, 10] 整數的方陣(f64 下精確)—— 幾乎必為 full rank,其行
    /// 就構成 ℝ^n 的一組基底。
    fn int_square_matrix(n: usize) -> impl Strategy<Value = Matrix> {
        prop::collection::vec(prop::collection::vec(-10i64..=10, n), n).prop_map(|grid| {
            Matrix::from_rows(
                grid.into_iter()
                    .map(|row| row.into_iter().map(|x| x as f64).collect())
                    .collect(),
            )
        })
    }

    /// 產生長度 `n`、元素為 [-10, 10] 整數的向量。
    fn int_vector(n: usize) -> impl Strategy<Value = Vector> {
        prop::collection::vec(-10i64..=10, n)
            .prop_map(|xs| Vector::from_vec(xs.into_iter().map(|x| x as f64).collect()))
    }

    proptest! {
        /// 把「座標映射是 bijection」轉成隨機試驗律:對基底 B 與任意向量 x,
        /// `from_coordinates(coordinates(x, B), B) = x`。
        ///
        /// 隨機方陣幾乎必 full rank,其行構成 ℝ^d 的基底;測度為零的奇異抽樣由 `is_basis`
        /// 過濾(`prop_assume!` 跳過,對應 Go 的 `continue`)。求解走緊 epsilon 讓整數資料上
        /// 的 pivot 判定俐落,最終比較則用較鬆容差,吸收 RREF 與重建求和累積的捨入。
        #[test]
        fn coordinates_round_trip_is_identity(
            (a, x) in (1usize..=4).prop_flat_map(|dim| {
                (int_square_matrix(dim), int_vector(dim))
            }),
        ) {
            const SOLVE_EPS: f64 = 1e-9; // 整數元素上俐落的 rank/pivot 判定
            const COMPARE_EPS: f64 = 1e-6; // 吸收 round-trip 累積的浮點誤差

            let dim = a.cols();
            let basis: Vec<Vector> = (0..dim).map(|j| a.column(j).unwrap()).collect();

            // 跳過奇異(測度為零)的抽樣 —— 它們的行不是基底。
            prop_assume!(is_basis(SOLVE_EPS, dim, &basis));

            let coords = coordinates(SOLVE_EPS, &x, &basis)
                .expect("對真正的基底,coordinates 不該出錯");
            let rebuilt = from_coordinates(&coords, &basis)
                .expect("from_coordinates 不該出錯");

            prop_assert!(
                rebuilt.approx_equals(&x, COMPARE_EPS),
                "round-trip x → [x]_B → x 把 {:?} 變成 {rebuilt:?}(座標 {coords:?})",
                x
            );
        }
    }
}
