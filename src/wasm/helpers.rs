//! 跨章共用的邊界工具 —— `wasm` 模組樹內部使用(`pub(super)`),不是公開 API。
//!
//! 收的是「不屬於任何一章」的橫切積木(沿 core 把 `error.rs` 獨立成橫切模組的
//! 慣例):快照攤平 `flatten`、寫死的容差 `TRACE_EPSILON`、ERO 的課本式描述
//! `describe_*`,以及僅測試編譯的對帳建構 helper。

use crate::Matrix;

/// 消去法搜尋 pivot 時「算零」的門檻;與 `elimination.rs` 的測試同量級。寫死在 binding
/// 內(沿用 `are_parallel` 把 epsilon 寫死的慣例),呼叫端不必煩惱容差。
pub(super) const TRACE_EPSILON: f64 = 1e-9;

/// 把 `Matrix` 攤平成 row-major `Vec<f64>`(快照用)。
pub(super) fn flatten(m: &Matrix) -> Vec<f64> {
    (0..m.rows())
        .flat_map(|i| m.row(i).unwrap().iter().copied())
        .collect()
}

/// 把純量收成簡潔字串:接近整數就顯示整數,否則最多 4 位、去尾零。給步驟描述用。
fn fmt_scalar(x: f64) -> String {
    let rounded = (x * 10_000.0).round() / 10_000.0;
    if (rounded - rounded.round()).abs() < TRACE_EPSILON {
        format!("{}", rounded.round() as i64) // -0 也歸 0
    } else {
        format!("{rounded:.4}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

/// ERO 描述(1-indexed,貼合數學課本的 R1, R2…):交換兩列。
pub(super) fn describe_swap(i: usize, j: usize) -> String {
    format!("R{} ↔ R{}", i + 1, j + 1)
}

/// ERO 描述:第 `row` 列乘以純量 `c`(pivot 正規化)。
pub(super) fn describe_scale(row: usize, c: f64) -> String {
    format!("R{} ← {}·R{}", row + 1, fmt_scalar(c), row + 1)
}

/// ERO 描述:`R_dst ← R_dst + c·R_src`。把符號併進 ± 讓它讀起來像課本(R2 ← R2 − 2·R1),
/// `|c| == 1` 時省略係數(R2 ← R2 − R1)。
pub(super) fn describe_add_scaled(dst: usize, src: usize, c: f64) -> String {
    let sign = if c < 0.0 { "−" } else { "+" };
    if (c.abs() - 1.0).abs() < TRACE_EPSILON {
        format!("R{} ← R{} {} R{}", dst + 1, dst + 1, sign, src + 1)
    } else {
        format!(
            "R{} ← R{} {} {}·R{}",
            dst + 1,
            dst + 1,
            sign,
            fmt_scalar(c.abs()),
            src + 1
        )
    }
}

// ---- 測試共用(僅 cfg(test) 編譯;multiply / elimination / inverse 的測試對帳用)----

/// 把一步的 flatten 快照 reshape 回 `Matrix`,方便與 core 的結果比對。
#[cfg(test)]
pub(super) fn snapshot_to_matrix(snap: &[f64], cols: usize) -> Matrix {
    Matrix::from_rows(snap.chunks(cols).map(<[f64]>::to_vec).collect())
}

/// 從 flatten 字面值建出輸入 `Matrix`(對照組用)。
#[cfg(test)]
pub(super) fn input_matrix(data: &[f64], cols: usize) -> Matrix {
    Matrix::from_rows(data.chunks(cols).map(<[f64]>::to_vec).collect())
}
