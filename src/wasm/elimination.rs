//! 高斯消去法(Gauss-Jordan)逐步 trace —— 給前端「過程圖解」用。
//!
//! 設計取向(與 core 的關係):core 的 `reduced_row_echelon_form` 只回最終矩陣,把
//! 每一步 ERO 吃掉了;且鐵律是 **core 零改動**。所以這裡在 binding 層用 `Matrix` 的
//! 公開 API 重跑同一套演算法,額外把每一步攔下來記錄 —— 這正是 `elimination.rs` 已
//! 示範過「在 matrix 模組外、只靠 public API 就能實作整個消去法」的延伸。

use super::helpers::{TRACE_EPSILON, describe_add_scaled, describe_scale, describe_swap, flatten};
use crate::Matrix;
use wasm_bindgen::prelude::*;

// 每一步所屬的階段。用 `u8` 而非 String 過邊界 —— 省每步一次字串 clone,前端再映射文字。
const PHASE_INITIAL: u8 = 0; // 原始矩陣(第 0 步,讓「上一步」能回到原貌)
const PHASE_FORWARD: u8 = 1; // forward pass:partial pivoting + 消 pivot 下方
const PHASE_BACKWARD: u8 = 2; // backward pass:正規化 pivot + 消 pivot 上方

/// 一個消去步驟的完整記錄(純 Rust,**不**過邊界 —— 透過 `EliminationTrace` 的 SoA
/// getter 攤平後才跨界)。`snapshot` 是「這一步做完之後」的矩陣 row-major flatten。
struct Step {
    description: String,    // 人類可讀的操作:如 "R2 ← R2 − 2·R1"
    phase: u8,              // PHASE_INITIAL / FORWARD / BACKWARD
    snapshot: Vec<f64>,     // 該步之後的快照,長度 = rows * cols
    pivot_row: i32,         // 當前 pivot 列(-1 = 無,如 initial 步)
    pivot_col: i32,         // 當前 pivot 行(-1 = 無)
    changed_rows: Vec<u32>, // 這一步被改動的列(前端高亮用)
}

/// 一趟完整消去的 trace,過 WASM 邊界的單一物件。
///
/// **過邊界策略:Structure-of-Arrays(SoA)**。不是「每個 step 一個 JS 物件」(那會讓
/// 每個 step 變成帶指標、需 `.free()` 的 wasm wrapper),而是把 N 個 step 的每個欄位各
/// 攤平成一條 typed array,一次搬完、GC 自動管。前端 wrapper 再把這些平行陣列縫回乾淨
/// 的 plain-JS 物件。
#[wasm_bindgen]
pub struct EliminationTrace {
    rows: usize,
    cols: usize,
    aug_col: i32, // -1 = 一般矩陣;>= 0 = [A|b] 常數欄索引(亦即未知數個數)
    rank: usize,
    solution_kind: u8,       // 0=NA, 1=Unique, 2=Infinite, 3=Inconsistent
    pivot_columns: Vec<u32>, // 終態的 pivot(基本變數)行
    free_columns: Vec<u32>,  // 終態的 free(自由變數)行
    steps: Vec<Step>,        // 私有;經下方 getter 攤平後才過界
}

#[wasm_bindgen]
impl EliminationTrace {
    // --- 純量 getter(在 JS 端是 property)---
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[wasm_bindgen(getter)]
    pub fn aug_col(&self) -> i32 {
        self.aug_col
    }
    #[wasm_bindgen(getter)]
    pub fn rank(&self) -> usize {
        self.rank
    }
    #[wasm_bindgen(getter)]
    pub fn solution_kind(&self) -> u8 {
        self.solution_kind
    }
    #[wasm_bindgen(getter)]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    // --- SoA 平行陣列(各回一條 typed array / string[],前端按 index 對齊)---

    /// 每步的人類可讀描述(→ JS `string[]`)。
    pub fn descriptions(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.description.clone()).collect()
    }
    /// 每步所屬階段(→ `Uint8Array`)。
    pub fn phases(&self) -> Vec<u8> {
        self.steps.iter().map(|s| s.phase).collect()
    }
    /// 每步的 pivot 列(-1 = 無;→ `Int32Array`)。
    pub fn pivot_rows(&self) -> Vec<i32> {
        self.steps.iter().map(|s| s.pivot_row).collect()
    }
    /// 每步的 pivot 行(-1 = 無;→ `Int32Array`)。
    pub fn pivot_cols(&self) -> Vec<i32> {
        self.steps.iter().map(|s| s.pivot_col).collect()
    }
    /// 所有步快照串接成一條 `Float64Array`;前端用 `i * rows * cols` 起、`rows * cols`
    /// 長切出第 `i` 步的矩陣再 reshape。
    pub fn snapshots(&self) -> Vec<f64> {
        self.steps
            .iter()
            .flat_map(|s| s.snapshot.iter().copied())
            .collect()
    }
    /// `changed_rows` 是鋸齒陣列(每步長度不一),用 CSR 風格過界:這裡是串接的值表,
    /// 搭配 [`changed_rows_offsets`](Self::changed_rows_offsets) 的前綴和切片。
    pub fn changed_rows_flat(&self) -> Vec<u32> {
        self.steps
            .iter()
            .flat_map(|s| s.changed_rows.iter().copied())
            .collect()
    }
    /// CSR offsets(長度 `step_count + 1`):第 `i` 步的 changed rows 是
    /// `flat[offsets[i] .. offsets[i+1]]`。
    pub fn changed_rows_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(self.steps.len() + 1);
        let mut acc = 0u32;
        offsets.push(0);
        for s in &self.steps {
            acc += s.changed_rows.len() as u32;
            offsets.push(acc);
        }
        offsets
    }
    /// 終態 pivot 行(→ `Uint32Array`)。
    pub fn pivot_columns(&self) -> Vec<u32> {
        self.pivot_columns.clone()
    }
    /// 終態 free 行(→ `Uint32Array`)。
    pub fn free_columns(&self) -> Vec<u32> {
        self.free_columns.clone()
    }
}

/// partial pivoting:在 `start_row` 及以下,回 column `col` 量值最大的列索引;整段都在
/// `epsilon` 內(沒有可用 pivot)回 `None`。
///
/// **逐行鏡像 `elimination.rs::pivot_row_below`**(規則:量值最大、門檻 `> epsilon`)——
/// 因該函式是 `elimination` 模組私有、跨模組無法呼叫,故在此重寫。若 core 的 pivot 選擇
/// 規則變動,需同步此處;下方測試的「黃金迴歸」(trace 終態 == core 的 RREF)會在漂移時報警。
fn pivot_row_below(m: &Matrix, col: usize, start_row: usize, epsilon: f64) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_mag = epsilon;
    for r in start_row..m.rows() {
        let mag = m.row(r).unwrap()[col].abs();
        if mag > best_mag {
            best = Some(r);
            best_mag = mag;
        }
    }
    best
}

/// 從最終 RREF 讀出線性方程組 `[A|b]` 解的型態,回傳 `solution_kind` 編碼:
/// **1 = Unique、2 = Infinite、3 = Inconsistent**(0 = NA 由呼叫端在一般矩陣模式處理)。
///
/// 高斯消去法的精髓:消完之後「怎麼從 RREF 讀出答案」。對應 `system.rs:152-172` 的邏輯,
/// 但直接讀 RREF、不建 `System`,所以**非方陣 A 也安全**(繞開 `System::new` 要求
/// `a.rows() == b.rows()` 的限制 —— 欠定 / 超定系統都能判)。
///
/// 參數:
/// - `rref`:已化簡到 RREF 的增廣矩陣 `[A|b]`(`rows × cols`,其中 `cols == aug_col + 1`)。
/// - `aug_col`:常數欄 b 的索引,亦即**未知數個數** `n`(呼叫端保證 `>= 0` 才會進來)。
/// - `rank`:終態矩陣的 pivot 數;非矛盾時即等於係數矩陣 A 的 rank。
/// - `epsilon`:判零容差。
fn classify_solution(rref: &Matrix, aug_col: i32, rank: usize, epsilon: f64) -> u8 {
    let n = aug_col as usize; // 未知數個數 = 常數欄索引

    // 1. 任一列的 pivot 落在常數欄(第 n 欄)→ 該列是「0 … 0 | 非零」的矛盾式 → 無解。
    //    pivot 是 leading entry,落在常數欄代表 A 那側整列為零、b 側非零。
    for i in 0..rref.rows() {
        if rref.pivot_col(i, epsilon) == Some(n) {
            return 3; // Inconsistent
        }
    }

    // 2/3. 無矛盾列:此時所有 pivot 都落在 A 部分,rank 即係數矩陣的 rank。
    //      rank == n → 每個未知數都被 pivot 釘住,唯一解;rank < n → 有自由變數,無限多解。
    if rank == n { 1 } else { 2 }
}

/// 高斯消去法(Gauss-Jordan)的**逐步 trace**:把矩陣化簡到 RREF,沿途記錄每個 ERO 與
/// 當下的矩陣快照,供前端逐步播放、圖解。
///
/// - `data`:row-major flatten 的元素,長度須為 `rows * cols`(前端保證,故用 reshape 不檢查)。
/// - `aug_col`:`-1` = 一般矩陣;`>= 0` = 增廣矩陣 `[A|b]` 的常數欄索引。它**只影響**解的型態
///   判讀與前端分隔線,**不影響消去法本身**(消去對整個矩陣一視同仁)。
/// - epsilon 內部寫死 `TRACE_EPSILON`。
#[wasm_bindgen]
pub fn eliminate(data: Vec<f64>, rows: usize, cols: usize, aug_col: i32) -> EliminationTrace {
    let grid: Vec<Vec<f64>> = data.chunks(cols).map(<[f64]>::to_vec).collect();
    let mut m = Matrix::from_rows(grid);
    let mut steps: Vec<Step> = Vec::new();

    // 第 0 步:原始矩陣快照。
    steps.push(Step {
        description: "初始矩陣".to_string(),
        phase: PHASE_INITIAL,
        snapshot: flatten(&m),
        pivot_row: -1,
        pivot_col: -1,
        changed_rows: Vec::new(),
    });

    // ---- Forward pass:鏡像 `row_echelon_form` ----
    let mut pivot_row = 0usize;
    for col in 0..cols {
        if pivot_row >= rows {
            break; // 列用完,剩下的 column 不會再有 pivot
        }
        let Some(p) = pivot_row_below(&m, col, pivot_row, TRACE_EPSILON) else {
            continue; // 這 column 沒 pivot → 跳過(pivot 可跨 column)
        };
        if p != pivot_row {
            m.swap_rows(pivot_row, p).unwrap();
            steps.push(Step {
                description: describe_swap(pivot_row, p),
                phase: PHASE_FORWARD,
                snapshot: flatten(&m),
                pivot_row: pivot_row as i32,
                pivot_col: col as i32,
                changed_rows: vec![pivot_row as u32, p as u32],
            });
        }
        let pivot_val = m.row(pivot_row).unwrap()[col];
        for r in (pivot_row + 1)..rows {
            let factor = m.row(r).unwrap()[col] / pivot_val;
            if factor.abs() <= TRACE_EPSILON {
                continue; // 該格已是零:消去無作用,也不污染 trace
            }
            m.add_scaled_row(r, pivot_row, -factor).unwrap();
            steps.push(Step {
                description: describe_add_scaled(r, pivot_row, -factor),
                phase: PHASE_FORWARD,
                snapshot: flatten(&m),
                pivot_row: pivot_row as i32,
                pivot_col: col as i32,
                changed_rows: vec![r as u32],
            });
        }
        pivot_row += 1;
    }

    // ---- Backward pass:鏡像 `reduced_row_echelon_form`(由下而上)----
    for row in (0..rows).rev() {
        let Some(pc) = m.pivot_col(row, TRACE_EPSILON) else {
            continue; // 零列沒有 pivot
        };
        let pivot_val = m.row(row).unwrap()[pc];
        if (pivot_val - 1.0).abs() > TRACE_EPSILON {
            m.scale_row(row, 1.0 / pivot_val).unwrap();
            steps.push(Step {
                description: describe_scale(row, 1.0 / pivot_val),
                phase: PHASE_BACKWARD,
                snapshot: flatten(&m),
                pivot_row: row as i32,
                pivot_col: pc as i32,
                changed_rows: vec![row as u32],
            });
        }
        for r in 0..row {
            let factor = m.row(r).unwrap()[pc];
            if factor.abs() <= TRACE_EPSILON {
                continue;
            }
            m.add_scaled_row(r, row, -factor).unwrap();
            steps.push(Step {
                description: describe_add_scaled(r, row, -factor),
                phase: PHASE_BACKWARD,
                snapshot: flatten(&m),
                pivot_row: row as i32,
                pivot_col: pc as i32,
                changed_rows: vec![r as u32],
            });
        }
    }

    // ---- 終態統計:全走 core 的公開方法,零重算、零漂移 ----
    let pivot_columns: Vec<u32> = m
        .pivot_columns(TRACE_EPSILON)
        .into_iter()
        .map(|c| c as u32)
        .collect();
    let free_columns: Vec<u32> = m
        .free_columns(TRACE_EPSILON)
        .into_iter()
        .map(|c| c as u32)
        .collect();
    let rank = pivot_columns.len();
    let solution_kind = if aug_col >= 0 {
        classify_solution(&m, aug_col, rank, TRACE_EPSILON)
    } else {
        0 // 一般矩陣:無「解的型態」概念
    };

    EliminationTrace {
        rows,
        cols,
        aug_col,
        rank,
        solution_kind,
        pivot_columns,
        free_columns,
        steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::helpers::{input_matrix, snapshot_to_matrix};

    /// **黃金迴歸**:trace 的最後一步快照,必須等於 core 的 `reduced_row_echelon_form`。
    /// 這把「在 binding 重寫 partial pivoting / 重放迴圈」可能的漂移,變成編譯後即可被
    /// `cargo test` 抓到的失敗 —— 是比註解更強的保險。
    #[test]
    fn eliminate_trace_ends_at_core_rref() {
        // (flatten data, rows, cols)
        let cases: Vec<(Vec<f64>, usize, usize)> = vec![
            (vec![2.0, 1.0, 1.0, 1.0], 2, 2), // 可逆 2×2
            (vec![0.0, 2.0, 1.0, 4.0, 1.0, 0.0, 2.0, 1.0, 1.0], 3, 3), // 第一個 pivot 要換列
            (vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], 2, 3), // rank deficient
            (vec![0.0, 0.0, 0.0, 0.0], 2, 2), // 全零
            (vec![0.0, 0.0, 3.0], 1, 3),      // 單列、pivot 跨 column
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2), // 高矩陣
        ];
        for (data, rows, cols) in cases {
            let trace = eliminate(data.clone(), rows, cols, -1);
            let got = snapshot_to_matrix(&trace.steps.last().unwrap().snapshot, cols);
            let want = input_matrix(&data, cols).reduced_row_echelon_form(TRACE_EPSILON);
            assert!(
                got.approx_equals(&want, 1e-7),
                "trace 終態應等於 core 的 RREF\n got={got:?}\n want={want:?}"
            );
        }
    }

    /// forward 階段最後一個快照應落在 REF(列階梯形)。
    #[test]
    fn eliminate_forward_phase_reaches_ref() {
        let data = vec![0.0, 2.0, 1.0, 4.0, 1.0, 0.0, 2.0, 1.0, 1.0];
        let (rows, cols) = (3, 3);
        let trace = eliminate(data, rows, cols, -1);
        let last_forward = trace
            .steps
            .iter()
            .rfind(|s| s.phase == PHASE_FORWARD)
            .expect("應有 forward 步驟");
        let m = snapshot_to_matrix(&last_forward.snapshot, cols);
        assert!(
            m.is_row_echelon_form(TRACE_EPSILON),
            "forward 終態應在 REF\n m={m:?}"
        );
    }

    /// rank / pivot 行 / free 行皆與 core 的公開方法一致。
    #[test]
    fn eliminate_rank_and_columns_match_core() {
        let data = vec![1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 1.0, 4.0];
        let (rows, cols) = (2, 4);
        let trace = eliminate(data.clone(), rows, cols, -1);
        let core = input_matrix(&data, cols);
        assert_eq!(trace.rank, core.rank(TRACE_EPSILON));
        assert_eq!(trace.pivot_columns, vec![0u32, 2]);
        assert_eq!(trace.free_columns, vec![1u32, 3]);
    }

    /// 第 0 步是原始矩陣(initial 快照),讓播放能回到原貌。
    #[test]
    fn eliminate_first_step_is_initial_snapshot() {
        let data = vec![2.0, 1.0, 1.0, 1.0];
        let trace = eliminate(data.clone(), 2, 2, -1);
        let first = &trace.steps[0];
        assert_eq!(first.phase, PHASE_INITIAL);
        assert_eq!(first.snapshot, data);
        assert_eq!(first.pivot_row, -1);
    }

    /// solution kind 三型各一(含一個**非方陣** A 的 Infinite,證明繞開 `System` 的方陣限制)。
    /// 編碼:1=Unique, 2=Infinite, 3=Inconsistent, 0=NA。
    /// ⚠️ 在 `classify_solution` 還是 placeholder(回 0)時,這個測試會失敗 —— 正是 TDD 紅燈,
    ///    等你實作判讀後轉綠。
    #[test]
    fn eliminate_classifies_solution() {
        // 唯一解:x + y = 3, x − y = 1 → x=2, y=1。aug_col = 2(兩個未知數)。
        let unique = eliminate(vec![1.0, 1.0, 3.0, 1.0, -1.0, 1.0], 2, 3, 2);
        assert_eq!(unique.solution_kind, 1, "應為 Unique");

        // 無限多解(非方陣、欠定):x + y + z = 1,一條方程式三個未知數。aug_col = 3。
        let infinite = eliminate(vec![1.0, 1.0, 1.0, 1.0], 1, 4, 3);
        assert_eq!(infinite.solution_kind, 2, "應為 Infinite");

        // 無解:x + y = 1, x + y = 2 → 矛盾列。aug_col = 2。
        let inconsistent = eliminate(vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0], 2, 3, 2);
        assert_eq!(inconsistent.solution_kind, 3, "應為 Inconsistent");

        // 一般矩陣(aug_col = -1)→ NA。
        let na = eliminate(vec![1.0, 0.0, 0.0, 1.0], 2, 2, -1);
        assert_eq!(na.solution_kind, 0, "一般矩陣應為 NA");
    }
}
