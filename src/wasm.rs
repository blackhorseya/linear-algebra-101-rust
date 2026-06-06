//! WASM binding —— 把 core 的純數學接到瀏覽器,薄 adapter,core 零改動。
//!
//! 整個模組鎖在 `#[cfg(feature = "wasm")]` 後面(見 `lib.rs` 的 gated `pub mod wasm`),
//! 不開 feature 時等於不存在:`cargo test` / `task check` 看不到它,也不把
//! `wasm-bindgen` 拉進依賴樹。
//!
//! 原則:**計算只在 Rust 一份**。JS 只負責 Canvas 繪圖與滑鼠事件,每個變換後的點
//! 都是 core 的 [`multiply_vector`](crate::Matrix::multiply_vector) 算的、每個平行
//! 判定都是 [`is_parallel`](crate::Vector::is_parallel) 算的 —— JS 不重寫任何線代。

use crate::{
    Matrix, Solution, System, Transformation, Vector, is_linearly_independent, standard_matrix,
    verify_linearity,
};
use wasm_bindgen::prelude::*;

/// 2×2 變換矩陣 A 作用在點 `(x, y)` 上,回傳變換後的 `[x', y']`。
///
/// 這是「矩陣作為 2D 線性變換」的核心:row-major 傳 4 個純量(`a b` / `c d`)——
/// `f64` 過邊界零 marshalling,且 2×2·2D 維度固定,比傳陣列更不易出錯。回傳的
/// `Vec<f64>`(長度 2)在 JS 端是 `Float64Array`。
#[wasm_bindgen]
pub fn transform_point(a: f64, b: f64, c: f64, d: f64, x: f64, y: f64) -> Vec<f64> {
    // 1. Matrix::from_rows 組出 2×2:row0 = [a, b],row1 = [c, d]
    // 2. Vector::from_vec 組出輸入點向量 (x, y)
    // 3. 呼叫 core 的 multiply_vector(&v) 算 A·v —— 計算的單一真相就在這一行
    // 4. 維度恆 2×2·2,multiply_vector 不可能回 DimensionMismatch,故用 .expect
    //    把「不會發生」寫成自證的不變式;再 .entries().to_vec() 轉成 Vec<f64> 回傳
    let matrix = Matrix::from_rows(vec![vec![a, b], vec![c, d]]);
    let point = Vector::from_vec(vec![x, y]);
    matrix
        .multiply_vector(&point)
        .expect("2×2·2D 不會維度不匹配")
        .entries()
        .to_vec()
}

/// 兩個 2D 向量是否**平行**(共線 = 線性相依)。直接委派 core 的
/// [`Vector::is_parallel`]。
///
/// `epsilon` 寫死 `1e-9`(與 crate 內測試同量級):視覺化的拖曳座標數量級穩定,
/// 不需要把容差開放到 JS 端 —— binding 替呼叫端決定一個合理的預設。
#[wasm_bindgen]
pub fn are_parallel(ux: f64, uy: f64, wx: f64, wy: f64) -> bool {
    let u = Vector::from_vec(vec![ux, uy]);
    let w = Vector::from_vec(vec![wx, wy]);
    u.is_parallel(&w, 1e-9)
}

/// 2×2 矩陣 `[[a, b], [c, d]]` 的**行列式** `ad − bc`。
///
/// 幾何意義:單位正方形經此變換後的平行四邊形之**有號面積**。`|det|` 是面積縮放倍率,
/// 正負號代表是否翻面(定向),`det == 0` 表示平面被壓扁成一條線(不可逆 / 線性相依)。
///
/// core 目前未提供 determinant,且視覺化軌道鐵律是「不為前端改 core」,故 2×2 的封閉式
/// 直接在 binding 計算 —— 與 `are_parallel` 把運算收在 binding 同屬「薄運算」,計算仍只在
/// Rust 一份,JS 不重寫。
#[wasm_bindgen]
pub fn determinant(a: f64, b: f64, c: f64, d: f64) -> f64 {
    a * d - b * c
}

// ===========================================================================
// 矩陣乘法 row × col 展開 —— 給前端「點 C 的任一格,看 A 第 i 列 · B 第 j 欄
// 的 dot product 攤開」圖解用。
//
// 設計取向(與 core 的關係):core 的 `multiply` 只回最終的 C,把每格的展開項
// a_ik·b_kj 吃掉了;且鐵律是 **core 零改動**。所以 C 本身仍由 core 計算(單一
// 真相),展開項在 binding 層補算(沿 `determinant` 的「薄運算」慣例,計算仍只
// 在 Rust 一份),並以下方測試對帳 Σₖ terms == c_ij,漂移會被 `cargo test` 抓到。
// ===========================================================================

/// `A(a_rows×a_cols)` 能否右乘 `B(b_rows×b_cols)`(內維相等)。
///
/// 維度規則的單一真相在 core:用兩個零矩陣把尺寸帶進 [`Matrix::can_multiply`]
/// 問答案,binding 不重寫 `a_cols == b_rows` 這條判斷。
#[wasm_bindgen]
pub fn can_multiply(a_rows: usize, a_cols: usize, b_rows: usize, b_cols: usize) -> bool {
    Matrix::new(a_rows, a_cols).can_multiply(&Matrix::new(b_rows, b_cols))
}

/// 一次矩陣乘法的結果與逐格展開,過 WASM 邊界的單一物件。
///
/// 過邊界策略同 [`EliminationTrace`]:SoA,每個欄位一條 typed array,前端 wrapper
/// 縫回 plain-JS 物件後 `free()`。
#[wasm_bindgen]
pub struct MultiplyExpansion {
    rows: usize,     // C 的列數 m(= A 的列數)
    cols: usize,     // C 的欄數 p(= B 的欄數)
    inner: usize,    // 內維 n(= A 的欄數 = B 的列數)
    c: Vec<f64>,     // C = A·B,row-major flatten(m×p),由 core 的 multiply 計算
    terms: Vec<f64>, // 展開項:terms[(i·p + j)·n + k] = a_ik·b_kj(m×p×n)
}

#[wasm_bindgen]
impl MultiplyExpansion {
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
    pub fn inner(&self) -> usize {
        self.inner
    }

    // --- SoA 陣列 getter(各跨界一次)---
    pub fn c(&self) -> Vec<f64> {
        self.c.clone()
    }
    pub fn terms(&self) -> Vec<f64> {
        self.terms.clone()
    }
}

/// 矩陣乘法 `C = A·B` 與每格的 row × col 展開項。
///
/// - `a_data` / `b_data`:row-major flatten 的元素,長度須為 `rows * cols`
///   (前端保證,故用 reshape 不檢查;沿 `eliminate` 慣例)。
/// - 維度相容性由前端先以 [`can_multiply`] 確認後才呼叫 —— 此處的 `expect`
///   把「不會發生」寫成自證的不變式(沿 `transform_point` 慣例)。
#[wasm_bindgen]
pub fn multiply_expand(
    a_data: Vec<f64>,
    a_rows: usize,
    a_cols: usize,
    b_data: Vec<f64>,
    b_rows: usize,
    b_cols: usize,
) -> MultiplyExpansion {
    // b_rows 只用來自證前置條件(內維相等);reshape 本身由 chunks 依欄數完成。
    debug_assert_eq!(a_cols, b_rows, "前端先以 can_multiply 檢查過維度");
    let a = Matrix::from_rows(a_data.chunks(a_cols).map(<[f64]>::to_vec).collect());
    let b = Matrix::from_rows(b_data.chunks(b_cols).map(<[f64]>::to_vec).collect());

    // C 的單一真相:core 的 multiply。
    let c = a.multiply(&b).expect("前端先以 can_multiply 檢查過維度");

    // 展開項 a_ik·b_kj:把每格 c_ij 的 dot product 攤開,給前端帶實際數字顯示。
    let mut terms = Vec::with_capacity(a_rows * b_cols * a_cols);
    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                terms.push(a.row(i).unwrap()[k] * b.row(k).unwrap()[j]);
            }
        }
    }

    MultiplyExpansion {
        rows: a_rows,
        cols: b_cols,
        inner: a_cols,
        c: flatten(&c),
        terms,
    }
}

// ===========================================================================
// 高斯消去法(Gauss-Jordan)逐步 trace —— 給前端「過程圖解」用。
//
// 設計取向(與 core 的關係):core 的 `reduced_row_echelon_form` 只回最終矩陣,把
// 每一步 ERO 吃掉了;且鐵律是 **core 零改動**。所以這裡在 binding 層用 `Matrix` 的
// 公開 API 重跑同一套演算法,額外把每一步攔下來記錄 —— 這正是 `elimination.rs` 已
// 示範過「在 matrix 模組外、只靠 public API 就能實作整個消去法」的延伸。
// ===========================================================================

/// 消去法搜尋 pivot 時「算零」的門檻;與 `elimination.rs` 的測試同量級。寫死在 binding
/// 內(沿用 `are_parallel` 把 epsilon 寫死的慣例),呼叫端不必煩惱容差。
const TRACE_EPSILON: f64 = 1e-9;

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

/// 把 `Matrix` 攤平成 row-major `Vec<f64>`(快照用)。
fn flatten(m: &Matrix) -> Vec<f64> {
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
fn describe_swap(i: usize, j: usize) -> String {
    format!("R{} ↔ R{}", i + 1, j + 1)
}

/// ERO 描述:第 `row` 列乘以純量 `c`(pivot 正規化)。
fn describe_scale(row: usize, c: f64) -> String {
    format!("R{} ← {}·R{}", row + 1, fmt_scalar(c), row + 1)
}

/// ERO 描述:`R_dst ← R_dst + c·R_src`。把符號併進 ± 讓它讀起來像課本(R2 ← R2 − 2·R1),
/// `|c| == 1` 時省略係數(R2 ← R2 − R1)。
fn describe_add_scaled(dst: usize, src: usize, c: f64) -> String {
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

// ===========================================================================
// 反矩陣(Gauss-Jordan 累乘基本矩陣)逐步 trace —— 「可逆矩陣與基本矩陣」圖解用。
//
// 與 core 的關係:`Matrix::inverse` 只回最終的 P = Eₖ⋯E₁,把每一步的 Eₖ 與中間
// 累積吃掉了;鐵律又是 **core 零改動**。所以沿 `eliminate` 的前例,在 binding 層
// 重跑同一套演算法、沿途把每一步攔下來記錄。pivot 搜尋直接用 pub(crate) 的
// [`Matrix::pivot_row_below`](與 core `inverse` 同一顆積木),不再鏡像重寫。
//
// 演算法是 `inverse()` 的**一般化**:同樣單趟 Gauss-Jordan,但 pivot row 改用
// 獨立游標(不釘對角線),某 column 找不到 pivot 時跳過該 column 繼續(而非 bail)。
// - 可逆時:每個 column 都有 pivot、游標恆等於 column,行為與 `inverse()` 一致,
//   終態 P 即 A⁻¹(黃金迴歸 `invert_trace_invertible_ends_at_core_inverse`)。
// - 奇異時:做完整趟得 working = RREF(A) 且 P·A = RREF(A) —— Theorem 2.3
//   (PA = R,P 為基本矩陣之乘積)的完整展示,「RREF 含零列 → 不可逆」自然浮現。
// ===========================================================================

/// 每一步施作的 ERO 種類。用 `u8` 過邊界(沿 `PHASE_*` 慣例),前端再映射文字
/// 與幾何名稱(swap = 鏡射、scale = 伸縮、add_scaled = 剪切)。
const ERO_INITIAL: u8 = 0; // 第 0 步:尚未施作(working = A、P = E = Iₙ)
const ERO_SWAP: u8 = 1;
const ERO_SCALE: u8 = 2;
const ERO_ADD_SCALED: u8 = 3;

/// 一個求逆步驟的完整記錄(純 Rust,不過邊界 —— 經 [`InverseTrace`] 的 SoA getter
/// 攤平後才跨界)。三條 n² 快照都是「這一步做完之後」的 row-major flatten。
struct InverseStep {
    description: String,    // 人類可讀的操作:如 "R2 ← R2 − 2·R1"
    ero: u8,                // ERO_INITIAL / SWAP / SCALE / ADD_SCALED
    pivot_row: i32,         // 當前 pivot 列(-1 = 無,如 initial 步)
    pivot_col: i32,         // 當前 pivot 行(-1 = 無)
    changed_rows: Vec<u32>, // 這一步被改動的列(前端高亮用)
    working: Vec<f64>,      // 消去中的矩陣(從 A 出發,終態 = RREF(A))
    p: Vec<f64>,            // 累積 P = Eₖ⋯E₁(從 Iₙ 出發,可逆時終態 = A⁻¹)
    e: Vec<f64>,            // 這一步施作的基本矩陣 Eₖ(initial 步 = Iₙ)
}

/// 一趟完整求逆的 trace,過 WASM 邊界的單一物件(SoA 策略同 [`EliminationTrace`])。
///
/// 終態旗標(invertible / rank / nullity / cols_independent)**各自獨立呼叫 core
/// 計算**、不互相推導 —— IMT 面板要「誠實地分別驗」每個等價條件,讓使用者看到
/// 它們一起翻轉是定理使然(laws 已隨機互驗),不是前端共用同一個布林的把戲。
#[wasm_bindgen]
pub struct InverseTrace {
    n: usize,
    invertible: bool,
    rank: usize,
    nullity: usize,
    cols_independent: bool,
    steps: Vec<InverseStep>, // 私有;經下方 getter 攤平後才過界
}

#[wasm_bindgen]
impl InverseTrace {
    // --- 純量 getter(在 JS 端是 property)---
    #[wasm_bindgen(getter)]
    pub fn n(&self) -> usize {
        self.n
    }
    #[wasm_bindgen(getter)]
    pub fn invertible(&self) -> bool {
        self.invertible
    }
    #[wasm_bindgen(getter)]
    pub fn rank(&self) -> usize {
        self.rank
    }
    #[wasm_bindgen(getter)]
    pub fn nullity(&self) -> usize {
        self.nullity
    }
    #[wasm_bindgen(getter)]
    pub fn cols_independent(&self) -> bool {
        self.cols_independent
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
    /// 每步的 ERO 種類(→ `Uint8Array`)。
    pub fn eros(&self) -> Vec<u8> {
        self.steps.iter().map(|s| s.ero).collect()
    }
    /// 每步的 pivot 列(-1 = 無;→ `Int32Array`)。
    pub fn pivot_rows(&self) -> Vec<i32> {
        self.steps.iter().map(|s| s.pivot_row).collect()
    }
    /// 每步的 pivot 行(-1 = 無;→ `Int32Array`)。
    pub fn pivot_cols(&self) -> Vec<i32> {
        self.steps.iter().map(|s| s.pivot_col).collect()
    }
    /// 所有步的 working 快照串接成一條 `Float64Array`;前端用 `i * n * n` 起、
    /// `n * n` 長切出第 `i` 步再 reshape(下兩條同此)。
    pub fn working_snapshots(&self) -> Vec<f64> {
        self.steps
            .iter()
            .flat_map(|s| s.working.iter().copied())
            .collect()
    }
    /// 所有步的累積 P 快照串接(→ `Float64Array`)。
    pub fn p_snapshots(&self) -> Vec<f64> {
        self.steps
            .iter()
            .flat_map(|s| s.p.iter().copied())
            .collect()
    }
    /// 所有步的當步基本矩陣 Eₖ 快照串接(→ `Float64Array`)。
    pub fn e_snapshots(&self) -> Vec<f64> {
        self.steps
            .iter()
            .flat_map(|s| s.e.iter().copied())
            .collect()
    }
    /// `changed_rows` 鋸齒陣列的 CSR 值表(同 [`EliminationTrace::changed_rows_flat`])。
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
}

/// 反矩陣求解(Gauss-Jordan 累乘基本矩陣)的**逐步 trace**:把 A 化簡到 RREF,
/// 沿途記錄每個 ERO 對應的基本矩陣 Eₖ 與累積 P = Eₖ⋯E₁,供前端逐步播放、圖解。
///
/// - `data`:row-major flatten 的 n×n 元素,長度須為 `n * n`(前端保證,reshape 不檢查)。
/// - 終態:可逆時 working = Iₙ、P = A⁻¹;奇異時 working = RREF(A)、P·A = RREF(A)。
/// - pivot 已是 1 時跳過 scale 步(沿 `eliminate` backward pass 慣例,不污染 trace
///   ——終態 P 不受影響:P·A = Iₙ 唯一決定 P,與施作路徑無關)。
/// - epsilon 內部寫死 `TRACE_EPSILON`(沿 `eliminate` 慣例)。
#[wasm_bindgen]
pub fn invert_trace(data: Vec<f64>, n: usize) -> InverseTrace {
    let grid: Vec<Vec<f64>> = data.chunks(n).map(<[f64]>::to_vec).collect();
    let a = Matrix::from_rows(grid);
    let mut working = a.clone();
    let mut p = Matrix::identity(n);
    let mut steps: Vec<InverseStep> = Vec::new();

    // 第 0 步:原始狀態快照(working = A、P = E = Iₙ),讓播放能回到原貌。
    steps.push(InverseStep {
        description: "初始矩陣".to_string(),
        ero: ERO_INITIAL,
        pivot_row: -1,
        pivot_col: -1,
        changed_rows: Vec::new(),
        working: flatten(&working),
        p: flatten(&p),
        e: flatten(&Matrix::identity(n)),
    });

    // 單趟 Gauss-Jordan:逐 column 放 pivot,正規化後該 column 上下全清。
    // 與 core `inverse` 的差異只在 pivot_cursor(見模組註解),其餘鏡像它:
    // working 原地施作 ERO 走快路徑,E 由 elementary_* 建出、P 走字面的 E·p 累乘。
    let mut pivot_cursor = 0usize;
    for col in 0..n {
        if pivot_cursor >= n {
            break; // 列用完,剩下的 column 不會再有 pivot
        }
        let Some(pr) = working.pivot_row_below(col, pivot_cursor, TRACE_EPSILON) else {
            continue; // 這 column 沒 pivot → 跳過(奇異矩陣才會走到;inverse 在此 bail)
        };

        if pr != pivot_cursor {
            working.swap_rows(pivot_cursor, pr).expect("索引皆 < n");
            let e = Matrix::elementary_swap(n, pivot_cursor, pr).expect("索引皆 < n");
            p = e.multiply(&p).expect("同維方陣必可乘");
            steps.push(InverseStep {
                description: describe_swap(pivot_cursor, pr),
                ero: ERO_SWAP,
                pivot_row: pivot_cursor as i32,
                pivot_col: col as i32,
                changed_rows: vec![pivot_cursor as u32, pr as u32],
                working: flatten(&working),
                p: flatten(&p),
                e: flatten(&e),
            });
        }

        // pivot 量值 > ε 是 pivot_row_below 的後置條件 → 1/pivot 合法非零。
        let pivot_val = working.row(pivot_cursor).expect("pivot_cursor < n")[col];
        if (pivot_val - 1.0).abs() > TRACE_EPSILON {
            working
                .scale_row(pivot_cursor, 1.0 / pivot_val)
                .expect("pivot 非零");
            let e = Matrix::elementary_scale(n, pivot_cursor, 1.0 / pivot_val).expect("pivot 非零");
            p = e.multiply(&p).expect("同維方陣必可乘");
            steps.push(InverseStep {
                description: describe_scale(pivot_cursor, 1.0 / pivot_val),
                ero: ERO_SCALE,
                pivot_row: pivot_cursor as i32,
                pivot_col: col as i32,
                changed_rows: vec![pivot_cursor as u32],
                working: flatten(&working),
                p: flatten(&p),
                e: flatten(&e),
            });
        }

        for r in 0..n {
            if r == pivot_cursor {
                continue; // pivot 自己
            }
            let coeff = working.row(r).expect("r < n")[col];
            if coeff.abs() <= TRACE_EPSILON {
                continue; // 該格已是零:消去無作用,也不污染 trace
            }
            working
                .add_scaled_row(r, pivot_cursor, -coeff)
                .expect("r ≠ pivot_cursor");
            let e = Matrix::elementary_add_scaled(n, r, pivot_cursor, -coeff)
                .expect("r ≠ pivot_cursor");
            p = e.multiply(&p).expect("同維方陣必可乘");
            steps.push(InverseStep {
                description: describe_add_scaled(r, pivot_cursor, -coeff),
                ero: ERO_ADD_SCALED,
                pivot_row: pivot_cursor as i32,
                pivot_col: col as i32,
                changed_rows: vec![r as u32],
                working: flatten(&working),
                p: flatten(&p),
                e: flatten(&e),
            });
        }
        pivot_cursor += 1; // 只有真的放了 pivot 才前進
    }

    // ---- 終態旗標:各自獨立呼叫 core 的公開方法,IMT 等價條件分別驗 ----
    let invertible = a.is_invertible(TRACE_EPSILON);
    let rank = a.rank(TRACE_EPSILON);
    let nullity = a.nullity(TRACE_EPSILON);
    let cols: Vec<Vector> = (0..n).map(|j| a.column(j).expect("j < n")).collect();
    let cols_independent = is_linearly_independent(TRACE_EPSILON, &cols);

    InverseTrace {
        n,
        invertible,
        rank,
        nullity,
        cols_independent,
        steps,
    }
}

// ===========================================================================
// 線性轉換守恆律(單元 5-1)—— 給前端「拖動向量看 shear / 投影的影像,並親眼
// 驗證 T(u+v) = T(u)+T(v)、T(ku) = k·T(u)」圖解用。
//
// 與 core 的關係:T(x) 沿用既有的 transform_point(計算同源 multiply_vector);
// 這裡補上前端需要、但 JS 不准重寫的兩個向量運算(add / scale),以及把 core
// `transformation` 模組的 verify_linearity 原樣接出來 —— 前端顯示的 ✓/✗ 是
// core 親自驗的,不是 JS 寫死的字。
// ===========================================================================

/// 2D 向量加法 `u + v`,回傳 `[x, y]`。委派 core 的 [`Vector::add`];
/// 維度恆 2,`expect` 把「不會發生」寫成自證的不變式(沿 `transform_point` 慣例)。
#[wasm_bindgen]
pub fn add_vectors(ux: f64, uy: f64, vx: f64, vy: f64) -> Vec<f64> {
    let u = Vector::from_vec(vec![ux, uy]);
    let v = Vector::from_vec(vec![vx, vy]);
    u.add(&v)
        .expect("同為 2D 不會維度不匹配")
        .entries()
        .to_vec()
}

/// 2D 向量純量乘 `k·u`,回傳 `[x, y]`。委派 core 的 [`Vector::scale`](不會失敗)。
#[wasm_bindgen]
pub fn scale_vector(x: f64, y: f64, k: f64) -> Vec<f64> {
    Vector::from_vec(vec![x, y]).scale(k).entries().to_vec()
}

/// 2×2 矩陣 `[[a, b], [c, d]]` 誘導的轉換 T_A 在樣本 `(u, v, k)` 上的**線性檢查**:
/// T(u+v) = T(u)+T(v) 且 T(ku) = k·T(u)。
///
/// 直接委派 core 的 [`verify_linearity`] + [`Transformation::apply`] ——
/// Theorem 2.7 說矩陣誘導的轉換必過此檢查,所以前端看到的永遠是 ✓;這顆 binding
/// 的價值正是「✓ 由 core 當場驗出來」,不是前端寫死的裝飾。
/// epsilon 寫死 `1e-9`(沿 `are_parallel` 慣例:拖曳座標數量級穩定)。
// 9 個參數沿 `transform_point` 的「f64 過邊界零 marshalling」慣例:2×2·2D·純量
// 形狀固定,攤平比包陣列更不易錯 —— 故 allow 而不改簽名。
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn check_linearity(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    ux: f64,
    uy: f64,
    vx: f64,
    vy: f64,
    k: f64,
) -> bool {
    let t = Transformation::new(Matrix::from_rows(vec![vec![a, b], vec![c, d]]));
    let u = Vector::from_vec(vec![ux, uy]);
    let v = Vector::from_vec(vec![vx, vy]);
    verify_linearity(
        |x| t.apply(x).expect("2×2·2D 不會維度不匹配"),
        &u,
        &v,
        k,
        1e-9,
    )
}

// ===========================================================================
// 標準矩陣取樣(單元 5-2)—— 給前端「選幾何規則,看 e₁、e₂ 的影像被取樣、
// 直放成矩陣的行」圖解用。
//
// 與 core 的關係:幾何規則(旋轉/反射/剪切/投影/縮放)以**座標規則**的形式
// 住在這裡 —— 只有規則,沒有矩陣。矩陣由 core 的 [`standard_matrix`] 對規則
// 取樣「發現」(Theorem 2.9 的工作流原樣上演,呼應 `transformation.rs` 的
// x 軸反射測試:寫規則,讓構造器去發現矩陣)。`apply_rule` 是同一條規則的
// 直接施作,給前端畫「規則路徑 vs 矩陣路徑」的兩路會合(對帳測試釘住)。
// ===========================================================================

// 幾何規則編碼。用 `u8` 過邊界(沿 `PHASE_*` 慣例),前端的對照表順序必須一致。
const RULE_ROTATE: u8 = 0; // 旋轉 θ(param = 弧度)
const RULE_REFLECT_X: u8 = 1; // x 軸反射
const RULE_REFLECT_DIAG: u8 = 2; // 對 y = x 反射
const RULE_SHEAR_X: u8 = 3; // 水平剪切(param = k)
const RULE_PROJECT_X: u8 = 4; // 投影到 x 軸
const RULE_SCALE: u8 = 5; // 等比縮放(param = k)

/// 幾何規則本體:把 2D 向量依規則送到影像 —— **這裡只有規則,沒有矩陣字面值**。
/// 旋轉是 (x cosθ − y sinθ, x sinθ + y cosθ)、反射是「x 不動、y 翻號」⋯⋯
/// 全是課本上「幾何直觀」那一側的描述;矩陣那一側交給 `standard_matrix` 取樣。
fn rule_image(rule: u8, param: f64, v: &Vector) -> Vector {
    let (x, y) = (v.entries()[0], v.entries()[1]);
    let (ix, iy) = match rule {
        RULE_ROTATE => (
            x * param.cos() - y * param.sin(),
            x * param.sin() + y * param.cos(),
        ),
        RULE_REFLECT_X => (x, -y),
        RULE_REFLECT_DIAG => (y, x),
        RULE_SHEAR_X => (x + param * y, y),
        RULE_PROJECT_X => (x, 0.0),
        RULE_SCALE => (param * x, param * y),
        _ => unreachable!("前端只送上方定義的 rule 編碼"),
    };
    Vector::from_vec(vec![ix, iy])
}

/// 幾何規則的**標準矩陣**:core 的 [`standard_matrix`] 對規則做 e₁、e₂ 取樣,
/// 回傳 row-major `[a, b, c, d]`(Theorem 2.9:A 的第 j 行 = T(eⱼ))。
///
/// 頁面上的矩陣數字就是這裡「發現」出來的 —— 不是前端寫死的字面值,
/// 也不是 binding 抄好的公式;單元 5-2 練習 1 的構造器親自上場。
#[wasm_bindgen]
pub fn sample_standard_matrix(rule: u8, param: f64) -> Vec<f64> {
    let a = standard_matrix(2, |v| rule_image(rule, param, v))
        .expect("n = 2 ≥ 1 且規則恆回 2D —— 兩個 Err 都是死路");
    flatten(&a)
}

/// 幾何規則**直接施作**在點 `(x, y)`,回傳 `[x', y']` —— 「規則路徑」。
/// 前端用它畫 T(e₁)、T(e₂) 與 T(v);「矩陣路徑」則是 `transform_point` 左乘
/// 取樣矩陣 —— 兩條路會合即 Theorem 2.9(下方對帳測試釘住)。
#[wasm_bindgen]
pub fn apply_rule(rule: u8, param: f64, x: f64, y: f64) -> Vec<f64> {
    rule_image(rule, param, &Vector::from_vec(vec![x, y]))
        .entries()
        .to_vec()
}

// ===========================================================================
// 值域與映成(單元 5-3)—— 給前端「值域覆蓋」圖解用:拖動 A 的行向量看
// Range(T) = Col(A) 從整個平面塌成直線、再塌到原點;拖 w 看可達性即時判定,
// 不映成時把 unreachable_vector 的見證標在圖上。
//
// 與 core 的關係:全部直接委派 `range` 模組(range_basis / range_contains /
// is_onto / unreachable_vector)與 `system` 模組(solve)—— binding 只做
// 2×2 形狀的攤平與 Option / enum 的邊界編碼,零演算法(本章 core 模組的
// 「只有積木接線」精神原樣延伸到邊界層)。
// epsilon 一律寫死 TRACE_EPSILON(沿 eliminate 慣例:拖曳座標數量級穩定)。
// ===========================================================================

/// 把 row-major 的 2×2 純量升格為轉換 T_A: ℝ² → ℝ²(本區段五顆 binding 共用)。
fn transformation_2x2(a: f64, b: f64, c: f64, d: f64) -> Transformation {
    Transformation::new(Matrix::from_rows(vec![vec![a, b], vec![c, d]]))
}

/// Range(T) 的基底(core 的 `range_basis`),行向量攤平串接回傳:
/// `[]`(rank 0:值域 = {0})、`[x, y]`(rank 1:值域塌成直線)、
/// `[x₁, y₁, x₂, y₂]`(rank 2:值域 = ℝ²)—— 支數 = rank,長度就把維度說完了。
#[wasm_bindgen]
pub fn range_basis(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    transformation_2x2(a, b, c, d)
        .range_basis(TRACE_EPSILON)
        .iter()
        .flat_map(|v| v.entries().iter().copied())
        .collect()
}

/// `w ∈ Range(T)`?直接委派 core 的 `range_contains`(w 可達 ⟺ Ax = w 相容)——
/// 前端 w 箭頭的綠 / 紅是 core 當場判的,不是 JS 寫死的條件。
#[wasm_bindgen]
pub fn range_contains(a: f64, b: f64, c: f64, d: f64, wx: f64, wy: f64) -> bool {
    transformation_2x2(a, b, c, d).range_contains(&Vector::from_vec(vec![wx, wy]), TRACE_EPSILON)
}

/// T 映成嗎?直接委派 core 的 `is_onto`(Theorem 2.10:rank = m)。
#[wasm_bindgen]
pub fn is_onto(a: f64, b: f64, c: f64, d: f64) -> bool {
    transformation_2x2(a, b, c, d).is_onto(TRACE_EPSILON)
}

/// 不可達向量的見證(core 的 `unreachable_vector`,標準基底掃描):
/// 不映成 → `[x, y]`(某支 eᵢ);映成 → `[]`(`Option` 的邊界編碼:空陣列 = None)。
#[wasm_bindgen]
pub fn unreachable_vector(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    transformation_2x2(a, b, c, d)
        .unreachable_vector(TRACE_EPSILON)
        .map(|v| v.entries().to_vec())
        .unwrap_or_default()
}

/// 解 `Ax = w`「哪個輸入到得了 w」,回傳 `[kind, x, y]` ——
/// kind 沿 [`EliminationTrace`] 的 solution_kind 編碼(1 = Unique、2 = Infinite、
/// 3 = Inconsistent;前端對照表共用),x、y 只在 Unique 時有意義(其餘補 0)。
///
/// 與 `range_contains` 是同一個問題的兩種問法(可達 ⟺ kind ≠ Inconsistent,
/// 對帳測試釘住),但 solve 多給了「存在性的見證」:那個輸入 x 本身。
#[wasm_bindgen]
pub fn solve_for_input(a: f64, b: f64, c: f64, d: f64, wx: f64, wy: f64) -> Vec<f64> {
    let system = System::new(
        Matrix::from_rows(vec![vec![a, b], vec![c, d]]),
        Vector::from_vec(vec![wx, wy]),
    )
    .expect("2×2 配 2D 常數向量,維度必合");
    match system.solve(TRACE_EPSILON) {
        Solution::Unique(x) => vec![1.0, x.entries()[0], x.entries()[1]],
        Solution::Infinite => vec![2.0, 0.0, 0.0],
        Solution::Inconsistent => vec![3.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 90° 逆時針旋轉矩陣 `[[0, -1], [1, 0]]` 把 `(1, 0)` 送到 `(0, 1)`;
    /// 單位矩陣不動點。整數值在 f64 下精確,可用 `assert_eq!`。
    #[test]
    fn transform_point_applies_matrix() {
        assert_eq!(
            transform_point(0.0, -1.0, 1.0, 0.0, 1.0, 0.0),
            vec![0.0, 1.0]
        );
        assert_eq!(
            transform_point(1.0, 0.0, 0.0, 1.0, 7.0, 8.0),
            vec![7.0, 8.0]
        );
    }

    #[test]
    fn are_parallel_detects_collinearity() {
        assert!(are_parallel(1.0, 2.0, 2.0, 4.0)); // 純量倍數
        assert!(!are_parallel(1.0, 0.0, 0.0, 1.0)); // 垂直軸
        assert!(are_parallel(0.0, 0.0, 5.0, 7.0)); // 零向量與任意向量平行
    }

    #[test]
    fn determinant_2x2() {
        assert_eq!(determinant(0.0, -1.0, 1.0, 0.0), 1.0); // 90° 旋轉:面積不變
        assert_eq!(determinant(2.0, 0.0, 0.0, 3.0), 6.0); // 縮放:面積 ×6
        assert_eq!(determinant(1.0, 0.0, 0.0, -1.0), -1.0); // 鏡射:翻面
        assert_eq!(determinant(1.0, 2.0, 2.0, 4.0), 0.0); // 退化:塌成線
    }

    /// 向量加法 / 純量乘 binding:整數值在 f64 下精確,可用 `assert_eq!`。
    #[test]
    fn add_and_scale_vectors_delegate_to_core() {
        assert_eq!(add_vectors(1.0, 2.0, 3.0, -1.0), vec![4.0, 1.0]);
        assert_eq!(scale_vector(1.5, -2.0, 2.0), vec![3.0, -4.0]);
        assert_eq!(scale_vector(1.0, 2.0, 0.0), vec![0.0, 0.0]); // k=0 → 零向量
    }

    /// **Theorem 2.7 的 binding 對帳**:單元 5-1 頁面的三個 preset(shear、投影、
    /// 零轉換)都必須通過 core 的線性檢查 —— ✓ 是 core 驗出來的,不是前端寫死。
    /// 投影 det = 0(不可逆)仍線性,正是「線性 ≠ 可逆」的教學點。
    #[test]
    fn check_linearity_passes_matrix_transformations() {
        // shear [[1,1],[0,1]]
        assert!(check_linearity(
            1.0, 1.0, 0.0, 1.0, 1.0, -2.0, 3.0, 0.5, -1.5
        ));
        // 投影到 x 軸 [[1,0],[0,0]](det = 0,不可逆但線性)
        assert!(check_linearity(
            1.0, 0.0, 0.0, 0.0, 1.0, 2.0, -3.0, 4.0, 2.0
        ));
        // 零轉換 [[0,0],[0,0]]
        assert!(check_linearity(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0));
    }

    /// 全部六條幾何規則,搭配各自有意義的 param(無 param 的規則給 0)。
    fn all_rules() -> Vec<(u8, f64)> {
        vec![
            (RULE_ROTATE, 0.7),
            (RULE_REFLECT_X, 0.0),
            (RULE_REFLECT_DIAG, 0.0),
            (RULE_SHEAR_X, -1.5),
            (RULE_PROJECT_X, 0.0),
            (RULE_SCALE, -0.5),
        ]
    }

    /// 取樣出的標準矩陣對帳幾何規則的課本矩陣 —— Theorem 2.9 的黃金案例。
    /// 旋轉 90° 的 cos(π/2) 帶 ~6e-17 浮點殘差,容差比對;其餘規則整數精確。
    #[test]
    fn sample_standard_matrix_discovers_textbook_matrices() {
        let rot = sample_standard_matrix(RULE_ROTATE, std::f64::consts::FRAC_PI_2);
        for (got, want) in rot.iter().zip([0.0, -1.0, 1.0, 0.0]) {
            assert!(
                (got - want).abs() < 1e-12,
                "90° 旋轉應取樣出 [[0,−1],[1,0]],got={rot:?}"
            );
        }
        assert_eq!(
            sample_standard_matrix(RULE_REFLECT_X, 0.0),
            vec![1.0, 0.0, 0.0, -1.0]
        );
        assert_eq!(
            sample_standard_matrix(RULE_REFLECT_DIAG, 0.0),
            vec![0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            sample_standard_matrix(RULE_SHEAR_X, 2.0),
            vec![1.0, 2.0, 0.0, 1.0]
        );
        assert_eq!(
            sample_standard_matrix(RULE_PROJECT_X, 0.0),
            vec![1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            sample_standard_matrix(RULE_SCALE, 1.5),
            vec![1.5, 0.0, 0.0, 1.5]
        );
    }

    /// **Theorem 2.9 的 binding 對帳**:每條規則、同一測試點,「規則直接算」
    /// (apply_rule)與「左乘取樣矩陣」(transform_point)兩條路必須會合 ——
    /// 頁面上綠箭頭與白圓環重合的數學保證。
    #[test]
    fn apply_rule_agrees_with_sampled_matrix() {
        for (rule, param) in all_rules() {
            let m = sample_standard_matrix(rule, param);
            let (x, y) = (2.5, -1.25);
            let via_rule = apply_rule(rule, param, x, y);
            let via_matrix = transform_point(m[0], m[1], m[2], m[3], x, y);
            for (r, a) in via_rule.iter().zip(via_matrix.iter()) {
                assert!((r - a).abs() < 1e-12, "rule={rule}:兩條路應會合");
            }
        }
    }

    /// 每條幾何規則都通過 core 的線性檢查 —— 它們才有資格談「標準矩陣」
    /// (Theorem 2.9 的「若 T 線性」前提;非線性規則取樣出的矩陣重現不了規則)。
    #[test]
    fn geometry_rules_are_linear() {
        let u = Vector::from_vec(vec![1.0, -2.0]);
        let w = Vector::from_vec(vec![3.0, 0.5]);
        for (rule, param) in all_rules() {
            assert!(
                verify_linearity(|v| rule_image(rule, param, v), &u, &w, -2.5, 1e-9),
                "rule={rule} 應為線性"
            );
        }
    }

    #[test]
    fn can_multiply_requires_inner_dims_match() {
        assert!(can_multiply(2, 3, 3, 2)); // (2×3)·(3×2):內維 3 = 3 ✓
        assert!(!can_multiply(2, 3, 2, 3)); // (2×3)·(2×3):內維 3 ≠ 2 ✗
        assert!(can_multiply(1, 4, 4, 1)); // 列向量 · 欄向量 → 1×1
        assert!(can_multiply(2, 2, 2, 2)); // 同維方陣必可乘
    }

    /// **黃金迴歸**:`multiply_expand` 的 C 必須等於 core 的 `multiply`(整數在 f64
    /// 下精確,可用 `assert_eq!`),且每格 c_ij 必須等於其展開項之和、每項必須等於
    /// a_ik·b_kj —— binding 補算的展開項與 core 的乘法對帳,漂移即測試失敗。
    #[test]
    fn multiply_expand_terms_reconcile_with_core() {
        // 經典 (2×3)·(3×2) → 2×2:C = [[58, 64], [139, 154]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let exp = multiply_expand(a.clone(), 2, 3, b.clone(), 3, 2);

        assert_eq!((exp.rows, exp.cols, exp.inner), (2, 2, 3));

        // C == core 的 multiply
        let core_c = input_matrix(&a, 3).multiply(&input_matrix(&b, 2)).unwrap();
        assert_eq!(exp.c, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(exp.c, flatten(&core_c));

        // 每項 terms[(i·p+j)·n+k] == a_ik·b_kj,且 Σₖ == c_ij
        let (p, n) = (exp.cols, exp.inner);
        for i in 0..exp.rows {
            for j in 0..p {
                let base = (i * p + j) * n;
                for k in 0..n {
                    assert_eq!(exp.terms[base + k], a[i * 3 + k] * b[k * 2 + j]);
                }
                let sum: f64 = exp.terms[base..base + n].iter().sum();
                assert_eq!(sum, exp.c[i * p + j]);
            }
        }
    }

    /// 把一步的 flatten 快照 reshape 回 `Matrix`,方便與 core 的結果比對。
    fn snapshot_to_matrix(snap: &[f64], cols: usize) -> Matrix {
        Matrix::from_rows(snap.chunks(cols).map(<[f64]>::to_vec).collect())
    }

    /// 從 flatten 字面值建出輸入 `Matrix`(對照組用)。
    fn input_matrix(data: &[f64], cols: usize) -> Matrix {
        Matrix::from_rows(data.chunks(cols).map(<[f64]>::to_vec).collect())
    }

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

    // ===== invert_trace =====

    /// 可逆測試案例:(flatten data, n)。涵蓋 1×1 邊界、2×2、需換列的 3×3、4×4。
    fn invertible_cases() -> Vec<(Vec<f64>, usize)> {
        vec![
            (vec![4.0], 1), // 1×1:[c] 可逆 ⟺ c ≠ 0
            (vec![2.0, 1.0, 1.0, 1.0], 2),
            (vec![0.0, 2.0, 1.0, 4.0, 1.0, 0.0, 2.0, 1.0, 1.0], 3), // 第一個 pivot 要換列
            // 4×4 上三角、對角全 1 → det = 1,可逆(各列:R1=[1,2,0,1]、R2=[0,1,3,0]、
            // R3=[0,0,1,2]、R4=[0,0,0,1])
            (
                vec![
                    1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0,
                ],
                4,
            ),
        ]
    }

    /// 奇異測試案例:零矩陣、第二列 = 2×第一列、rank 2 的 3×3。
    fn singular_cases() -> Vec<(Vec<f64>, usize)> {
        vec![
            (vec![0.0, 0.0, 0.0, 0.0], 2),
            (vec![1.0, 2.0, 2.0, 4.0], 2),
            (vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0], 3),
        ]
    }

    /// **黃金迴歸 A**:可逆時,trace 終態的 P 必須等於 core 的 `inverse()`、working
    /// 必須化到 Iₙ。P·A = Iₙ 唯一決定 P,與施作路徑無關 —— 即使步驟序列漂移
    /// (如 no-op scale 的取捨),終態比對仍然必須成立。
    #[test]
    fn invert_trace_invertible_ends_at_core_inverse() {
        for (data, n) in invertible_cases() {
            let trace = invert_trace(data.clone(), n);
            let a = input_matrix(&data, n);
            let last = trace.steps.last().unwrap();
            let got_p = snapshot_to_matrix(&last.p, n);
            let want = a.inverse(TRACE_EPSILON).unwrap();
            assert!(
                got_p.approx_equals(&want, 1e-7),
                "終態 P 應等於 core 的 inverse\n got={got_p:?}\n want={want:?}"
            );
            let got_w = snapshot_to_matrix(&last.working, n);
            assert!(
                got_w.approx_equals(&Matrix::identity(n), 1e-7),
                "可逆時 working 終態應為 Iₙ\n got={got_w:?}"
            );
        }
    }

    /// **黃金迴歸 B**:奇異時 trace 不 bail、做完整趟,working 終態 = core 的 RREF
    /// (RREF 唯一,同樣與路徑無關)——「RREF 含零列 → 不可逆」由畫面自然呈現。
    #[test]
    fn invert_trace_singular_ends_at_core_rref() {
        for (data, n) in singular_cases() {
            let trace = invert_trace(data.clone(), n);
            let a = input_matrix(&data, n);
            let got_w = snapshot_to_matrix(&trace.steps.last().unwrap().working, n);
            let want = a.reduced_row_echelon_form(TRACE_EPSILON);
            assert!(
                got_w.approx_equals(&want, 1e-7),
                "奇異時 working 終態應為 core 的 RREF\n got={got_w:?}\n want={want:?}"
            );
        }
    }

    /// **Theorem 2.3 的 trace 版**:無論可逆或奇異,終態恆有 P·A = working
    /// (可逆時 working = Iₙ 即 P = A⁻¹;奇異時 working = RREF)。
    #[test]
    fn invert_trace_p_times_a_equals_working() {
        for (data, n) in invertible_cases().into_iter().chain(singular_cases()) {
            let trace = invert_trace(data.clone(), n);
            let a = input_matrix(&data, n);
            let last = trace.steps.last().unwrap();
            let pa = snapshot_to_matrix(&last.p, n).multiply(&a).unwrap();
            let w = snapshot_to_matrix(&last.working, n);
            assert!(
                pa.approx_equals(&w, 1e-7),
                "P·A 應等於 working(Theorem 2.3)\n pa={pa:?}\n w={w:?}"
            );
        }
    }

    /// **Proposition 的 trace 版(逐步重放)**:每一步的 Eₖ 左乘上一步的 working / P,
    /// 必須得到這一步的 working / P。working 走原地 ERO 快路徑、P 走字面 E·p 累乘,
    /// 兩條線若與記錄下來的 Eₖ 漂移,這裡會抓到。
    #[test]
    fn invert_trace_replays_step_by_step() {
        for (data, n) in invertible_cases().into_iter().chain(singular_cases()) {
            let trace = invert_trace(data.clone(), n);
            for k in 1..trace.steps.len() {
                let prev = &trace.steps[k - 1];
                let cur = &trace.steps[k];
                let e = snapshot_to_matrix(&cur.e, n);
                let got_w = e.multiply(&snapshot_to_matrix(&prev.working, n)).unwrap();
                let got_p = e.multiply(&snapshot_to_matrix(&prev.p, n)).unwrap();
                assert!(
                    got_w.approx_equals(&snapshot_to_matrix(&cur.working, n), 1e-7),
                    "第 {k} 步:Eₖ·workingₖ₋₁ ≠ workingₖ\n data={data:?}"
                );
                assert!(
                    got_p.approx_equals(&snapshot_to_matrix(&cur.p, n), 1e-7),
                    "第 {k} 步:Eₖ·Pₖ₋₁ ≠ Pₖ\n data={data:?}"
                );
            }
        }
    }

    /// 終態旗標與 core 的對應方法逐一一致(IMT 面板的「誠實分別驗」)。
    #[test]
    fn invert_trace_flags_match_core() {
        for (data, n) in invertible_cases().into_iter().chain(singular_cases()) {
            let trace = invert_trace(data.clone(), n);
            let a = input_matrix(&data, n);
            assert_eq!(trace.invertible, a.is_invertible(TRACE_EPSILON));
            assert_eq!(trace.rank, a.rank(TRACE_EPSILON));
            assert_eq!(trace.nullity, a.nullity(TRACE_EPSILON));
            let cols: Vec<Vector> = (0..n).map(|j| a.column(j).unwrap()).collect();
            assert_eq!(
                trace.cols_independent,
                is_linearly_independent(TRACE_EPSILON, &cols)
            );
        }
    }

    /// 第 0 步是 initial:working = A、P = E = Iₙ、無 pivot —— 讓播放能回到原貌。
    #[test]
    fn invert_trace_first_step_is_initial() {
        let data = vec![2.0, 1.0, 1.0, 1.0];
        let trace = invert_trace(data.clone(), 2);
        let first = &trace.steps[0];
        assert_eq!(first.ero, ERO_INITIAL);
        assert_eq!(first.working, data);
        assert_eq!(first.p, vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(first.e, vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(first.pivot_row, -1);
        assert_eq!(first.pivot_col, -1);
    }

    // ---- 值域與映成(單元 5-3 頁面)的 binding 對帳 ----

    /// 三種秩的代表矩陣:rank 2(可逆)、rank 1(行成比例)、rank 0(零矩陣),
    /// 以 row-major `[a, b, c, d]` 表示 —— 值域分別是 ℝ²、直線、{0}。
    const FULL_RANK: [f64; 4] = [2.0, 1.0, 1.0, 1.0];
    const RANK_ONE: [f64; 4] = [1.0, 2.0, 2.0, 4.0]; // 行 (1,2) 與 (2,4) 共線
    const RANK_ZERO: [f64; 4] = [0.0, 0.0, 0.0, 0.0];

    /// 基底的攤平長度把 Range 的維度說完:4 個數(平面)、2 個數(直線)、
    /// 空(原點)。rank 1 的基底是 pivot 行對應的**原始行** (1, 2)。
    #[test]
    fn range_basis_length_encodes_collapse() {
        let [a, b, c, d] = FULL_RANK;
        assert_eq!(range_basis(a, b, c, d).len(), 4, "rank 2:兩支基底");
        let [a, b, c, d] = RANK_ONE;
        assert_eq!(range_basis(a, b, c, d), vec![1.0, 2.0], "rank 1:原始行 0");
        let [a, b, c, d] = RANK_ZERO;
        assert!(range_basis(a, b, c, d).is_empty(), "rank 0:空基底");
    }

    /// 可達性判定:rank 1 的值域是直線 span{(1,2)},線上的 (3,6) 可達、
    /// 線外的 (1,1) 不可達;滿秩則整個 ℝ² 都可達。
    #[test]
    fn range_contains_judges_reachability() {
        let [a, b, c, d] = RANK_ONE;
        assert!(range_contains(a, b, c, d, 3.0, 6.0));
        assert!(!range_contains(a, b, c, d, 1.0, 1.0));
        let [a, b, c, d] = FULL_RANK;
        assert!(range_contains(a, b, c, d, -7.5, 4.25), "滿秩:處處可達");
    }

    /// 映成判定與見證的對偶(core 對偶律的 binding 重述):onto ⟺ 見證為空;
    /// 不映成時見證必須真的不可達 —— 前端紅色標記的數學保證。
    #[test]
    fn unreachable_witness_dual_to_is_onto() {
        for m in [FULL_RANK, RANK_ONE, RANK_ZERO] {
            let [a, b, c, d] = m;
            let witness = unreachable_vector(a, b, c, d);
            assert_eq!(witness.is_empty(), is_onto(a, b, c, d), "對偶律");
            if let [wx, wy] = witness[..] {
                assert!(!range_contains(a, b, c, d, wx, wy), "見證居然可達");
            }
        }
    }

    /// solve_for_input 的三種結局,與 range_contains 對帳(可達 ⟺ kind ≠ 3):
    /// 滿秩 → Unique 且回傳的 x 經 transform_point 左乘必須回到 w(兩路會合);
    /// rank 1 線上 → Infinite(一整條輸入);線外 → Inconsistent。
    #[test]
    fn solve_for_input_classifies_and_returns_witness() {
        // 滿秩:唯一輸入,且 A·x = w(存在性的見證拿去矩陣路徑驗收)
        let [a, b, c, d] = FULL_RANK;
        let out = solve_for_input(a, b, c, d, 5.0, 3.0);
        assert_eq!(out[0], 1.0, "可逆 → Unique");
        let back = transform_point(a, b, c, d, out[1], out[2]);
        assert!((back[0] - 5.0).abs() < 1e-9 && (back[1] - 3.0).abs() < 1e-9);

        // rank 1:線上 → Infinite、線外 → Inconsistent;與 range_contains 一致
        let [a, b, c, d] = RANK_ONE;
        assert_eq!(solve_for_input(a, b, c, d, 3.0, 6.0)[0], 2.0);
        assert_eq!(solve_for_input(a, b, c, d, 1.0, 1.0)[0], 3.0);
        for (wx, wy) in [(3.0, 6.0), (1.0, 1.0), (0.0, 0.0)] {
            let kind = solve_for_input(a, b, c, d, wx, wy)[0];
            assert_eq!(
                range_contains(a, b, c, d, wx, wy),
                kind != 3.0,
                "可達 ⟺ 有解"
            );
        }
    }
}
