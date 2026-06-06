//! 反矩陣(Gauss-Jordan 累乘基本矩陣)逐步 trace —— 「可逆矩陣與基本矩陣」圖解用。
//!
//! 與 core 的關係:`Matrix::inverse` 只回最終的 P = Eₖ⋯E₁,把每一步的 Eₖ 與中間
//! 累積吃掉了;鐵律又是 **core 零改動**。所以沿 `eliminate` 的前例,在 binding 層
//! 重跑同一套演算法、沿途把每一步攔下來記錄。pivot 搜尋直接用 pub(crate) 的
//! `Matrix::pivot_row_below`(與 core `inverse` 同一顆積木),不再鏡像重寫。
//!
//! 演算法是 `inverse()` 的**一般化**:同樣單趟 Gauss-Jordan,但 pivot row 改用
//! 獨立游標(不釘對角線),某 column 找不到 pivot 時跳過該 column 繼續(而非 bail)。
//! - 可逆時:每個 column 都有 pivot、游標恆等於 column,行為與 `inverse()` 一致,
//!   終態 P 即 A⁻¹(黃金迴歸 `invert_trace_invertible_ends_at_core_inverse`)。
//! - 奇異時:做完整趟得 working = RREF(A) 且 P·A = RREF(A) —— Theorem 2.3
//!   (PA = R,P 為基本矩陣之乘積)的完整展示,「RREF 含零列 → 不可逆」自然浮現。

use super::helpers::{TRACE_EPSILON, describe_add_scaled, describe_scale, describe_swap, flatten};
use crate::{Matrix, Vector, is_linearly_independent};
use wasm_bindgen::prelude::*;

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

/// 一趟完整求逆的 trace,過 WASM 邊界的單一物件(SoA 策略同
/// [`EliminationTrace`](super::elimination::EliminationTrace))。
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
    /// `changed_rows` 鋸齒陣列的 CSR 值表(同
    /// [`EliminationTrace::changed_rows_flat`](super::elimination::EliminationTrace::changed_rows_flat))。
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::helpers::{input_matrix, snapshot_to_matrix};

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
}
