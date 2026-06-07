// 高斯消去視覺化(/elimination)的 binding —— 鏡像 src/wasm/elimination.rs。
import { eliminate } from "../wasm/linear_algebra_101.js";
import { csrRow, sliceMatrix } from "./helpers";

/** 消去法每一步所屬的階段。 */
export type ElimPhase = "initial" | "forward" | "backward";

/**
 * 線性方程組解的型態(一般矩陣模式為 "NA")。
 *
 * 編碼定義在 elimination 章(wasm 端同樣由 elimination.rs 定義),
 * range 章的 `solveForInput` 沿用同一張對照表。
 */
export type SolutionKind = "NA" | "Unique" | "Infinite" | "Inconsistent";

// WASM 端的 u8 編碼 → 字串(index 即編碼值,順序必須與 elimination.rs 一致)。
const PHASE_NAMES: ElimPhase[] = ["initial", "forward", "backward"];
export const SOLUTION_NAMES: SolutionKind[] = [
  "NA",
  "Unique",
  "Infinite",
  "Inconsistent",
];

/** 消去法的單一步驟(已從 WASM 的 SoA 縫回乾淨的 plain-JS 物件)。 */
export interface EliminationStepJS {
  /** 人類可讀的操作描述,如 "R2 ← R2 − 2·R1"。 */
  description: string;
  /** 這一步屬於哪個階段。 */
  phase: ElimPhase;
  /** 這一步「做完之後」的矩陣快照(row-major,rows×cols)。 */
  matrix: number[][];
  /** 當前 pivot 列(-1 = 無)。 */
  pivotRow: number;
  /** 當前 pivot 行(-1 = 無)。 */
  pivotCol: number;
  /** 這一步被改動的列(高亮用)。 */
  changedRows: number[];
}

/** 一趟完整消去的 trace(plain-JS,不持有任何 WASM 指標)。 */
export interface EliminationTraceJS {
  steps: EliminationStepJS[];
  rows: number;
  cols: number;
  /** 增廣欄起點;-1 = 一般矩陣。 */
  augCol: number;
  rank: number;
  solutionKind: SolutionKind;
  /** 終態 pivot(基本變數)行。 */
  pivotColumns: number[];
  /** 終態 free(自由變數)行。 */
  freeColumns: number[];
}

/**
 * 把 WASM 的 SoA 平行陣列縫回乾淨的 plain-JS trace,並在抽完資料後 `free()` 掉
 * WASM 物件 —— 回傳的物件不持有任何 WASM 指標,呼叫端不必管生命週期。
 * (SoA 縫合的共用切片工具在 helpers.ts;本函數是這個模式的最早範例。)
 */
function runEliminate(
  data: number[],
  rows: number,
  cols: number,
  augCol: number,
): EliminationTraceJS {
  const trace = eliminate(Float64Array.from(data), rows, cols, augCol);
  try {
    const stepCount = trace.step_count;
    const tRows = trace.rows;
    const tCols = trace.cols;

    // 各 SoA 陣列各跨界一次(typed array / string[] 皆已是 JS heap 的 copy)。
    const snapshots = trace.snapshots();
    const descriptions = trace.descriptions();
    const phases = trace.phases();
    const pivotRows = trace.pivot_rows();
    const pivotCols = trace.pivot_cols();
    const changedFlat = trace.changed_rows_flat();
    const changedOffsets = trace.changed_rows_offsets();

    const steps: EliminationStepJS[] = [];
    for (let i = 0; i < stepCount; i++) {
      steps.push({
        description: descriptions[i],
        phase: PHASE_NAMES[phases[i]],
        // 從串接的快照切出第 i 步,再 reshape 成 rows×cols。
        matrix: sliceMatrix(snapshots, i, tRows, tCols),
        pivotRow: pivotRows[i],
        pivotCol: pivotCols[i],
        changedRows: csrRow(changedFlat, changedOffsets, i),
      });
    }

    return {
      steps,
      rows: tRows,
      cols: tCols,
      augCol: trace.aug_col,
      rank: trace.rank,
      solutionKind: SOLUTION_NAMES[trace.solution_kind],
      pivotColumns: Array.from(trace.pivot_columns()),
      freeColumns: Array.from(trace.free_columns()),
    };
  } finally {
    trace.free(); // 抽完資料,釋放 WASM 物件(回傳的 plain-JS 不再依賴它)
  }
}

/** elimination 章的運算(攤平進 `Linalg`)。 */
export interface EliminationOps {
  /**
   * 高斯消去法(Gauss-Jordan)逐步 trace。`data` 為 row-major flatten 的矩陣;
   * `augCol` 為增廣欄起點(-1 = 一般矩陣)。回傳乾淨的 plain-JS trace。
   */
  eliminate: (
    data: number[],
    rows: number,
    cols: number,
    augCol: number,
  ) => EliminationTraceJS;
}

export const eliminationOps: EliminationOps = {
  eliminate: runEliminate,
};
