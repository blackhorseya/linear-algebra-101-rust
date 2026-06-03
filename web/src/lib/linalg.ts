import init, {
  transform_point,
  are_parallel,
  determinant,
  eliminate,
} from "./wasm/linear_algebra_101.js";
// `--target web` 的 glue 不會自動 import .wasm,要把它的 URL 交給 init()。
// `?url` 讓 Vite 把這顆 wasm 當資產處理並回傳可 fetch 的網址(dev / build 皆然)。
import wasmUrl from "./wasm/linear_algebra_101_bg.wasm?url";

/** 消去法每一步所屬的階段。 */
export type ElimPhase = "initial" | "forward" | "backward";

/** 線性方程組解的型態(一般矩陣模式為 "NA")。 */
export type SolutionKind = "NA" | "Unique" | "Infinite" | "Inconsistent";

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

// WASM 端的 u8 編碼 → 字串(index 即編碼值,順序必須與 wasm.rs 一致)。
const PHASE_NAMES: ElimPhase[] = ["initial", "forward", "backward"];
const SOLUTION_NAMES: SolutionKind[] = [
  "NA",
  "Unique",
  "Infinite",
  "Inconsistent",
];

/** 初始化後可用的線代運算(全部在 Rust 算,JS 只是轉呼叫)。 */
export interface Linalg {
  /** 2×2 矩陣 `[[a,b],[c,d]]` 作用在點 `(x,y)`,回傳變換後的 `[x', y']`。 */
  transformPoint: (
    a: number,
    b: number,
    c: number,
    d: number,
    x: number,
    y: number,
  ) => Float64Array;
  /** 兩個 2D 向量是否平行(共線 / 線性相依)。 */
  areParallel: (ux: number, uy: number, wx: number, wy: number) => boolean;
  /** 2×2 矩陣 `[[a,b],[c,d]]` 的行列式 ad−bc(= 單位正方形像的有號面積)。 */
  determinant: (a: number, b: number, c: number, d: number) => number;
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

/**
 * 把 WASM 的 SoA 平行陣列縫回乾淨的 plain-JS trace,並在抽完資料後 `free()` 掉 WASM
 * 物件 —— 回傳的物件不持有任何 WASM 指標,呼叫端不必管生命週期。
 *
 * SoA(Structure-of-Arrays)的好處:整趟 trace 只跨 WASM 邊界少數幾次(每個欄位一條
 * typed array),而非「每個 step 一個帶指標的 JS 物件」。
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
    const cells = tRows * tCols;

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
      // 從串接的快照切出第 i 步,再 reshape 成 rows×cols。
      const base = i * cells;
      const matrix: number[][] = [];
      for (let r = 0; r < tRows; r++) {
        const start = base + r * tCols;
        matrix.push(Array.from(snapshots.subarray(start, start + tCols)));
      }
      steps.push({
        description: descriptions[i],
        phase: PHASE_NAMES[phases[i]],
        matrix,
        pivotRow: pivotRows[i],
        pivotCol: pivotCols[i],
        // CSR:第 i 步的 changed rows = flat[offsets[i] .. offsets[i+1]]
        changedRows: Array.from(
          changedFlat.subarray(changedOffsets[i], changedOffsets[i + 1]),
        ),
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

// 模組層級 memoize:init 是非同步且只該跑一次。即使多個元件同時呼叫,
// 也共用同一個 Promise(配合 Query 的 staleTime: Infinity 是雙重保險)。
let instance: Promise<Linalg> | null = null;

/** 載入並初始化 WASM 模組,回傳綁好的運算 API。重複呼叫共用同一次初始化。 */
export function loadLinalg(): Promise<Linalg> {
  // 已初始化就早退(讓 TS 在後續流程把 instance 收窄為非 null)。
  if (instance) return instance;
  instance = init({ module_or_path: wasmUrl }).then(() => ({
    transformPoint: transform_point,
    areParallel: are_parallel,
    determinant,
    eliminate: runEliminate,
  }));
  return instance;
}
