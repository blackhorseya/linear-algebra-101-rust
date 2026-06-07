// 反矩陣視覺化(/invertibility)的 binding —— 鏡像 src/wasm/inverse.rs。
import { invert_trace } from "../wasm/linear_algebra_101.js";
import { csrRow, sliceMatrix } from "./helpers";

/** 求逆 trace 每一步施作的 ERO 種類("initial" = 第 0 步,尚未施作)。 */
export type EroKind = "initial" | "swap" | "scale" | "addScaled";

// WASM 端的 u8 編碼 → 字串(index 即編碼值,順序必須與 inverse.rs 一致)。
const ERO_NAMES: EroKind[] = ["initial", "swap", "scale", "addScaled"];

/** 求逆(Gauss-Jordan 累乘基本矩陣)的單一步驟(plain-JS)。 */
export interface InverseStepJS {
  /** 人類可讀的操作描述,如 "R2 ← R2 − 2·R1"。 */
  description: string;
  /** 這一步施作的 ERO 種類(前端據此映射幾何名稱:鏡射 / 伸縮 / 剪切)。 */
  ero: EroKind;
  /** 消去中的矩陣(這一步做完之後;從 A 出發,終態 = RREF(A))。 */
  working: number[][];
  /** 累積 P = Eₖ⋯E₁(從 Iₙ 出發,可逆時終態 = A⁻¹)。 */
  p: number[][];
  /** 這一步施作的基本矩陣 Eₖ(initial 步 = Iₙ)。 */
  e: number[][];
  /** 當前 pivot 列(-1 = 無)。 */
  pivotRow: number;
  /** 當前 pivot 行(-1 = 無)。 */
  pivotCol: number;
  /** 這一步被改動的列(高亮用)。 */
  changedRows: number[];
}

/** 一趟完整求逆的 trace(plain-JS,不持有任何 WASM 指標)。 */
export interface InverseTraceJS {
  steps: InverseStepJS[];
  n: number;
  /** 以下 IMT 旗標各自由 Rust 端獨立計算,非彼此推導(等價是定理,不是實作)。 */
  invertible: boolean;
  rank: number;
  nullity: number;
  colsIndependent: boolean;
}

/**
 * 同 `runEliminate` 的 SoA 縫合 + `free()` 模式,差別在每步有三份 n×n 快照
 * (working / 累積 P / 當步 Eₖ),各自從串接的 typed array 切片 reshape。
 */
function runInvertTrace(data: number[], n: number): InverseTraceJS {
  const trace = invert_trace(Float64Array.from(data), n);
  try {
    const stepCount = trace.step_count;

    // 各 SoA 陣列各跨界一次。
    const workingSnaps = trace.working_snapshots();
    const pSnaps = trace.p_snapshots();
    const eSnaps = trace.e_snapshots();
    const descriptions = trace.descriptions();
    const eros = trace.eros();
    const pivotRows = trace.pivot_rows();
    const pivotCols = trace.pivot_cols();
    const changedFlat = trace.changed_rows_flat();
    const changedOffsets = trace.changed_rows_offsets();

    const steps: InverseStepJS[] = [];
    for (let i = 0; i < stepCount; i++) {
      steps.push({
        description: descriptions[i],
        ero: ERO_NAMES[eros[i]],
        working: sliceMatrix(workingSnaps, i, n, n),
        p: sliceMatrix(pSnaps, i, n, n),
        e: sliceMatrix(eSnaps, i, n, n),
        pivotRow: pivotRows[i],
        pivotCol: pivotCols[i],
        changedRows: csrRow(changedFlat, changedOffsets, i),
      });
    }

    return {
      steps,
      n: trace.n,
      invertible: trace.invertible,
      rank: trace.rank,
      nullity: trace.nullity,
      colsIndependent: trace.cols_independent,
    };
  } finally {
    trace.free(); // 同上:抽完即釋放
  }
}

/** inverse 章的運算(攤平進 `Linalg`)。 */
export interface InverseOps {
  /**
   * 反矩陣(Gauss-Jordan 累乘基本矩陣)逐步 trace。`data` 為 row-major flatten
   * 的 n×n 矩陣。可逆時終態 P = A⁻¹;奇異時終態 working = RREF(A)(Theorem 2.3)。
   */
  invertTrace: (data: number[], n: number) => InverseTraceJS;
}

export const inverseOps: InverseOps = {
  invertTrace: runInvertTrace,
};
