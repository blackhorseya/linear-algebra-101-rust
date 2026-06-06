import init, {
  transform_point,
  are_parallel,
  determinant,
  can_multiply,
  multiply_expand,
  eliminate,
  invert_trace,
  add_vectors,
  scale_vector,
  check_linearity,
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

/** 矩陣乘法 C = A·B 的結果與逐格 row × col 展開(plain-JS,不持有 WASM 指標)。 */
export interface MultiplyExpansionJS {
  /** C 的列數 m(= A 的列數)。 */
  rows: number;
  /** C 的欄數 p(= B 的欄數)。 */
  cols: number;
  /** 內維 n(= A 的欄數 = B 的列數)。 */
  inner: number;
  /** C = A·B(rows×cols),由 core 的 multiply 計算。 */
  c: number[][];
  /** 展開項:terms[i][j][k] = aᵢₖ·bₖⱼ,且 c[i][j] = Σₖ terms[i][j][k]。 */
  terms: number[][][];
}

/** 求逆 trace 每一步施作的 ERO 種類("initial" = 第 0 步,尚未施作)。 */
export type EroKind = "initial" | "swap" | "scale" | "addScaled";

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

// WASM 端的 u8 編碼 → 字串(index 即編碼值,順序必須與 wasm.rs 一致)。
const PHASE_NAMES: ElimPhase[] = ["initial", "forward", "backward"];
const SOLUTION_NAMES: SolutionKind[] = [
  "NA",
  "Unique",
  "Infinite",
  "Inconsistent",
];
const ERO_NAMES: EroKind[] = ["initial", "swap", "scale", "addScaled"];

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
  /** A(aRows×aCols) 能否右乘 B(bRows×bCols)。維度規則由 core 的 can_multiply 判定。 */
  canMultiply: (
    aRows: number,
    aCols: number,
    bRows: number,
    bCols: number,
  ) => boolean;
  /**
   * 矩陣乘法 C = A·B 與每格的 row × col 展開項。`aData` / `bData` 為 row-major
   * flatten。**呼叫前先以 `canMultiply` 確認維度相容**(內維不等會直接 panic)。
   */
  multiplyExpand: (
    aData: number[],
    aRows: number,
    aCols: number,
    bData: number[],
    bRows: number,
    bCols: number,
  ) => MultiplyExpansionJS;
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
  /**
   * 反矩陣(Gauss-Jordan 累乘基本矩陣)逐步 trace。`data` 為 row-major flatten
   * 的 n×n 矩陣。可逆時終態 P = A⁻¹;奇異時終態 working = RREF(A)(Theorem 2.3)。
   */
  invertTrace: (data: number[], n: number) => InverseTraceJS;
  /** 2D 向量加法 u + v(core 的 `Vector::add`),回傳 `[x, y]`。 */
  addVectors: (
    ux: number,
    uy: number,
    vx: number,
    vy: number,
  ) => Float64Array;
  /** 2D 向量純量乘 k·u(core 的 `Vector::scale`),回傳 `[x, y]`。 */
  scaleVector: (x: number, y: number, k: number) => Float64Array;
  /**
   * 2×2 矩陣誘導的 T_A 在樣本 (u, v, k) 上的線性檢查:T(u+v) = T(u)+T(v) 且
   * T(ku) = k·T(u)。委派 core 的 `verify_linearity`(Theorem 2.7:矩陣誘導必過)。
   */
  checkLinearity: (
    a: number,
    b: number,
    c: number,
    d: number,
    ux: number,
    uy: number,
    vx: number,
    vy: number,
    k: number,
  ) => boolean;
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

/**
 * 同 `runEliminate` 的 SoA 縫合 + `free()` 模式,差別在每步有三份 n×n 快照
 * (working / 累積 P / 當步 Eₖ),各自從串接的 typed array 切片 reshape。
 */
function runInvertTrace(data: number[], n: number): InverseTraceJS {
  const trace = invert_trace(Float64Array.from(data), n);
  try {
    const stepCount = trace.step_count;
    const cells = n * n;

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

    // 從串接快照切出第 i 步的 n×n 矩陣。
    const sliceMatrix = (snaps: Float64Array, i: number): number[][] => {
      const base = i * cells;
      const matrix: number[][] = [];
      for (let r = 0; r < n; r++) {
        const start = base + r * n;
        matrix.push(Array.from(snaps.subarray(start, start + n)));
      }
      return matrix;
    };

    const steps: InverseStepJS[] = [];
    for (let i = 0; i < stepCount; i++) {
      steps.push({
        description: descriptions[i],
        ero: ERO_NAMES[eros[i]],
        working: sliceMatrix(workingSnaps, i),
        p: sliceMatrix(pSnaps, i),
        e: sliceMatrix(eSnaps, i),
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

/**
 * 同 `runEliminate` 的 SoA 縫合 + `free()` 模式:c 與 terms 各跨界一次,
 * 縫回 plain-JS 的巢狀陣列後釋放 WASM 物件。
 */
function runMultiplyExpand(
  aData: number[],
  aRows: number,
  aCols: number,
  bData: number[],
  bRows: number,
  bCols: number,
): MultiplyExpansionJS {
  const exp = multiply_expand(
    Float64Array.from(aData),
    aRows,
    aCols,
    Float64Array.from(bData),
    bRows,
    bCols,
  );
  try {
    const rows = exp.rows;
    const cols = exp.cols;
    const inner = exp.inner;
    const cFlat = exp.c();
    const termsFlat = exp.terms();

    const c: number[][] = [];
    const terms: number[][][] = [];
    for (let i = 0; i < rows; i++) {
      const cRow: number[] = [];
      const tRow: number[][] = [];
      for (let j = 0; j < cols; j++) {
        cRow.push(cFlat[i * cols + j]);
        // 第 (i, j) 格的 n 個展開項:flat[(i·p + j)·n .. +n]
        const base = (i * cols + j) * inner;
        tRow.push(Array.from(termsFlat.subarray(base, base + inner)));
      }
      c.push(cRow);
      terms.push(tRow);
    }

    return { rows, cols, inner, c, terms };
  } finally {
    exp.free(); // 抽完資料,釋放 WASM 物件(回傳的 plain-JS 不再依賴它)
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
    canMultiply: can_multiply,
    multiplyExpand: runMultiplyExpand,
    eliminate: runEliminate,
    invertTrace: runInvertTrace,
    addVectors: add_vectors,
    scaleVector: scale_vector,
    checkLinearity: check_linearity,
  }));
  return instance;
}
