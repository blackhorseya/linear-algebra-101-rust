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
  sample_standard_matrix,
  apply_rule,
  range_basis,
  range_contains,
  is_onto,
  unreachable_vector,
  solve_for_input,
  compose_matrix,
  inverse_matrix,
  is_one_to_one,
  transformation_report,
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

/** 幾何規則(標準矩陣取樣頁)。具名字串對前端友善,過邊界前再映射成 u8。 */
export type RuleKind =
  | "rotate"
  | "reflectX"
  | "reflectDiag"
  | "shearX"
  | "projectX"
  | "scale";

// 規則名 → WASM 的 u8 編碼(數值必須與 wasm.rs 的 RULE_* 一致)。
const RULE_CODES: Record<RuleKind, number> = {
  rotate: 0,
  reflectX: 1,
  reflectDiag: 2,
  shearX: 3,
  projectX: 4,
  scale: 5,
};

// WASM 端的 u8 編碼 → 字串(index 即編碼值,順序必須與 wasm.rs 一致)。
const PHASE_NAMES: ElimPhase[] = ["initial", "forward", "backward"];
const SOLUTION_NAMES: SolutionKind[] = [
  "NA",
  "Unique",
  "Infinite",
  "Inconsistent",
];
const ERO_NAMES: EroKind[] = ["initial", "swap", "scale", "addScaled"];

/** 可逆性綜合判定表(合成與可逆性頁):core 的 `report` 三燈,縫回具名物件。 */
export interface TransformationReportJS {
  /** 一對一(Theorem 2.11:rank = n)。 */
  isOneToOne: boolean;
  /** 映成(Theorem 2.10:rank = m)。 */
  isOnto: boolean;
  /** 可逆(Theorem 2.12:1-1 且 onto)—— 合取在 Rust 算,JS 純讀。 */
  isInvertible: boolean;
}

/** 解 Ax = w「哪個輸入到得了 w」的結局(值域覆蓋頁)。 */
export interface SolveResult {
  /** 三種結局之一(編碼沿 EliminationTrace 的 solution_kind,共用對照表)。 */
  kind: SolutionKind;
  /** 唯一解時的輸入 x(滿足 T(x) = w);Infinite / Inconsistent 為 null。 */
  x: [number, number] | null;
}

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
  /**
   * 幾何規則的標準矩陣:core 的 `standard_matrix` 對規則做 e₁、e₂ 取樣,
   * 回傳 row-major `[a, b, c, d]`(Theorem 2.9:A 的第 j 行 = T(eⱼ))。
   * 矩陣是取樣「發現」的,不是寫死的字面值。
   */
  sampleStandardMatrix: (rule: RuleKind, param: number) => Float64Array;
  /**
   * 幾何規則直接施作在點 `(x, y)`(「規則路徑」;「矩陣路徑」用
   * `transformPoint` 左乘取樣矩陣 —— 兩條路會合即 Theorem 2.9)。
   * `param`:rotate 收弧度、shearX / scale 收 k,其餘忽略。
   */
  applyRule: (
    rule: RuleKind,
    param: number,
    x: number,
    y: number,
  ) => Float64Array;
  /**
   * Range(T_A) 的基底(core 的 `range_basis`,行向量攤平串接):長度 0 / 2 / 4
   * 分別代表值域 = {0} / 直線 / ℝ² —— 支數 = rank,長度就把維度說完了。
   */
  rangeBasis: (a: number, b: number, c: number, d: number) => Float64Array;
  /** w ∈ Range(T_A)?(core 的 `range_contains`:w 可達 ⟺ Ax = w 相容)。 */
  rangeContains: (
    a: number,
    b: number,
    c: number,
    d: number,
    wx: number,
    wy: number,
  ) => boolean;
  /** T_A 映成嗎?(core 的 `is_onto`,Theorem 2.10:rank = m)。 */
  isOnto: (a: number, b: number, c: number, d: number) => boolean;
  /**
   * 不可達向量的見證(core 的 `unreachable_vector`,標準基底掃描):
   * 不映成 → `[x, y]`(某支 eᵢ);映成 → 空陣列(= None)。
   */
  unreachableVector: (
    a: number,
    b: number,
    c: number,
    d: number,
  ) => Float64Array;
  /** 解 Ax = w:「哪個輸入到得了 w」—— 唯一解時連輸入 x 一起交出來。 */
  solveForInput: (
    a: number,
    b: number,
    c: number,
    d: number,
    wx: number,
    wy: number,
  ) => SolveResult;
  /**
   * U ∘ T 的標準矩陣 = B·A(core 的 `compose`,T_B ∘ T_A = T_BA):
   * 參數序與讀序同向(先外層 U、再內層 T),回傳 row-major `[a, b, c, d]`。
   */
  composeMatrix: (
    ua: number,
    ub: number,
    uc: number,
    ud: number,
    ta: number,
    tb: number,
    tc: number,
    td: number,
  ) => Float64Array;
  /**
   * T⁻¹ 的標準矩陣 = A⁻¹(core 的 `inverse`,Theorem 2.13):
   * 可逆 → row-major 四元素;不可逆 → null(邊界編碼:空陣列 = 無逆轉換)。
   */
  inverseMatrix: (
    a: number,
    b: number,
    c: number,
    d: number,
  ) => [number, number, number, number] | null;
  /** T_A 一對一嗎?(core 的 `is_one_to_one`,Theorem 2.11:rank = n)。 */
  isOneToOne: (a: number, b: number, c: number, d: number) => boolean;
  /** Summary Table 三燈(core 的 `report` 一次算好,JS 純讀)。 */
  transformationReport: (
    a: number,
    b: number,
    c: number,
    d: number,
  ) => TransformationReportJS;
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
    sampleStandardMatrix: (rule, param) =>
      sample_standard_matrix(RULE_CODES[rule], param),
    applyRule: (rule, param, x, y) =>
      apply_rule(RULE_CODES[rule], param, x, y),
    rangeBasis: range_basis,
    rangeContains: range_contains,
    isOnto: is_onto,
    unreachableVector: unreachable_vector,
    // [kind, x, y] → 物件:kind 走共用對照表;x 只在 Unique(1)時有意義。
    solveForInput: (a, b, c, d, wx, wy) => {
      const out = solve_for_input(a, b, c, d, wx, wy);
      return {
        kind: SOLUTION_NAMES[out[0]],
        x: out[0] === 1 ? [out[1], out[2]] : null,
      };
    },
    composeMatrix: compose_matrix,
    // 空陣列 = 無逆轉換(邊界編碼)→ null;有值則縫成定長 tuple。
    inverseMatrix: (a, b, c, d) => {
      const out = inverse_matrix(a, b, c, d);
      return out.length === 4 ? [out[0], out[1], out[2], out[3]] : null;
    },
    isOneToOne: is_one_to_one,
    // [1-1, onto, invertible] 的 0/1 → 具名三燈(合取已在 Rust 算完)。
    transformationReport: (a, b, c, d) => {
      const lights = transformation_report(a, b, c, d);
      return {
        isOneToOne: lights[0] === 1,
        isOnto: lights[1] === 1,
        isInvertible: lights[2] === 1,
      };
    },
  }));
  return instance;
}
