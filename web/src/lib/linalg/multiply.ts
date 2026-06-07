// 矩陣乘法視覺化(/multiply)的 binding —— 鏡像 src/wasm/multiply.rs。
import { can_multiply, multiply_expand } from "../wasm/linear_algebra_101.js";

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

/** multiply 章的運算(攤平進 `Linalg`)。 */
export interface MultiplyOps {
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
}

export const multiplyOps: MultiplyOps = {
  canMultiply: can_multiply,
  multiplyExpand: runMultiplyExpand,
};
