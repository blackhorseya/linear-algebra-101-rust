// 線性運算子的矩陣表示(/operator)的 binding —— 鏡像 src/wasm/operator.rs。
import { b_matrix_2d, b_matrix_n } from "../wasm/linear_algebra_101.js";

/** operator 章的運算(攤平進 `Linalg`)。 */
export interface OperatorOps {
  /**
   * 運算子 A(row-major 2×2 flatten `[a11, a12, a21, a22]`)相對於基底 B = {b₁, b₂}
   * 的矩陣 [T]_B(core 的 `b_matrix`):回 row-major `[t11, t12, t21, t22]`;b₁ ∥ b₂
   * (退化、不是 ℝ² 的基底)時回空陣列 —— [T]_B 未定義。
   */
  bMatrix2d: (
    a: number[],
    b1x: number,
    b1y: number,
    b2x: number,
    b2y: number,
  ) => Float64Array;
  /**
   * `bMatrix2d` 的 n×n 通用版(計算機):`basisData` 為 row-major n×n、其**行**為
   * 基底向量(沿 `coordinatesN`);回 row-major flatten 的 [T]_B,基底退化回空陣列。
   */
  bMatrixN: (a: number[], n: number, basisData: number[]) => Float64Array;
}

export const operatorOps: OperatorOps = {
  bMatrix2d: (a, b1x, b1y, b2x, b2y) =>
    b_matrix_2d(Float64Array.from(a), b1x, b1y, b2x, b2y),
  bMatrixN: (a, n, basisData) =>
    b_matrix_n(Float64Array.from(a), n, Float64Array.from(basisData)),
};
