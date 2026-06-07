// 行列式視覺化(/determinant)的 binding —— 鏡像 src/wasm/determinant.rs。
// `determinant` 隨 wasm 端搬家自 transform 章(一章一檔歸位)。
import { determinant } from "../wasm/linear_algebra_101.js";

/** determinant 章的運算(攤平進 `Linalg`)。 */
export interface DeterminantOps {
  /**
   * 2×2 矩陣 `[[a,b],[c,d]]` 的行列式(= 單位正方形像的有號面積;
   * 委派 core 的 Gaussian 版)。
   */
  determinant: (a: number, b: number, c: number, d: number) => number;
}

export const determinantOps: DeterminantOps = { determinant };
