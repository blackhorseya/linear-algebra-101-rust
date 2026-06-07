// 線性性質視覺化(/linearity)的 binding —— 鏡像 src/wasm/linearity.rs。
import {
  add_vectors,
  check_linearity,
  scale_vector,
} from "../wasm/linear_algebra_101.js";

/** linearity 章的運算(攤平進 `Linalg`)。 */
export interface LinearityOps {
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

export const linearityOps: LinearityOps = {
  addVectors: add_vectors,
  scaleVector: scale_vector,
  checkLinearity: check_linearity,
};
