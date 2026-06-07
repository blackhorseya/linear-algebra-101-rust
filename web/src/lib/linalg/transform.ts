// 線性變換視覺化(/transform)的 binding —— 鏡像 src/wasm/transform.rs。
import { are_parallel, transform_point } from "../wasm/linear_algebra_101.js";

/** transform 章的運算(攤平進 `Linalg`)。 */
export interface TransformOps {
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
}

export const transformOps: TransformOps = {
  transformPoint: transform_point,
  areParallel: are_parallel,
};
