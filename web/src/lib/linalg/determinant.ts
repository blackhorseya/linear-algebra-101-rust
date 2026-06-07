// 行列式視覺化(/determinant)的 binding —— 鏡像 src/wasm/determinant.rs。
// `determinant` 隨 wasm 端搬家自 transform 章(一章一檔歸位);頁面升級 n×n
// 後改收 row-major flatten + n(沿 invertTrace 的邊界慣例)。
import { determinant, is_invertible } from "../wasm/linear_algebra_101.js";

/** determinant 章的運算(攤平進 `Linalg`)。 */
export interface DeterminantOps {
  /**
   * n×n 矩陣(row-major flatten,長度 n·n)的行列式(= 單位 n 維方體像的
   * 有號體積;委派 core 的 Gaussian 版)。
   */
  determinant: (data: number[], n: number) => number;
  /**
   * n×n 矩陣可逆嗎?(rank 路)與 det 路是兩條**獨立**計算 —— /determinant
   * 頁拿兩者對帳,Theorem 3.4(a)(可逆 ⟺ det ≠ 0)每一幀上演。
   */
  isInvertible: (data: number[], n: number) => boolean;
}

export const determinantOps: DeterminantOps = {
  determinant: (data, n) => determinant(Float64Array.from(data), n),
  isInvertible: (data, n) => is_invertible(Float64Array.from(data), n),
};
