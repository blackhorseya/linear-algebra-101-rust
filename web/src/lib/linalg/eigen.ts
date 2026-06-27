// 特徵值與特徵向量視覺化(/eigenvalues)的 binding —— 鏡像 src/wasm/eigen.rs。
import {
  characteristic_matrix_2d,
  eigenspace_basis_2d,
  has_real_eigenvalues_2x2,
} from "../wasm/linear_algebra_101.js";

/** eigen 章的運算(攤平進 `Linalg`)。 */
export interface EigenOps {
  /**
   * 特徵閘門矩陣 M = A − λI(core 的 `characteristic_matrix`):回 row-major
   * `[m11, m12, m21, m22]`。前端用它畫「M 把單位方塊送到的平行四邊形」並算 det(A−λI)。
   */
  characteristicMatrix2d: (a: number[], lambda: number) => Float64Array;
  /**
   * 特徵空間 Eλ = Null(A − λI) 的基底攤平(core 的 `eigenspace_basis` → null_space_basis):
   * `[]`(λ 非特徵值)/ `[vx, vy]`(一維)/ `[v1x, v1y, v2x, v2y]`(整個平面,純量矩陣)。
   * `epsilon` = 「λ 多接近特徵值,特徵向量才浮現」的吸附範圍。
   */
  eigenspaceBasis2d: (a: number[], lambda: number, epsilon: number) => Float64Array;
  /** 此 2×2 是否有實特徵值(core 的 `has_real_eigenvalues_2x2`):false = 純旋轉,永不塌。 */
  hasRealEigenvalues2x2: (a: number[]) => boolean;
}

export const eigenOps: EigenOps = {
  characteristicMatrix2d: (a, lambda) =>
    characteristic_matrix_2d(Float64Array.from(a), lambda),
  eigenspaceBasis2d: (a, lambda, epsilon) =>
    eigenspace_basis_2d(Float64Array.from(a), lambda, epsilon),
  hasRealEigenvalues2x2: (a) => has_real_eigenvalues_2x2(Float64Array.from(a)),
};
