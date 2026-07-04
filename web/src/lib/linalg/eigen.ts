// 特徵值與特徵向量視覺化(/eigenvalues)的 binding —— 鏡像 src/wasm/eigen.rs。
import {
  characteristic_matrix_2d,
  characteristic_matrix_n,
  eigen_reports_n,
  eigenspace_basis_2d,
  eigenspace_basis_n,
  has_real_eigenvalues_2x2,
  matrix_trace,
  triangular_eigenvalues_n,
} from "../wasm/linear_algebra_101.js";

/** 單一特徵值的重數報告(8-3 的 EigenReport,plain-JS)。 */
export interface EigenReportJS {
  /** 特徵值 λ。 */
  value: number;
  /** 代數重數:λ 在特徵多項式根列表中的出現次數。 */
  algebraic: number;
  /** 幾何重數:dim Eλ = nullity(A − λI)。 */
  geometric: number;
  /** 虧損:幾何 < 代數 —— λ 交不出足額的獨立特徵向量(不可對角化的元凶)。 */
  isDeficient: boolean;
}

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
  /** `characteristicMatrix2d` 的 n×n 通用版(計算機):回 row-major flatten 的 A − λI。 */
  characteristicMatrixN: (a: number[], n: number, lambda: number) => Float64Array;
  /** `eigenspaceBasis2d` 的 n×n 通用版(計算機):支數 = 總長 ÷ n = 幾何重數。 */
  eigenspaceBasisN: (a: number[], n: number, lambda: number, epsilon: number) => Float64Array;
  /** 跡 tr(A) = Σ aᵢᵢ(core 的 `Matrix::trace`,8-2)。 */
  matrixTrace: (a: number[], n: number) => number;
  /**
   * 三角矩陣特徵值直讀(core 的 `triangular_eigenvalues`,8-2):上/下三角回主對角線
   * (保留重複 = 代數重數);非三角回空陣列(直讀失效,一般矩陣要走特徵方程)。
   */
  triangularEigenvaluesN: (a: number[], n: number, epsilon: number) => Float64Array;
  /**
   * 特徵值重數報告(core 的 `analyze_eigen_properties`,8-3):每個相異 λ 一份,
   * 代數 / 幾何重數並排、虧損一眼可判(Theorem 5.1:幾何 ≤ 代數)。
   */
  eigenReportsN: (
    a: number[],
    n: number,
    eigenvalues: number[],
    epsilon: number,
  ) => EigenReportJS[];
}

export const eigenOps: EigenOps = {
  characteristicMatrix2d: (a, lambda) =>
    characteristic_matrix_2d(Float64Array.from(a), lambda),
  eigenspaceBasis2d: (a, lambda, epsilon) =>
    eigenspace_basis_2d(Float64Array.from(a), lambda, epsilon),
  hasRealEigenvalues2x2: (a) => has_real_eigenvalues_2x2(Float64Array.from(a)),
  characteristicMatrixN: (a, n, lambda) =>
    characteristic_matrix_n(Float64Array.from(a), n, lambda),
  eigenspaceBasisN: (a, n, lambda, epsilon) =>
    eigenspace_basis_n(Float64Array.from(a), n, lambda, epsilon),
  matrixTrace: (a, n) => matrix_trace(Float64Array.from(a), n),
  triangularEigenvaluesN: (a, n, epsilon) =>
    triangular_eigenvalues_n(Float64Array.from(a), n, epsilon),
  // 定寬 4-tuple 列([λ, 代數, 幾何, 虧損 0/1])縫回具名物件。
  eigenReportsN: (a, n, eigenvalues, epsilon) => {
    const flat = eigen_reports_n(
      Float64Array.from(a),
      n,
      Float64Array.from(eigenvalues),
      epsilon,
    );
    const reports: EigenReportJS[] = [];
    for (let i = 0; i < flat.length; i += 4) {
      reports.push({
        value: flat[i],
        algebraic: flat[i + 1],
        geometric: flat[i + 2],
        isDeficient: flat[i + 3] === 1,
      });
    }
    return reports;
  },
};
