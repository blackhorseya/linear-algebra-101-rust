// 合成與可逆性視覺化(/composition)的 binding —— 鏡像 src/wasm/composition.rs。
import {
  compose_matrix,
  inverse_matrix,
  is_one_to_one,
  transformation_report,
} from "../wasm/linear_algebra_101.js";

/** 可逆性綜合判定表(合成與可逆性頁):core 的 `report` 三燈,縫回具名物件。 */
export interface TransformationReportJS {
  /** 一對一(Theorem 2.11:rank = n)。 */
  isOneToOne: boolean;
  /** 映成(Theorem 2.10:rank = m)。 */
  isOnto: boolean;
  /** 可逆(Theorem 2.12:1-1 且 onto)—— 合取在 Rust 算,JS 純讀。 */
  isInvertible: boolean;
}

/** composition 章的運算(攤平進 `Linalg`)。 */
export interface CompositionOps {
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

export const compositionOps: CompositionOps = {
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
};
