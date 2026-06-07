// 標準矩陣取樣視覺化(/standard-matrix)的 binding —— 鏡像 src/wasm/standard_matrix.rs。
import {
  apply_rule,
  sample_standard_matrix,
} from "../wasm/linear_algebra_101.js";

/** 幾何規則(標準矩陣取樣頁)。具名字串對前端友善,過邊界前再映射成 u8。 */
export type RuleKind =
  | "rotate"
  | "reflectX"
  | "reflectDiag"
  | "shearX"
  | "projectX"
  | "scale";

// 規則名 → WASM 的 u8 編碼(數值必須與 standard_matrix.rs 的 RULE_* 一致)。
const RULE_CODES: Record<RuleKind, number> = {
  rotate: 0,
  reflectX: 1,
  reflectDiag: 2,
  shearX: 3,
  projectX: 4,
  scale: 5,
};

/** standard_matrix 章的運算(攤平進 `Linalg`)。 */
export interface StandardMatrixOps {
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
}

export const standardMatrixOps: StandardMatrixOps = {
  sampleStandardMatrix: (rule, param) =>
    sample_standard_matrix(RULE_CODES[rule], param),
  applyRule: (rule, param, x, y) => apply_rule(RULE_CODES[rule], param, x, y),
};
