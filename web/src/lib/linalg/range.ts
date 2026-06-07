// 值域與映成視覺化(/range)的 binding —— 鏡像 src/wasm/range.rs。
import {
  is_onto,
  range_basis,
  range_contains,
  solve_for_input,
  unreachable_vector,
} from "../wasm/linear_algebra_101.js";
import { SOLUTION_NAMES, type SolutionKind } from "./elimination";

/** 解 Ax = w「哪個輸入到得了 w」的結局(值域覆蓋頁)。 */
export interface SolveResult {
  /** 三種結局之一(編碼沿 EliminationTrace 的 solution_kind,共用對照表)。 */
  kind: SolutionKind;
  /** 唯一解時的輸入 x(滿足 T(x) = w);Infinite / Inconsistent 為 null。 */
  x: [number, number] | null;
}

/** range 章的運算(攤平進 `Linalg`)。 */
export interface RangeOps {
  /**
   * Range(T_A) 的基底(core 的 `range_basis`,行向量攤平串接):長度 0 / 2 / 4
   * 分別代表值域 = {0} / 直線 / ℝ² —— 支數 = rank,長度就把維度說完了。
   */
  rangeBasis: (a: number, b: number, c: number, d: number) => Float64Array;
  /** w ∈ Range(T_A)?(core 的 `range_contains`:w 可達 ⟺ Ax = w 相容)。 */
  rangeContains: (
    a: number,
    b: number,
    c: number,
    d: number,
    wx: number,
    wy: number,
  ) => boolean;
  /** T_A 映成嗎?(core 的 `is_onto`,Theorem 2.10:rank = m)。 */
  isOnto: (a: number, b: number, c: number, d: number) => boolean;
  /**
   * 不可達向量的見證(core 的 `unreachable_vector`,標準基底掃描):
   * 不映成 → `[x, y]`(某支 eᵢ);映成 → 空陣列(= None)。
   */
  unreachableVector: (
    a: number,
    b: number,
    c: number,
    d: number,
  ) => Float64Array;
  /** 解 Ax = w:「哪個輸入到得了 w」—— 唯一解時連輸入 x 一起交出來。 */
  solveForInput: (
    a: number,
    b: number,
    c: number,
    d: number,
    wx: number,
    wy: number,
  ) => SolveResult;
}

export const rangeOps: RangeOps = {
  rangeBasis: range_basis,
  rangeContains: range_contains,
  isOnto: is_onto,
  unreachableVector: unreachable_vector,
  // [kind, x, y] → 物件:kind 走共用對照表;x 只在 Unique(1)時有意義。
  solveForInput: (a, b, c, d, wx, wy) => {
    const out = solve_for_input(a, b, c, d, wx, wy);
    return {
      kind: SOLUTION_NAMES[out[0]],
      x: out[0] === 1 ? [out[1], out[2]] : null,
    };
  },
};
