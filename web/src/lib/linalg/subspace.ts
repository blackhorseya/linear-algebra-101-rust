// 零空間(Null space)視覺化(/nullspace)的 binding —— 鏡像 src/wasm/subspace.rs。
import {
  null_space_contains,
  nullity,
  rank,
} from "../wasm/linear_algebra_101.js";

/** subspace 章的運算(攤平進 `Linalg`)。 */
export interface SubspaceOps {
  /** v ∈ Null A?(core 的 `null_space_contains`:v ∈ Null A ⟺ Av ≈ 0)。 */
  nullSpaceContains: (
    a: number,
    b: number,
    c: number,
    d: number,
    vx: number,
    vy: number,
  ) => boolean;
  /**
   * Null A 的維度(nullity):被壓到原點的獨立輸入方向數。
   * 0 → 核 = {0}、1 → 核是一條過原點的線、2 → 整個 domain 被壓扁(A = 0)。
   */
  nullity: (a: number, b: number, c: number, d: number) => number;
  /** Col A 的維度(rank);與 nullity 滿足 rank + nullity = 2(domain 維度)。 */
  rank: (a: number, b: number, c: number, d: number) => number;
}

export const subspaceOps: SubspaceOps = {
  nullSpaceContains: null_space_contains,
  nullity,
  rank,
};
