// 子空間視覺化的 binding —— 鏡像 src/wasm/subspace.rs。服務 /nullspace(Null A)
// 與 /rank(Row A / Col A 維度對偶)兩頁。
import {
  null_space_basis_n,
  null_space_contains,
  nullity,
  rank,
  rank_transpose,
  row_space_basis,
  row_space_basis_n,
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
  /**
   * Row A 的基底(core 的 `row_space_basis`,Theorem 4.8:**RREF 的非零列**),
   * 列向量攤平串接:長度 0 / 2 / 4 = dim Row A = 0 / 1 / 2。與 `rangeBasis`
   * (Col A 基底)對偶 —— /rank 頁並排畫 Row A(domain)與 Col A(codomain)。
   */
  rowSpaceBasis: (a: number, b: number, c: number, d: number) => Float64Array;
  /**
   * dim Row A,經 rank(Aᵀ) 獨立算出。與 `rank`(= dim Col A)**恆相等** ——
   * 即定理 rank(A) = rank(Aᵀ);前端把兩個獨立計算的數並列當場對帳。
   */
  rankTranspose: (a: number, b: number, c: number, d: number) => number;
  /** `rowSpaceBasis` 的 n×n 通用版(計算機):支數 = 總長 ÷ n = dim Row A。 */
  rowSpaceBasisN: (data: number[], n: number) => Float64Array;
  /** Null A 基底的 n×n 版(core 的 `null_space_basis`):支數 = 總長 ÷ n = nullity。 */
  nullSpaceBasisN: (data: number[], n: number) => Float64Array;
}

export const subspaceOps: SubspaceOps = {
  nullSpaceContains: null_space_contains,
  nullity,
  rank,
  rowSpaceBasis: row_space_basis,
  rankTranspose: rank_transpose,
  rowSpaceBasisN: (data, n) => row_space_basis_n(Float64Array.from(data), n),
  nullSpaceBasisN: (data, n) => null_space_basis_n(Float64Array.from(data), n),
};
