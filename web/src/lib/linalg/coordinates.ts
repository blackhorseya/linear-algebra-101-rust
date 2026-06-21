// 座標系統視覺化(/coordinates)的 binding —— 鏡像 src/wasm/coordinates.rs。
import {
  coordinates_2d,
  from_coordinates_2d,
} from "../wasm/linear_algebra_101.js";

/** coordinates 章的運算(攤平進 `Linalg`)。 */
export interface CoordinatesOps {
  /**
   * x 在有序基底 B = {b₁, b₂} 下的座標 [x]_B = (c₁, c₂)(core 的 `coordinates`):
   * 回長度 2 的陣列;b₁ ∥ b₂(退化、不是 ℝ² 的基底)時回空陣列 —— 座標未定義。
   */
  coordinates2d: (
    b1x: number,
    b1y: number,
    b2x: number,
    b2y: number,
    px: number,
    py: number,
  ) => Float64Array;
  /** 由座標重建 x = c₁·b₁ + c₂·b₂(core 的 `from_coordinates`),回 `[x, y]`。 */
  fromCoordinates2d: (
    b1x: number,
    b1y: number,
    b2x: number,
    b2y: number,
    c1: number,
    c2: number,
  ) => Float64Array;
}

export const coordinatesOps: CoordinatesOps = {
  coordinates2d: coordinates_2d,
  fromCoordinates2d: from_coordinates_2d,
};
