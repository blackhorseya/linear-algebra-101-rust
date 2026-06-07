// 跨章共用的 SoA 縫合工具 —— 鏡像 src/wasm/helpers.rs 的角色:只給各章的
// run* 縫合函數用,不經 index.ts re-export(呼叫端看不到)。
//
// SoA(Structure-of-Arrays)的好處:整趟 trace 只跨 WASM 邊界少數幾次(每個
// 欄位一條 typed array),而非「每個 step 一個帶指標的 JS 物件」。各章從串接
// 的 typed array 切片 reshape,縫回 plain-JS 後立刻 `free()` WASM 物件。

/** 從串接的 SoA 快照切出第 i 步,reshape 成 rows×cols 的巢狀陣列。 */
export function sliceMatrix(
  snaps: Float64Array,
  i: number,
  rows: number,
  cols: number,
): number[][] {
  const base = i * rows * cols;
  const matrix: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const start = base + r * cols;
    matrix.push(Array.from(snaps.subarray(start, start + cols)));
  }
  return matrix;
}

/** CSR 切片:第 i 步的元素 = flat[offsets[i] .. offsets[i+1]]。 */
export function csrRow(
  flat: Uint32Array,
  offsets: Uint32Array,
  i: number,
): number[] {
  return Array.from(flat.subarray(offsets[i], offsets[i + 1]));
}
