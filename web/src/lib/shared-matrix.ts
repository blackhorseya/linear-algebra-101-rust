import type { Matrix2x2 } from './canvas'

// 跨頁共享矩陣:URL 的 ?m= 編解碼與投影工具。
//
// 定位:URL 是「持久化投影」,各頁互動的真相仍是本地 state(見
// use-shared-matrix.ts 的單向資料流)。共享矩陣的正規型別是 2..4 的
// number[][] 方陣;固定 2×2 的頁面讀寫它的左上子塊(toMatrix2x2 /
// patchTopLeft),語意與 grid.ts 的 resizeGrid「保留左上角」一致。

/** 共享矩陣允許的方陣尺寸範圍(與 grid.ts 的 Size 2|3|4 對齊)。 */
const MIN_N = 2
const MAX_N = 4

/**
 * 編碼成 URL 安全字串:cell 用 `_`、row 用 `~` 分隔,如 `2_1~1_1`。
 * 兩者皆為 RFC 3986 unreserved 字元,router 不會轉義(選逗號分號會變 %2C/%3B)。
 * 數值捨入到 4 位小數 —— 拖曳產生的長浮點不進 URL,微小 drift 可接受。
 */
export function encodeMatrix(g: number[][]): string {
  return g
    .map((row) => row.map((v) => String(Math.round(v * 1e4) / 1e4)).join('_'))
    .join('~')
}

/** 解碼 + 驗證:必須是 2..4 的方陣、全部有限數;任何壞值一律回 null。 */
export function decodeMatrix(s: string): number[][] | null {
  const rows = s
    .split('~')
    .map((row) => row.split('_').map((cell) => (cell === '' ? NaN : Number(cell))))
  const n = rows.length
  if (n < MIN_N || n > MAX_N) return null
  if (!rows.every((row) => row.length === n && row.every(Number.isFinite))) {
    return null
  }
  return rows
}

/** 取左上 2×2 子塊給固定 2×2 的頁面當初始值;共享不存在時回 null。 */
export function toMatrix2x2(g: number[][] | null): Matrix2x2 | null {
  if (!g) return null
  return { a: g[0][0], b: g[0][1], c: g[1][0], d: g[1][1] }
}

/**
 * 2×2 頁寫回共享:patch 左上子塊、保留其餘 —— 在 /determinant 設好的
 * 3×3 外圈不會被 2×2 頁的編輯毀掉。g 為 null 時直接回純 2×2。
 */
export function patchTopLeft(g: number[][] | null, m: Matrix2x2): number[][] {
  const top: number[][] = [
    [m.a, m.b],
    [m.c, m.d],
  ]
  if (!g) return top
  return g.map((row, r) => row.map((v, c) => (r < 2 && c < 2 ? top[r][c] : v)))
}
