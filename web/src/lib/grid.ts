// n×n 方陣輸入的共用邏輯(UI 元件在 components/MatrixGrid.tsx;
// 拆兩檔是 react-refresh 的要求:元件檔只能 export 元件)。

/** 視覺化頁支援的方陣尺寸。 */
export type Size = 2 | 3 | 4

/** 切換尺寸:保留共有左上角;放大時向 Iₙ 靠攏(新對角補 1、其餘補 0),預設仍可逆。 */
export function resizeGrid(g: number[][], next: Size): number[][] {
  return Array.from({ length: next }, (_, r) =>
    Array.from({ length: next }, (_, c) => g[r]?.[c] ?? (r === c ? 1 : 0)),
  )
}
