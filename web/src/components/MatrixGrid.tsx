// n×n 方陣輸入的共用元件:尺寸切換 + 可編輯網格。
// 原為 /invertibility 頁內元件,/determinant 也用到即升格至此
// (沿 wasm 端「跨章即升格進 helpers」的慣例);resize 邏輯在 lib/grid.ts
// (react-refresh 要求元件檔只 export 元件)。

import type { Size } from '../lib/grid'

export function SizeToggle({
  n,
  onChange,
}: {
  n: Size
  onChange: (n: Size) => void
}) {
  return (
    <div className="inline-flex rounded-lg border border-slate-700 p-0.5">
      {([2, 3, 4] as const).map((s) => (
        <button
          key={s}
          onClick={() => onChange(s)}
          className={[
            'rounded-md px-3 py-1.5 text-sm transition',
            n === s
              ? 'bg-violet-600 text-white'
              : 'text-slate-400 hover:text-slate-100',
          ].join(' ')}
        >
          {s}×{s}
        </button>
      ))}
    </div>
  )
}

/** 可編輯的 n×n 輸入網格(綁定原始 grid state)。 */
export function EditableGrid({
  grid,
  onCell,
}: {
  grid: number[][]
  onCell: (r: number, c: number, value: number) => void
}) {
  const cols = grid[0]?.length ?? 0
  return (
    <div
      className="inline-grid gap-1.5"
      style={{ gridTemplateColumns: `repeat(${cols}, 5rem)` }}
    >
      {grid.map((row, r) =>
        row.map((val, c) => (
          <input
            key={`${r}-${c}`}
            type="number"
            step="any"
            value={val}
            onChange={(e) => onCell(r, c, Number(e.target.value))}
            className="h-10 w-20 rounded border border-slate-700 bg-slate-900 px-2 text-center font-mono text-sm text-slate-100 focus:border-violet-500 focus:outline-none"
          />
        )),
      )}
    </div>
  )
}
