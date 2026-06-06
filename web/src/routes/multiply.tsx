import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { loadLinalg } from '../lib/linalg'
import { Status } from '../components/ui'
import { fmt } from '../lib/format'

export const Route = createFileRoute('/multiply')({
  component: MultiplyDemo,
})

/** 尺寸上限:4×4 已足夠展示一般性,展開式也還排得下。 */
const MAX_DIM = 4
const MIN_DIM = 1

// 預設:經典 (2×3)·(3×2) → 2×2,一開頁就展示「不只是方陣」的一般尺寸。
const DEFAULT_A: number[][] = [
  [1, 2, 3],
  [4, 5, 6],
]
const DEFAULT_B: number[][] = [
  [7, 8],
  [9, 10],
  [11, 12],
]

// 高亮色票(行內 class):A 的列 = violet、B 的欄 = cyan、C 的選中格 = emerald,
// 與 /span 的 v(紫)/ w(青)同一套對比配色。
const A_HIGHLIGHT = 'border-violet-500 bg-violet-950/50'
const B_HIGHLIGHT = 'border-cyan-500 bg-cyan-950/50'

/** 把 grid 重設為 rows×cols:保留交集內的舊值,新格補 0。 */
function resizeGrid(g: number[][], rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => g[r]?.[c] ?? 0),
  )
}

/** 乘積因子顯示:負數加括號(−2)·7,照課本的寫法。 */
function fmtFactor(v: number): string {
  return v < 0 ? `(${fmt(v)})` : fmt(v)
}

function MultiplyDemo() {
  // WASM 載入交給 Query 管;與其他頁共用同一 query key + staleTime: Infinity。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:只存兩個 grid,尺寸從資料導出(沿 core「維度從資料導出,不另存」)。
  const [aGrid, setAGrid] = useState<number[][]>(DEFAULT_A)
  const [bGrid, setBGrid] = useState<number[][]>(DEFAULT_B)
  // 選中 C 的哪一格(展開它的 dot product)。尺寸縮小時在下方 clamp,不另作重設。
  const [sel, setSel] = useState({ i: 0, j: 0 })

  const aRows = aGrid.length
  const aCols = aGrid[0]?.length ?? 0
  const bRows = bGrid.length
  const bCols = bGrid[0]?.length ?? 0

  // 維度相容性由 core 的 can_multiply 判定(同步呼叫很便宜,直接在 render 算)。
  const compatible = linalg
    ? linalg.canMultiply(aRows, aCols, bRows, bCols)
    : false

  // C 與展開項:相容才呼叫(multiplyExpand 假設前置條件成立)。grid 一變就重算。
  const expansion = useMemo(
    () =>
      linalg && compatible
        ? linalg.multiplyExpand(aGrid.flat(), aRows, aCols, bGrid.flat(), bRows, bCols)
        : null,
    [linalg, compatible, aGrid, bGrid, aRows, aCols, bRows, bCols],
  )

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  // 尺寸縮小時把選中格 clamp 回範圍內(沿 /elimination 的 safeStep 模式)。
  const selI = Math.min(sel.i, aRows - 1)
  const selJ = Math.min(sel.j, bCols - 1)

  const setCell =
    (setGrid: typeof setAGrid) => (r: number, c: number, value: number) =>
      setGrid((g) =>
        g.map((row, ri) =>
          ri === r ? row.map((v, ci) => (ci === c ? value : v)) : row,
        ),
      )
  // 用小整數(−9…9)隨機重填兩個矩陣:數字小,展開式才讀得動。
  const randomize = () => {
    const rand = () => Math.floor(Math.random() * 19) - 9
    setAGrid((g) => g.map((row) => row.map(rand)))
    setBGrid((g) => g.map((row) => row.map(rand)))
  }

  const terms = expansion?.terms[selI]?.[selJ] ?? []

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          矩陣乘法(Row × Column)
        </h1>
        <p className="text-sm text-slate-400">
          C = A·B 的每一格 c<sub>ij</sub>,是 A 的第 i <span className="text-violet-400">列</span>與
          B 的第 j <span className="text-cyan-400">欄</span>的點積(dot product)。維度相容性
          (can_multiply)與每一格的展開項全由 Rust(WASM)計算,JS 只負責排版與互動。
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-slate-400">
        <DimSteppers
          name="A"
          rows={aRows}
          cols={aCols}
          onResize={(r, c) => setAGrid((g) => resizeGrid(g, r, c))}
        />
        <DimSteppers
          name="B"
          rows={bRows}
          cols={bCols}
          onResize={(r, c) => setBGrid((g) => resizeGrid(g, r, c))}
        />
        <button
          onClick={randomize}
          className="rounded border border-slate-700 px-3 py-1.5 text-sm text-slate-300 transition hover:border-violet-500 hover:text-violet-300"
        >
          🎲 隨機填入
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-6">
        <MatrixBlock label={`A(${aRows}×${aCols})`}>
          <EditableMatrix
            grid={aGrid}
            onCell={setCell(setAGrid)}
            highlightClass={(r) =>
              compatible && r === selI ? A_HIGHLIGHT : ''
            }
          />
        </MatrixBlock>

        <span className="pt-5 text-xl text-slate-500">·</span>

        <MatrixBlock label={`B(${bRows}×${bCols})`}>
          <EditableMatrix
            grid={bGrid}
            onCell={setCell(setBGrid)}
            highlightClass={(_r, c) =>
              compatible && c === selJ ? B_HIGHLIGHT : ''
            }
          />
        </MatrixBlock>

        <span className="pt-5 text-xl text-slate-500">=</span>

        {expansion ? (
          <MatrixBlock label={`C(${expansion.rows}×${expansion.cols})`}>
            <ResultMatrix
              c={expansion.c}
              selI={selI}
              selJ={selJ}
              onSelect={(i, j) => setSel({ i, j })}
            />
          </MatrixBlock>
        ) : (
          <div className="max-w-sm space-y-1 rounded-lg border border-rose-900/60 bg-rose-950/30 p-4">
            <p className="font-medium text-rose-300">
              ({aRows}×{aCols})·({bRows}×{bCols}) 不能相乘
            </p>
            <p className="text-sm text-rose-200/70">
              A 的欄數 {aCols} ≠ B 的列數 {bRows}:row × col 點積需要等長的一列與一欄,
              內維不相等就對不齊。
            </p>
            <p className="font-mono text-xs text-rose-200/50">
              can_multiply = false(由 core 判定)
            </p>
          </div>
        )}
      </div>

      {expansion && (
        <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
          <p className="text-sm text-slate-400">
            點 C 的任一格展開它的點積:c
            <sub>
              {selI + 1},{selJ + 1}
            </sub>{' '}
            = A 第 {selI + 1} <span className="text-violet-400">列</span> · B 第{' '}
            {selJ + 1} <span className="text-cyan-400">欄</span>
          </p>
          <div className="space-y-1 overflow-x-auto font-mono text-lg whitespace-nowrap">
            <p>
              <span className="text-slate-500">= </span>
              {aGrid[selI].map((av, k) => (
                <span key={k}>
                  {k > 0 && <span className="text-slate-500"> + </span>}
                  <span className="text-violet-300">{fmtFactor(av)}</span>
                  <span className="text-slate-500">·</span>
                  <span className="text-cyan-300">{fmtFactor(bGrid[k][selJ])}</span>
                </span>
              ))}
            </p>
            {/* 內維 1 時乘積行與上一行重複,略過。 */}
            {terms.length > 1 && (
              <p>
                <span className="text-slate-500">= </span>
                {terms.map((t, k) => (
                  <span key={k}>
                    {k > 0 && <span className="text-slate-500"> + </span>}
                    <span className="text-emerald-300">{fmtFactor(t)}</span>
                  </span>
                ))}
              </p>
            )}
            <p>
              <span className="text-slate-500">= </span>
              <span className="font-semibold text-slate-50">
                {fmt(expansion.c[selI][selJ])}
              </span>
            </p>
          </div>
        </div>
      )}

      <div className="space-y-2 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <h2 className="text-sm font-medium text-slate-300">維度法則</h2>
        <p className="font-mono text-slate-100">
          (m×<span className="text-amber-400">n</span>)·(
          <span className="text-amber-400">n</span>×p) → m×p
        </p>
        <p className="text-xs text-slate-500">
          內維 <span className="text-amber-400">n</span>(A 的欄數 = B 的列數)必須相等,
          一列與一欄的點積才有對齊的長度;它在結果中「消掉」,外維 m、p 留下成為 C 的形狀。
          試試把 B 的列數加減一格,讓內維不合,看看會發生什麼。
        </p>
      </div>
    </section>
  )
}

/** 矩陣區塊:標題 + 內容(置中對齊用)。 */
function MatrixBlock({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <p className="font-mono text-sm text-slate-400">{label}</p>
      {children}
    </div>
  )
}

/** 可編輯的輸入網格;`highlightClass` 依格座標決定高亮樣式(選中列 / 欄)。 */
function EditableMatrix({
  grid,
  onCell,
  highlightClass,
}: {
  grid: number[][]
  onCell: (r: number, c: number, value: number) => void
  highlightClass: (r: number, c: number) => string
}) {
  const cols = grid[0]?.length ?? 0
  return (
    <div
      className="inline-grid gap-1.5"
      style={{ gridTemplateColumns: `repeat(${cols}, 3.5rem)` }}
    >
      {grid.map((row, r) =>
        row.map((val, c) => (
          <input
            key={`${r}-${c}`}
            type="number"
            step="any"
            value={val}
            onChange={(e) => onCell(r, c, Number(e.target.value))}
            className={[
              'h-10 w-14 rounded border bg-slate-900 px-1 text-center font-mono text-sm text-slate-100 transition-colors focus:border-violet-500 focus:outline-none',
              highlightClass(r, c) || 'border-slate-700',
            ].join(' ')}
          />
        )),
      )}
    </div>
  )
}

/** 結果矩陣 C:每格是按鈕,點了展開該格的點積。 */
function ResultMatrix({
  c,
  selI,
  selJ,
  onSelect,
}: {
  c: number[][]
  selI: number
  selJ: number
  onSelect: (i: number, j: number) => void
}) {
  const cols = c[0]?.length ?? 0
  return (
    <div
      className="inline-grid gap-1.5"
      style={{ gridTemplateColumns: `repeat(${cols}, 3.5rem)` }}
    >
      {c.map((row, i) =>
        row.map((val, j) => {
          const selected = i === selI && j === selJ
          return (
            <button
              key={`${i}-${j}`}
              onClick={() => onSelect(i, j)}
              className={[
                'flex h-10 w-14 items-center justify-center rounded font-mono text-sm tabular-nums transition-colors',
                selected
                  ? 'bg-emerald-600 text-white ring-2 ring-emerald-300'
                  : 'bg-slate-800 text-slate-200 hover:bg-slate-700',
              ].join(' ')}
            >
              {fmt(val)}
            </button>
          )
        }),
      )}
    </div>
  )
}

/** 單一矩陣的列 / 欄 stepper(1…MAX_DIM)。 */
function DimSteppers({
  name,
  rows,
  cols,
  onResize,
}: {
  name: string
  rows: number
  cols: number
  onResize: (rows: number, cols: number) => void
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="font-mono text-slate-200">{name}</span>
      列 <span className="font-mono text-slate-200">{rows}</span>
      <StepperButton
        disabled={rows <= MIN_DIM}
        onClick={() => onResize(rows - 1, cols)}
      >
        −
      </StepperButton>
      <StepperButton
        disabled={rows >= MAX_DIM}
        onClick={() => onResize(rows + 1, cols)}
      >
        +
      </StepperButton>
      欄 <span className="font-mono text-slate-200">{cols}</span>
      <StepperButton
        disabled={cols <= MIN_DIM}
        onClick={() => onResize(rows, cols - 1)}
      >
        −
      </StepperButton>
      <StepperButton
        disabled={cols >= MAX_DIM}
        onClick={() => onResize(rows, cols + 1)}
      >
        +
      </StepperButton>
    </span>
  )
}

function StepperButton({
  disabled,
  onClick,
  children,
}: {
  disabled: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className="flex h-6 w-6 items-center justify-center rounded border border-slate-700 text-slate-300 transition hover:border-violet-500 hover:text-violet-300 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:border-slate-700 disabled:hover:text-slate-300"
    >
      {children}
    </button>
  )
}
