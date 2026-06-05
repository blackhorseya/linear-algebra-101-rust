import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { loadLinalg } from '../lib/linalg'
import type { ElimPhase, EliminationStepJS, SolutionKind } from '../lib/linalg'
import { PlaybackControls } from '../components/Playback'

export const Route = createFileRoute('/elimination')({
  component: EliminationDemo,
})

type Mode = 'system' | 'matrix'

// 預設:經典 3×3 唯一解系統(x=2, y=3, z=−1)。最後一欄是常數 b。
const DEFAULT_GRID: number[][] = [
  [2, 1, -1, 8],
  [-3, -1, 2, -11],
  [-2, 1, 2, -3],
]

const PHASE_LABEL: Record<ElimPhase, string> = {
  initial: '初始矩陣',
  forward: 'Forward — 消去 pivot 下方',
  backward: 'Backward — 正規化並消去上方',
}

const PHASE_STYLE: Record<ElimPhase, string> = {
  initial: 'bg-slate-700 text-slate-200',
  forward: 'bg-violet-600/80 text-violet-50',
  backward: 'bg-sky-600/80 text-sky-50',
}

const SOLUTION_LABEL: Record<SolutionKind, string> = {
  NA: '—',
  Unique: '唯一解',
  Infinite: '無限多解',
  Inconsistent: '無解(系統矛盾)',
}

const SOLUTION_STYLE: Record<SolutionKind, string> = {
  NA: 'text-slate-400',
  Unique: 'text-emerald-400',
  Infinite: 'text-amber-400',
  Inconsistent: 'text-rose-400',
}

const SOLUTION_HINT: Record<SolutionKind, string> = {
  NA: '',
  Unique: 'RREF 化成 [I | x],最後一欄即為解。',
  Infinite: '存在自由變數,解可由特解 + 零空間參數化。',
  Inconsistent: '出現「0 = 非零」的矛盾列,無任何解。',
}

function EliminationDemo() {
  // WASM 載入交給 Query 管;與 /transform 共用同一 query key + staleTime: Infinity,
  // 整個 app 只載入一次。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  const [mode, setMode] = useState<Mode>('system')
  const [grid, setGrid] = useState<number[][]>(DEFAULT_GRID)
  const [currentStep, setCurrentStep] = useState(0)

  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  // 方程組模式:最後一欄是常數 b → augCol = cols − 1;一般矩陣:無增廣欄 → −1。
  const augCol = mode === 'system' ? cols - 1 : -1

  // trace 計算便宜,直接同步算(同 /transform 不另包 Query)。grid / mode 一變就重算。
  const trace = useMemo(
    () => (linalg ? linalg.eliminate(grid.flat(), rows, cols, augCol) : null),
    [linalg, grid, rows, cols, augCol],
  )

  const stepCount = trace?.steps.length ?? 0

  // trace 換了(輸入或模式變)→ 回到第 0 步。用 React 認可的「render 期間比對前次值」
  // 模式,而非在 effect 裡同步 setState(那會觸發 cascading render)。
  const [prevTrace, setPrevTrace] = useState(trace)
  if (trace !== prevTrace) {
    setPrevTrace(trace)
    setCurrentStep(0)
  }

  // 鍵盤左右鍵逐步導航。
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'ArrowLeft') setCurrentStep((s) => Math.max(0, s - 1))
      else if (e.key === 'ArrowRight')
        setCurrentStep((s) => Math.min(stepCount - 1, s + 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [stepCount])

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg || !trace)
    return <Status>WASM 載入失敗:{String(error)}</Status>

  const safeStep = Math.min(currentStep, stepCount - 1)
  const step = trace.steps[safeStep]

  // --- 編輯輔助 ---
  const setCell = (r: number, c: number, value: number) =>
    setGrid((g) =>
      g.map((row, ri) =>
        ri === r ? row.map((v, ci) => (ci === c ? value : v)) : row,
      ),
    )
  const addRow = () => setGrid((g) => [...g, Array(g[0].length).fill(0)])
  const removeRow = () => setGrid((g) => (g.length > 1 ? g.slice(0, -1) : g))
  const addCol = () => setGrid((g) => g.map((row) => [...row, 0]))
  const removeCol = () => {
    const minCols = mode === 'system' ? 2 : 1
    setGrid((g) =>
      g[0].length > minCols ? g.map((row) => row.slice(0, -1)) : g,
    )
  }
  // 用整數(−20…20)隨機重填。維度不變。格子已加寬到容納兩位數;消去後的長小數
  // 若仍超出,矩陣區有 overflow-x-auto 可橫向捲動。
  const randomize = () =>
    setGrid((g) =>
      g.map((row) => row.map(() => Math.floor(Math.random() * 41) - 20)),
    )

  // --- 終態結構摘要(方程組模式只看係數矩陣 A 的部分,即 < augCol)---
  const isSystem = trace.augCol >= 0
  const pivotCols = isSystem
    ? trace.pivotColumns.filter((c) => c < trace.augCol)
    : trace.pivotColumns
  const freeCols = isSystem
    ? trace.freeColumns.filter((c) => c < trace.augCol)
    : trace.freeColumns

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          高斯消去法
        </h1>
        <p className="text-sm text-slate-400">
          逐步圖解 Gauss–Jordan 消去(化簡到 RREF)。每一步的矩陣與操作都由 Rust(WASM)
          計算,JS 只負責呈現。用 ← / → 鍵或下方按鈕逐步播放。
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <ModeToggle mode={mode} onChange={setMode} />
        <DimControls
          rows={rows}
          cols={cols}
          onAddRow={addRow}
          onRemoveRow={removeRow}
          onAddCol={addCol}
          onRemoveCol={removeCol}
        />
        <button
          onClick={randomize}
          className="rounded border border-slate-700 px-3 py-1.5 text-sm text-slate-300 transition hover:border-violet-500 hover:text-violet-300"
        >
          🎲 隨機填入
        </button>
      </div>

      <fieldset className="space-y-2">
        <legend className="mb-1 text-sm font-medium text-slate-300">
          {mode === 'system'
            ? '增廣矩陣 [A | b](最後一欄為常數 b)'
            : '矩陣'}
        </legend>
        <EditableGrid grid={grid} augCol={augCol} onCell={setCell} />
      </fieldset>

      <div className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <div className="flex items-center justify-between gap-4">
          <span
            className={`rounded px-2 py-1 text-xs font-medium ${PHASE_STYLE[step.phase]}`}
          >
            {PHASE_LABEL[step.phase]}
          </span>
          <span className="font-mono text-sm text-slate-400">
            第 {safeStep + 1} / {stepCount} 步
          </span>
        </div>

        <p className="text-center font-mono text-lg text-slate-100">
          {step.description}
        </p>

        <div className="flex justify-center overflow-x-auto py-2">
          <MatrixGrid step={step} augCol={trace.augCol} />
        </div>

        <PlaybackControls
          step={safeStep}
          count={stepCount}
          onChange={setCurrentStep}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <InfoCard title="矩陣結構">
          <Row label="rank(秩)">{trace.rank}</Row>
          <Row label={isSystem ? '基本變數' : 'pivot 行'}>
            {pivotCols.length
              ? pivotCols.map((c) => varName(c, isSystem)).join(', ')
              : '—'}
          </Row>
          <Row label={isSystem ? '自由變數' : 'free 行'}>
            {freeCols.length
              ? freeCols.map((c) => varName(c, isSystem)).join(', ')
              : '—'}
          </Row>
        </InfoCard>

        {isSystem && (
          <InfoCard title="解的型態">
            <p
              className={`text-lg font-semibold ${SOLUTION_STYLE[trace.solutionKind]}`}
            >
              {SOLUTION_LABEL[trace.solutionKind]}
            </p>
            <p className="text-xs text-slate-500">
              {SOLUTION_HINT[trace.solutionKind]}
            </p>
          </InfoCard>
        )}
      </div>
    </section>
  )
}

/** 唯讀地畫出某一步的矩陣,並高亮 pivot 交點 / 被改動的列 / 增廣欄分隔。 */
function MatrixGrid({
  step,
  augCol,
}: {
  step: EliminationStepJS
  augCol: number
}) {
  const { matrix, pivotRow, pivotCol, changedRows } = step
  const cols = matrix[0]?.length ?? 0
  const changed = new Set(changedRows)
  return (
    <div
      className="inline-grid gap-1.5"
      style={{ gridTemplateColumns: `repeat(${cols}, 5rem)` }}
    >
      {matrix.map((row, r) =>
        row.map((val, c) => {
          const isPivot = r === pivotRow && c === pivotCol
          const inChanged = changed.has(r)
          const isAug = augCol >= 0 && c === augCol
          return (
            <div
              key={`${r}-${c}`}
              className={[
                'flex h-12 w-20 items-center justify-center rounded font-mono text-sm tabular-nums transition-colors',
                isPivot
                  ? 'bg-violet-600 text-white ring-2 ring-violet-300'
                  : inChanged
                    ? 'bg-emerald-900/50 text-emerald-100'
                    : 'bg-slate-800 text-slate-200',
                isAug ? 'border-l-2 border-l-slate-500' : '',
              ].join(' ')}
            >
              {fmt(val)}
            </div>
          )
        }),
      )}
    </div>
  )
}

/** 可編輯的輸入網格(綁定原始 grid state)。 */
function EditableGrid({
  grid,
  augCol,
  onCell,
}: {
  grid: number[][]
  augCol: number
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
            className={[
              'h-10 w-20 rounded border bg-slate-900 px-2 text-center font-mono text-sm text-slate-100 focus:border-violet-500 focus:outline-none',
              augCol >= 0 && c === augCol
                ? 'border-slate-700 border-l-2 border-l-slate-500'
                : 'border-slate-700',
            ].join(' ')}
          />
        )),
      )}
    </div>
  )
}

function ModeToggle({
  mode,
  onChange,
}: {
  mode: Mode
  onChange: (m: Mode) => void
}) {
  return (
    <div className="inline-flex rounded-lg border border-slate-700 p-0.5">
      {(['system', 'matrix'] as const).map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={[
            'rounded-md px-3 py-1.5 text-sm transition',
            mode === m
              ? 'bg-violet-600 text-white'
              : 'text-slate-400 hover:text-slate-100',
          ].join(' ')}
        >
          {m === 'system' ? '方程組 [A|b]' : '一般矩陣'}
        </button>
      ))}
    </div>
  )
}

function DimControls({
  rows,
  cols,
  onAddRow,
  onRemoveRow,
  onAddCol,
  onRemoveCol,
}: {
  rows: number
  cols: number
  onAddRow: () => void
  onRemoveRow: () => void
  onAddCol: () => void
  onRemoveCol: () => void
}) {
  return (
    <div className="flex items-center gap-4 text-sm text-slate-400">
      <span className="flex items-center gap-1.5">
        列 <span className="font-mono text-slate-200">{rows}</span>
        <StepperButton onClick={onRemoveRow}>−</StepperButton>
        <StepperButton onClick={onAddRow}>+</StepperButton>
      </span>
      <span className="flex items-center gap-1.5">
        行 <span className="font-mono text-slate-200">{cols}</span>
        <StepperButton onClick={onRemoveCol}>−</StepperButton>
        <StepperButton onClick={onAddCol}>+</StepperButton>
      </span>
    </div>
  )
}

function StepperButton({
  onClick,
  children,
}: {
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className="flex h-6 w-6 items-center justify-center rounded border border-slate-700 text-slate-300 transition hover:border-violet-500 hover:text-violet-300"
    >
      {children}
    </button>
  )
}

function InfoCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-2 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
      <h2 className="text-sm font-medium text-slate-300">{title}</h2>
      {children}
    </div>
  )
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="font-mono text-slate-100">{children}</span>
    </div>
  )
}

function Status({ children }: { children: ReactNode }) {
  return <p className="text-slate-400">{children}</p>
}

/** 欄索引 → 顯示名:方程組用變數名 x₁…,一般矩陣用「第 n 行」。 */
function varName(c: number, isSystem: boolean): string {
  return isSystem ? `x${c + 1}` : `第 ${c + 1} 行`
}

/** 把浮點數收成最多 4 位、去掉尾隨 0(−0 也歸 0)。 */
function fmt(n: number): string {
  return Number(n.toFixed(4)).toString()
}
