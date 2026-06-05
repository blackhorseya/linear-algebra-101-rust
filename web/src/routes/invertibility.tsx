import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import type { EroKind, InverseStepJS, InverseTraceJS } from '../lib/linalg'
import { InverseCanvas } from '../components/InverseCanvas'
import { PlaybackControls } from '../components/Playback'
import { Status } from '../components/ui'
import { fmt } from '../lib/format'

export const Route = createFileRoute('/invertibility')({
  component: InvertibilityDemo,
})

type Size = 2 | 3 | 4

/** 各尺寸的預設矩陣(皆可逆;3×3 刻意讓第一個 pivot 需要換列)。 */
const DEFAULT_GRIDS: Record<Size, number[][]> = {
  2: [
    [2, 1],
    [1, 1],
  ],
  3: [
    [0, 2, 1],
    [4, 1, 0],
    [2, 1, 1],
  ],
  4: [
    [1, 2, 0, 1],
    [0, 1, 3, 0],
    [0, 0, 1, 2],
    [0, 0, 0, 1],
  ],
}

const ERO_LABEL: Record<EroKind, string> = {
  initial: '初始矩陣',
  swap: '列交換',
  scale: '列伸縮',
  addScaled: '列倍加',
}

/** n = 2 時各 ERO 對應的幾何意義(基本矩陣作為 2D 變換)。 */
const ERO_GEOMETRY: Record<EroKind, string> = {
  initial: '尚未施作任何列運算',
  swap: '鏡射(對 y = x 翻面)',
  scale: '伸縮(沿座標軸拉伸 / 壓縮)',
  addScaled: '剪切(shear)',
}

const ERO_STYLE: Record<EroKind, string> = {
  initial: 'bg-slate-700 text-slate-200',
  swap: 'bg-sky-600/80 text-sky-50',
  scale: 'bg-amber-600/80 text-amber-50',
  addScaled: 'bg-violet-600/80 text-violet-50',
}

const SUBSCRIPT: Record<Size, string> = { 2: '₂', 3: '₃', 4: '₄' }

function InvertibilityDemo() {
  // WASM 載入交給 Query 管;與其他頁共用同一 query key,整個 app 只載入一次。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  const [n, setN] = useState<Size>(2)
  const [grid, setGrid] = useState<number[][]>(DEFAULT_GRIDS[2])
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  // trace 計算便宜,同步算(同 /elimination)。grid / n 一變就重算。
  const trace = useMemo(
    () => (linalg ? linalg.invertTrace(grid.flat(), n) : null),
    [linalg, grid, n],
  )
  const stepCount = trace?.steps.length ?? 0

  // trace 換了(改 grid 或 n)→ 回第 0 步並停止播放。「render 期間比對前次值」
  // 模式(同 /elimination),不在 effect 裡 setState。
  const [prevTrace, setPrevTrace] = useState(trace)
  if (trace !== prevTrace) {
    setPrevTrace(trace)
    setCurrentStep(0)
    setIsPlaying(false)
  }

  // 鍵盤左右鍵逐步導航;任何手動互動先停掉自動播放(使用者動手即接管)。
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'ArrowLeft') {
        setIsPlaying(false)
        setCurrentStep((s) => Math.max(0, s - 1))
      } else if (e.key === 'ArrowRight') {
        setIsPlaying(false)
        setCurrentStep((s) => Math.min(stepCount - 1, s + 1))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [stepCount])

  // 自動播放:每 700ms 推進一步(每步排一個 timeout,cleanup 清掉)。
  // 到尾自停的 setState 收在 timeout callback 內(effect body 不直接 setState)。
  useEffect(() => {
    if (!isPlaying) return
    const id = setTimeout(() => {
      const next = Math.min(stepCount - 1, currentStep + 1)
      setCurrentStep(next)
      if (next >= stepCount - 1) setIsPlaying(false) // 到尾自停
    }, 700)
    return () => clearTimeout(id)
  }, [isPlaying, currentStep, stepCount])

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg || !trace)
    return <Status>WASM 載入失敗:{String(error)}</Status>

  const safeStep = Math.min(currentStep, stepCount - 1)
  const step = trace.steps[safeStep]

  // --- 互動 ---
  const goTo = (s: number) => {
    setIsPlaying(false) // 手動互動接管,停掉自動播放
    setCurrentStep(s)
  }
  const togglePlay = () => {
    if (!isPlaying && safeStep >= stepCount - 1) setCurrentStep(0) // 在結尾按 ▶:從頭重播
    setIsPlaying((p) => !p)
  }
  const setCell = (r: number, c: number, value: number) =>
    setGrid((g) =>
      g.map((row, ri) =>
        ri === r ? row.map((v, ci) => (ci === c ? value : v)) : row,
      ),
    )
  const changeSize = (next: Size) => {
    setN(next)
    setGrid((g) => resizeGrid(g, next))
  }
  const reset = () => setGrid(DEFAULT_GRIDS[n].map((row) => [...row]))
  // 小整數(−5…5)隨機重填:消去後的 P 元素仍是好讀的分數。
  const randomize = () =>
    setGrid((g) =>
      g.map((row) => row.map(() => Math.floor(Math.random() * 11) - 5)),
    )

  const eroCount = stepCount - 1 // 扣掉 initial 步,= 實際施作的基本矩陣個數

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          可逆矩陣與基本矩陣
        </h1>
        <p className="text-sm text-slate-400">
          對 A 做 Gauss–Jordan 消去:每施一個列運算,等於<strong>左乘一個基本矩陣
          E</strong>。累積 P = Eₖ⋯E₂E₁ 滿足 P·A = R(RREF)—— A 可逆 ⟺ R =
          I{SUBSCRIPT[n]},此時 P 就是 A⁻¹。每一步皆由 Rust(WASM)計算,用 ← / →
          鍵或 ▶ 自動播放。
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <SizeToggle n={n} onChange={changeSize} />
        <button
          onClick={randomize}
          className="rounded border border-slate-700 px-3 py-1.5 text-sm text-slate-300 transition hover:border-violet-500 hover:text-violet-300"
        >
          🎲 隨機填入
        </button>
        <button
          onClick={reset}
          className="rounded border border-slate-700 px-3 py-1.5 text-sm text-slate-300 transition hover:border-violet-500 hover:text-violet-300"
        >
          ↺ 重設
        </button>
      </div>

      <fieldset className="space-y-2">
        <legend className="mb-1 text-sm font-medium text-slate-300">
          輸入矩陣 A({n}×{n})
        </legend>
        <EditableGrid grid={grid} onCell={setCell} />
      </fieldset>

      {n === 2 && (
        <div className="space-y-2">
          <h2 className="text-sm font-medium text-slate-300">
            幾何視圖:基本矩陣逐步把 A 的平面變回標準網格
          </h2>
          <p className="text-xs text-slate-500">
            網格是當前 working(Eₖ⋯E₁·A)作為 2D 變換的像 —— 列交換 = 鏡射、列伸縮
            = 軸向伸縮、列倍加 = 剪切。可逆時終點回到標準網格(I₂);奇異時平面塌成一條線。
          </p>
          <InverseCanvas
            linalg={linalg}
            target={{
              a: step.working[0][0],
              b: step.working[0][1],
              c: step.working[1][0],
              d: step.working[1][1],
            }}
          />
        </div>
      )}

      <div className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <div className="flex items-center justify-between gap-4">
          <span
            className={`rounded px-2 py-1 text-xs font-medium ${ERO_STYLE[step.ero]}`}
          >
            {ERO_LABEL[step.ero]}
            {n === 2 ? ` — ${ERO_GEOMETRY[step.ero]}` : ''}
          </span>
          <span className="font-mono text-sm text-slate-400">
            第 {safeStep + 1} / {stepCount} 步
          </span>
        </div>

        <p className="text-center font-mono text-lg text-slate-100">
          {step.description}
        </p>

        <div className="flex flex-wrap items-start justify-center gap-x-8 gap-y-4 overflow-x-auto py-2">
          <LabeledMatrix
            title="working(消去中,終態 = R)"
            matrix={step.working}
            step={step}
            showPivot
          />
          <LabeledMatrix
            title="P = Eₖ⋯E₁(可逆時終態 = A⁻¹)"
            matrix={step.p}
            step={step}
          />
          <LabeledMatrix
            title={safeStep === 0 ? 'Eₖ(尚未施作 = I)' : '當步 Eₖ'}
            matrix={step.e}
            step={step}
          />
        </div>

        <PlaybackControls
          step={safeStep}
          count={stepCount}
          onChange={goTo}
          isPlaying={isPlaying}
          onTogglePlay={togglePlay}
        />
      </div>

      <IMTPanel
        trace={trace}
        eroCount={eroCount}
        det={
          n === 2
            ? linalg.determinant(grid[0][0], grid[0][1], grid[1][0], grid[1][1])
            : null
        }
      />
    </section>
  )
}

/** 切換尺寸:保留共有左上角;放大時向 Iₙ 靠攏(新對角補 1、其餘補 0),預設仍可逆。 */
function resizeGrid(g: number[][], next: Size): number[][] {
  return Array.from({ length: next }, (_, r) =>
    Array.from({ length: next }, (_, c) => g[r]?.[c] ?? (r === c ? 1 : 0)),
  )
}

function SizeToggle({
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
function EditableGrid({
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

/**
 * 帶標題的唯讀矩陣。三個矩陣(working / P / Eₖ)共用同一步的高亮:
 * 被改動的列同步翠綠(同一個 E 左乘,改的列必相同);pivot 只在 working 上標。
 */
function LabeledMatrix({
  title,
  matrix,
  step,
  showPivot,
}: {
  title: string
  matrix: number[][]
  step: InverseStepJS
  showPivot?: boolean
}) {
  const cols = matrix[0]?.length ?? 0
  const changed = new Set(step.changedRows)
  return (
    <figure className="space-y-2">
      <figcaption className="text-center text-xs text-slate-400">
        {title}
      </figcaption>
      <div
        className="inline-grid gap-1"
        style={{ gridTemplateColumns: `repeat(${cols}, 3.5rem)` }}
      >
        {matrix.map((row, r) =>
          row.map((val, c) => {
            const isPivot =
              (showPivot ?? false) && r === step.pivotRow && c === step.pivotCol
            const inChanged = changed.has(r)
            return (
              <div
                key={`${r}-${c}`}
                className={[
                  'flex h-10 w-14 items-center justify-center rounded font-mono text-xs tabular-nums transition-colors',
                  isPivot
                    ? 'bg-violet-600 text-white ring-2 ring-violet-300'
                    : inChanged
                      ? 'bg-emerald-900/50 text-emerald-100'
                      : 'bg-slate-800 text-slate-200',
                ].join(' ')}
              >
                {fmt(val)}
              </div>
            )
          }),
        )}
      </div>
    </figure>
  )
}

/**
 * 可逆矩陣定理(IMT,Theorem 2.6)面板:每個等價條件都由 Rust 端**獨立計算**
 * (非彼此推導)—— 改動矩陣時它們永遠一起翻轉,這正是定理的內容。
 */
function IMTPanel({
  trace,
  eroCount,
  det,
}: {
  trace: InverseTraceJS
  eroCount: number
  det: number | null
}) {
  const { n, invertible, rank, nullity, colsIndependent } = trace
  const sub = SUBSCRIPT[n as Size] ?? 'ₙ'
  const conditions: { label: string; value: string; ok: boolean }[] = [
    {
      label: 'A 是可逆的(存在 A⁻¹)',
      value: invertible ? '可逆' : '不可逆',
      ok: invertible,
    },
    {
      label: `RREF(A) = I${sub}`,
      value: invertible ? '成立' : 'RREF 含零列',
      ok: invertible,
    },
    {
      label: `rank(A) = ${n}`,
      value: `rank = ${rank}`,
      ok: rank === n,
    },
    {
      label: 'nullity(A) = 0',
      value: `nullity = ${nullity}`,
      ok: nullity === 0,
    },
    {
      label: '行向量線性獨立',
      value: colsIndependent ? '獨立' : '相依',
      ok: colsIndependent,
    },
    {
      label: 'A 可寫成基本矩陣的乘積',
      value: invertible ? `A = E₁⁻¹⋯Eₖ⁻¹(k = ${eroCount})` : '不可',
      ok: invertible,
    },
  ]
  if (det != null) {
    conditions.splice(5, 0, {
      label: 'det(A) ≠ 0',
      value: `det = ${fmt(det)}`,
      ok: Math.abs(det) > 1e-9,
    })
  }
  return (
    <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
      <h2 className="text-sm font-medium text-slate-300">
        可逆矩陣定理(IMT)— 等價條件一起翻轉
      </h2>
      <ul className="grid gap-2 sm:grid-cols-2">
        {conditions.map((c) => (
          <li
            key={c.label}
            className="flex items-center justify-between gap-3 rounded border border-slate-800 bg-slate-950/40 px-3 py-2"
          >
            <span className="text-sm text-slate-300">{c.label}</span>
            <span
              className={`font-mono text-xs ${c.ok ? 'text-emerald-400' : 'text-rose-400'}`}
            >
              {c.ok ? '✓' : '✗'} {c.value}
            </span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-slate-500">
        每個條件都由 Rust 端獨立計算(非彼此推導)。改動上方矩陣,看它們永遠同進退
        —— 這正是 Theorem 2.6 說的「等價」。
      </p>
    </div>
  )
}
