import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { DeterminantCanvas } from '../components/DeterminantCanvas'
import { EditableGrid, SizeToggle } from '../components/MatrixGrid'
import { resizeGrid, type Size } from '../lib/grid'
import { Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Matrix2x2 } from '../lib/canvas'

export const Route = createFileRoute('/determinant')({
  component: DeterminantDemo,
})

/** 各尺寸的預設矩陣,各講一個 det 的故事。 */
const DEFAULT_GRIDS: Record<Size, number[][]> = {
  // det = 3:面積 ×3,定向不變(平行四邊形看得最清楚)
  2: [
    [2, 1],
    [1, 2],
  ],
  // det = −6:體積 ×6 且定向反轉(鏡像)—— 負號在 n = 3 也有意義
  3: [
    [0, 2, 1],
    [4, 1, 0],
    [2, 1, 1],
  ],
  // 上三角:det = 對角線乘積 1·1·2·3 = 6(Theorem 3.2 親眼看)
  4: [
    [1, 2, 0, 1],
    [0, 1, 3, 0],
    [0, 0, 2, 2],
    [0, 0, 0, 3],
  ],
}

/** det 的幾何量詞隨維度升級:面積 → 體積 → 超體積。 */
const VOLUME_NOUN: Record<Size, string> = {
  2: '面積',
  3: '體積',
  4: '4 維超體積',
}

/** 塌縮(det = 0)在各維度的畫面。 */
const COLLAPSE_DESC: Record<Size, string> = {
  2: '平面被壓扁成一條線(或一點)',
  3: '空間被壓扁成一個平面(或更低維)',
  4: 'ℝ⁴ 被壓扁到更低維的子空間',
}

function DeterminantDemo() {
  // WASM 載入交給 Query 管;與其他頁共用同一 query key,整個 app 只載入一次。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  const [n, setN] = useState<Size>(2)
  const [grid, setGrid] = useState<number[][]>(DEFAULT_GRIDS[2])

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  // 章門面兩顆,同步呼叫(計算全在 Rust):
  // det 路(Gaussian 消去)給數值與正負號;rank 路(is_invertible)給塌縮判定。
  // 刻意不彼此推導 —— 面板拿兩者對帳,Theorem 3.4(a) 每一幀上演。
  const det = linalg.determinant(grid.flat(), n)
  const invertible = linalg.isInvertible(grid.flat(), n)
  const detNonzero = Math.abs(det) > 1e-9
  const agree = detNonzero === invertible // 定理保證恆 true

  // --- 互動 ---
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
  // 小整數(−5…5)隨機重填,det 仍是好讀的整數。
  const randomize = () =>
    setGrid((g) =>
      g.map((row) => row.map(() => Math.floor(Math.random() * 11) - 5)),
    )

  // 2×2 時 canvas 與輸入網格共用同一份 grid(單一真相);拖曳寫回 grid。
  const m: Matrix2x2 = {
    a: grid[0][0],
    b: grid[0][1],
    c: grid[1][0],
    d: grid[1][1],
  }
  const setM = (next: Matrix2x2) =>
    setGrid([
      [next.a, next.b],
      [next.c, next.d],
    ])

  const noun = VOLUME_NOUN[n]
  const detMeaning = !detNonzero
    ? `det = 0 → ${COLLAPSE_DESC[n]},變換不可逆。`
    : det < 0
      ? `${noun}縮放 ${fmt(Math.abs(det))} 倍,且定向反轉(鏡像翻面)。`
      : `${noun}縮放 ${fmt(det)} 倍,定向不變。`

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          行列式 — 有號{noun}
        </h1>
        <p className="text-sm text-slate-400">
          det(A) 是單位{n === 2 ? '正方形' : n === 3 ? '立方體' : '超立方體'}
          經 A 變換後的<strong>有號{noun}</strong>:|det| 是{noun}
          縮放倍率,負號代表定向反轉,det = 0 代表塌縮(不可逆)。數值由
          Rust(WASM)的 Gaussian 消去計算(O(n³),不是 O(n!) 的餘因子展開)。
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
          矩陣 A({n}×{n})
        </legend>
        <EditableGrid grid={grid} onCell={setCell} />
      </fieldset>

      {n === 2 && (
        <div className="space-y-3">
          <DeterminantCanvas linalg={linalg} m={m} onChangeMatrix={setM} />
          <p className="text-xs text-slate-500">
            虛線是變換前的單位正方形(面積 1);著色平行四邊形是它的像,面積 =
            |det|。拖
            <span className="text-emerald-400"> î′(綠)</span>、
            <span className="text-red-400">ĵ′(紅)</span>
            改變矩陣:<span className="text-violet-400">紫色</span>
            = 定向不變、<span className="text-amber-400">琥珀</span>
            = 翻面(det &lt; 0);把兩箭頭拖到共線,平行四邊形塌成
            <span className="text-rose-400">紅線</span>(det = 0)。
          </p>
        </div>
      )}

      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <Row label="det(A)">
          <span
            className={
              !detNonzero
                ? 'text-rose-400'
                : det < 0
                  ? 'text-amber-400'
                  : 'text-emerald-400'
            }
          >
            {fmt(det)}
          </span>
        </Row>
        {n === 2 && (
          // 2×2 才有好寫的封閉式;n ≥ 3 的「展開」是 O(n!) 的餘因子,不攤了。
          <p className="font-mono text-xs text-slate-500">
            det = a·d − b·c = {fmt(m.a)}·{fmt(m.d)} − {fmt(m.b)}·{fmt(m.c)} ={' '}
            <span className="text-slate-300">{fmt(det)}</span>
          </p>
        )}
        <p className="text-sm text-slate-400">{detMeaning}</p>
      </div>

      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <h2 className="text-sm font-medium text-slate-300">
          Theorem 3.4(a):A 可逆 ⟺ det(A) ≠ 0 — 兩條獨立計算的對帳
        </h2>
        <ul className="grid gap-2 sm:grid-cols-3">
          <Lamp
            ok={detNonzero}
            title="det 路"
            sub="Gaussian 消去求 det"
            value={detNonzero ? `det = ${fmt(det)} ≠ 0` : 'det = 0'}
          />
          <Lamp
            ok={invertible}
            title="rank 路"
            sub="is_invertible(rank 滿不滿)"
            value={invertible ? '可逆' : '塌縮(不可逆)'}
          />
          <Lamp
            ok={agree}
            title="對帳"
            sub="兩路必同答案"
            value={agree ? '一致 ✓' : '分歧?!'}
          />
        </ul>
        <p className="text-xs text-slate-500">
          det 路與 rank 路是 Rust 端<strong>兩條獨立的計算</strong>
          (非彼此推導)。隨意改動矩陣,前兩燈永遠同進退、第三燈永遠亮 ——
          「可逆 ⟺ det ≠ 0」不是寫死的假設,是每一幀上演的定理。
        </p>
      </div>
    </section>
  )
}

/** 對帳面板的單顆燈:條件名 + 計算路徑 + 當前值。 */
function Lamp({
  ok,
  title,
  sub,
  value,
}: {
  ok: boolean
  title: string
  sub: string
  value: string
}) {
  return (
    <li className="space-y-1 rounded border border-slate-800 bg-slate-950/40 px-3 py-2">
      <p className="text-sm text-slate-300">{title}</p>
      <p className="text-xs text-slate-500">{sub}</p>
      <p
        className={`font-mono text-xs ${ok ? 'text-emerald-400' : 'text-rose-400'}`}
      >
        {ok ? '✓' : '✗'} {value}
      </p>
    </li>
  )
}
