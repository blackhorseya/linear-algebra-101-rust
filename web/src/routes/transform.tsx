import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import {
  TransformCanvas,
  type Matrix2x2,
  type Vec2,
} from '../components/TransformCanvas'

export const Route = createFileRoute('/transform')({
  component: TransformDemo,
})

function TransformDemo() {
  // WASM 模組初次載入是這裡唯一真正非同步的部分 → 交給 Query 管 loading / error。
  // QueryClient 已設 staleTime: Infinity,所以整個 app 只會載入一次。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:數字輸入框與 Canvas 共用這份 m / v(Canvas 全 controlled)。
  // 預設 90° 逆時針旋轉作用在 (1,1)。
  const [m, setM] = useState<Matrix2x2>({ a: 0, b: -1, c: 1, d: 0 })
  const [v, setV] = useState<Vec2>({ x: 1, y: 1 })

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  // 同步呼叫:2×2 乘法很便宜,不必再包一層 Query —— 計算全在 Rust 完成。
  const [tx, ty] = linalg.transformPoint(m.a, m.b, m.c, m.d, v.x, v.y)
  const isZero = v.x === 0 && v.y === 0
  const parallel = linalg.areParallel(v.x, v.y, tx, ty)

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          2D 線性變換
        </h1>
        <p className="text-sm text-slate-400">
          矩陣與向量運算全部由 Rust(WASM)計算,JS 只負責畫圖與互動。
        </p>
      </div>

      <div className="space-y-3">
        <TransformCanvas
          linalg={linalg}
          m={m}
          v={v}
          onChangeMatrix={setM}
          onChangeV={setV}
        />
        <p className="text-xs text-slate-500">
          拖
          <span className="text-emerald-400"> î′(綠)</span>、
          <span className="text-red-400">ĵ′(紅)</span> 端點即時改矩陣;拖
          <span className="text-violet-400"> v(紫)</span> 看
          <span className="text-amber-400"> A·v(琥珀)</span>。把 î′ 拖到與 ĵ′
          共線會看到平面塌成一條線。
        </p>
      </div>

      <div className="grid gap-8 sm:grid-cols-2">
        <fieldset className="space-y-3">
          <legend className="mb-1 text-sm font-medium text-slate-300">
            變換矩陣 A
          </legend>
          <div className="grid w-fit grid-cols-2 gap-2">
            <NumberField label="a" value={m.a} onChange={(a) => setM({ ...m, a })} />
            <NumberField label="b" value={m.b} onChange={(b) => setM({ ...m, b })} />
            <NumberField label="c" value={m.c} onChange={(c) => setM({ ...m, c })} />
            <NumberField label="d" value={m.d} onChange={(d) => setM({ ...m, d })} />
          </div>
        </fieldset>

        <fieldset className="space-y-3">
          <legend className="mb-1 text-sm font-medium text-slate-300">
            輸入點 (x, y)
          </legend>
          <div className="flex gap-2">
            <NumberField label="x" value={v.x} onChange={(x) => setV({ ...v, x })} />
            <NumberField label="y" value={v.y} onChange={(y) => setV({ ...v, y })} />
          </div>
        </fieldset>
      </div>

      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <Row label="A · v">
          <code className="text-violet-300">
            ({fmt(tx)}, {fmt(ty)})
          </code>
        </Row>
        {/* 把矩陣乘法攤開:每一項都帶入實際數字,看得到哪個格子貢獻了多少。 */}
        <div className="space-y-1 border-t border-slate-800 pt-3 font-mono text-xs text-slate-500">
          <p>
            x′ = a·x + b·y = {fmt(m.a)}·{fmt(v.x)} + {fmt(m.b)}·{fmt(v.y)} ={' '}
            <span className="text-slate-300">{fmt(tx)}</span>
          </p>
          <p>
            y′ = c·x + d·y = {fmt(m.c)}·{fmt(v.x)} + {fmt(m.d)}·{fmt(v.y)} ={' '}
            <span className="text-slate-300">{fmt(ty)}</span>
          </p>
        </div>
        <Row label="v 與 A·v 平行?">
          <span className={parallel ? 'text-emerald-400' : 'text-slate-400'}>
            {parallel ? '是' : '否'}
          </span>
        </Row>
        {parallel && !isZero && (
          <p className="text-sm text-slate-400">
            v 與 A·v 共線 → <span className="text-slate-200">v 落在這個變換的特徵向量方向上</span>
            (A·v = λv)。
          </p>
        )}
      </div>
    </section>
  )
}

function Status({ children }: { children: React.ReactNode }) {
  return <p className="text-slate-400">{children}</p>
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="font-mono text-slate-100">{children}</span>
    </div>
  )
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (value: number) => void
}) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-slate-400">{label}</span>
      <input
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-24 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100 focus:border-violet-500 focus:outline-none"
      />
    </label>
  )
}

/** 把浮點數收成最多 4 位、去掉尾隨 0(-0 也歸 0)。 */
function fmt(n: number): string {
  return Number(n.toFixed(4)).toString()
}
