import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { SpanCanvas } from '../components/SpanCanvas'
import { NumberField, Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { ORIGIN_EPS, type Vec2 } from '../lib/canvas'

export const Route = createFileRoute('/span')({
  component: SpanDemo,
})

function SpanDemo() {
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 兩個向量;預設線性獨立 → 張成整個平面。
  const [v, setV] = useState<Vec2>({ x: 2, y: 1 })
  const [w, setW] = useState<Vec2>({ x: -1, y: 1 })

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  // 相依與否、張成什麼,全用 WASM 的 are_parallel 判斷(線性相依 = 共線)。
  const vZero = Math.hypot(v.x, v.y) < ORIGIN_EPS
  const wZero = Math.hypot(w.x, w.y) < ORIGIN_EPS
  const bothZero = vZero && wZero
  const dependent = bothZero || linalg.areParallel(v.x, v.y, w.x, w.y)

  const spanText = bothZero
    ? '只有原點 {0}(0 維)'
    : dependent
      ? '一條過原點的直線(1 維)'
      : '整個平面 ℝ²(2 維)'

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          向量的張成(Span)
        </h1>
        <p className="text-sm text-slate-400">
          兩個向量能「張成」多大的空間?線性相依(共線)與否的判定全由 Rust(WASM)計算。
        </p>
      </div>

      <div className="space-y-3">
        <SpanCanvas linalg={linalg} v={v} w={w} onChangeV={setV} onChangeW={setW} />
        <p className="text-xs text-slate-500">
          拖
          <span className="text-violet-400"> v(紫)</span>、
          <span className="text-cyan-400">w(青)</span> 兩向量。獨立時可看到它們張出的
          網格鋪滿整個平面;把其中一個拖到與另一個共線,平面就塌成
          <span className="text-teal-300"> 一條線</span>。
        </p>
      </div>

      <div className="grid gap-8 sm:grid-cols-2">
        <fieldset className="space-y-3">
          <legend className="mb-1 text-sm font-medium text-slate-300">向量 v</legend>
          <div className="flex gap-2">
            <NumberField label="x" value={v.x} onChange={(x) => setV({ ...v, x })} />
            <NumberField label="y" value={v.y} onChange={(y) => setV({ ...v, y })} />
          </div>
        </fieldset>

        <fieldset className="space-y-3">
          <legend className="mb-1 text-sm font-medium text-slate-300">向量 w</legend>
          <div className="flex gap-2">
            <NumberField label="x" value={w.x} onChange={(x) => setW({ ...w, x })} />
            <NumberField label="y" value={w.y} onChange={(y) => setW({ ...w, y })} />
          </div>
        </fieldset>
      </div>

      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <Row label="v, w">
          <code className="text-violet-300">
            ({fmt(v.x)}, {fmt(v.y)})
          </code>
          <span className="px-1 text-slate-600">·</span>
          <code className="text-cyan-300">
            ({fmt(w.x)}, {fmt(w.y)})
          </code>
        </Row>
        <Row label="線性關係">
          <span className={dependent ? 'text-amber-400' : 'text-emerald-400'}>
            {bothZero ? '皆為零向量' : dependent ? '線性相依(共線)' : '線性獨立'}
          </span>
        </Row>
        <Row label="span(v, w)">
          <span className="text-slate-200">{spanText}</span>
        </Row>
        {!dependent && (
          <p className="border-t border-slate-800 pt-3 text-sm text-slate-400">
            任意點都能寫成{' '}
            <span className="text-slate-200">a·v + b·w</span> —— v、w
            是平面的一組基底,網格鋪滿整個 ℝ²。
          </p>
        )}
      </div>
    </section>
  )
}
