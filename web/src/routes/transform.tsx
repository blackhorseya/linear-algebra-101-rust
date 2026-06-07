import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { TransformCanvas } from '../components/TransformCanvas'
import { NumberField, Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Matrix2x2, type Vec2 } from '../lib/canvas'

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

  // 行列式 = 單位正方形像的有號面積(由 WASM 算)。
  const det = linalg.determinant([m.a, m.b, m.c, m.d], 2)
  const detClass =
    Math.abs(det) < 1e-9
      ? 'text-red-400'
      : det < 0
        ? 'text-amber-400'
        : 'text-emerald-400'
  const detMeaning =
    Math.abs(det) < 1e-9
      ? 'det = 0 → 平面被壓扁成一條線,變換不可逆(沒有反矩陣)。'
      : det < 0
        ? `面積縮放 ${fmt(Math.abs(det))} 倍,且平面翻面(定向反轉)。`
        : `面積縮放 ${fmt(det)} 倍,定向不變。`

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
        <Row label="行列式 det">
          <span className={detClass}>{fmt(det)}</span>
        </Row>
        <p className="font-mono text-xs text-slate-500">
          det = a·d − b·c = {fmt(m.a)}·{fmt(m.d)} − {fmt(m.b)}·{fmt(m.c)} ={' '}
          <span className="text-slate-300">{fmt(det)}</span>
        </p>
        <p className="text-sm text-slate-400">{detMeaning}</p>

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
