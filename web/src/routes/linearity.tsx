import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { LinearityCanvas, type LinearityMode } from '../components/LinearityCanvas'
import { NumberField, Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Matrix2x2, type Vec2 } from '../lib/canvas'

export const Route = createFileRoute('/linearity')({
  component: LinearityDemo,
})

// 單元 5-1 點名的轉換 preset。shear 保面積(det = 1)、投影塌陷(det = 0,
// 不可逆但仍線性)、identity / zero 是「最簡單的兩個線性轉換」。
const PRESETS = [
  {
    id: 'shear',
    name: '剪切 shear',
    m: { a: 1, b: 1, c: 0, d: 1 },
    desc: 'x 分量被 y 分量推著走、y 不動;網格斜掉但面積不變(det = 1)。',
  },
  {
    id: 'projX',
    name: '投影到 x 軸',
    m: { a: 1, b: 0, c: 0, d: 0 },
    desc: '丟掉 y 分量,整個平面被拍到 x 軸上(det = 0:不可逆,但仍線性)。',
  },
  {
    id: 'projY',
    name: '投影到 y 軸',
    m: { a: 0, b: 0, c: 0, d: 1 },
    desc: '丟掉 x 分量,整個平面被拍到 y 軸上(det = 0:不可逆,但仍線性)。',
  },
  {
    id: 'projDiag',
    name: '投影到 y = x',
    m: { a: 0.5, b: 0.5, c: 0.5, d: 0.5 },
    desc: '每個點被垂直拍到對角線 y = x 上 —— 投影不必沿著座標軸。',
  },
  {
    id: 'identity',
    name: '單位 I',
    m: { a: 1, b: 0, c: 0, d: 1 },
    desc: 'I(x) = x,什麼都不動 —— 變換後網格與原網格完全重合。',
  },
  {
    id: 'zero',
    name: '零 0',
    m: { a: 0, b: 0, c: 0, d: 0 },
    desc: 'T₀(x) = 0,整個平面被吸進原點 —— 所有影像箭頭消失。',
  },
] as const

function sameMatrix(p: Matrix2x2, q: Matrix2x2): boolean {
  return p.a === q.a && p.b === q.b && p.c === q.c && p.d === q.d
}

function LinearityDemo() {
  // WASM 模組初次載入是唯一非同步點 → 交給 Query 管 loading / error。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:preset 按鈕、矩陣輸入框、Canvas 共用這份 state(Canvas 全 controlled)。
  const [m, setM] = useState<Matrix2x2>(PRESETS[0].m)
  const [u, setU] = useState<Vec2>({ x: 2, y: 1 })
  const [v, setV] = useState<Vec2>({ x: -1, y: 2 })
  const [k, setK] = useState(2)
  const [mode, setMode] = useState<LinearityMode>('add')

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  const activePreset = PRESETS.find((p) => sameMatrix(p.m, m))

  // 兩條獨立計算路徑全在 Rust:先合成再過 T vs 各自過 T 再合成。
  const T = (x: number, y: number) => linalg.transformPoint(m.a, m.b, m.c, m.d, x, y)
  const sum = linalg.addVectors(u.x, u.y, v.x, v.y)
  const tsum = T(sum[0], sum[1]) // T(u+v)
  const tu = T(u.x, u.y)
  const tv = T(v.x, v.y)
  const tutv = linalg.addVectors(tu[0], tu[1], tv[0], tv[1]) // T(u)+T(v)
  const ku = linalg.scaleVector(u.x, u.y, k)
  const tku = T(ku[0], ku[1]) // T(k·u)
  const ktu = linalg.scaleVector(tu[0], tu[1], k) // k·T(u)

  // 線性檢查的單一真相:core 的 verify_linearity(經 WASM)。
  const linear = linalg.checkLinearity(m.a, m.b, m.c, m.d, u.x, u.y, v.x, v.y, k)
  const det = linalg.determinant([m.a, m.b, m.c, m.d], 2)
  const collapsed = Math.abs(det) < 1e-9

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          線性轉換與守恆律
        </h1>
        <p className="text-sm text-slate-400">
          矩陣 A 不只是數字表,它是一個函數 T_A: ℝ² → ℝ²,T_A(x) = Ax。拖動向量,
          看 shear / 投影怎麼搬動它的影像 —— 所有計算由 Rust(WASM)完成。
        </p>
      </div>

      {/* preset 選擇:單元 5-1 點名的轉換 */}
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => setM(p.m)}
              className={`rounded-md border px-3 py-1.5 text-sm transition ${
                activePreset?.id === p.id
                  ? 'border-violet-500 bg-violet-500/15 text-violet-200'
                  : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500'
              }`}
            >
              {p.name}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500">
          {activePreset?.desc ?? '自訂矩陣 —— 任何 2×2 矩陣誘導的轉換都是線性的(Theorem 2.7)。'}
        </p>
      </div>

      {/* 守恆律模式切換 */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex rounded-md border border-slate-700 p-0.5">
          {(
            [
              ['add', '加法守恆 T(u+v) = T(u)+T(v)'],
              ['scale', '純量乘守恆 T(k·u) = k·T(u)'],
            ] as const
          ).map(([id, name]) => (
            <button
              key={id}
              type="button"
              onClick={() => setMode(id)}
              className={`rounded px-3 py-1.5 text-sm transition ${
                mode === id
                  ? 'bg-violet-500/20 text-violet-200'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {name}
            </button>
          ))}
        </div>
        {mode === 'scale' && (
          <label className="flex items-center gap-3 text-sm text-slate-300">
            k = <span className="w-10 font-mono text-slate-100">{fmt(k)}</span>
            <input
              type="range"
              min={-3}
              max={3}
              step={0.5}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-40 accent-violet-500"
            />
          </label>
        )}
      </div>

      <div className="space-y-3">
        <LinearityCanvas
          linalg={linalg}
          m={m}
          u={u}
          v={v}
          k={k}
          mode={mode}
          onChangeU={setU}
          onChangeV={setV}
        />
        <p className="text-xs text-slate-500">
          {mode === 'add' ? (
            <>
              拖<span className="text-violet-400"> u(紫)</span>與
              <span className="text-sky-400"> v(藍)</span>。
              <span className="text-emerald-400">T(u+v)(綠箭頭)</span>是「先合成、再變換」,
              <span className="text-slate-200">白色圓環</span>是「先變換、再合成」的 T(u)+T(v) ——
              兩條路永遠在同一點會合。
            </>
          ) : (
            <>
              拖<span className="text-violet-400"> u(紫)</span>、用滑桿調 k。
              <span className="text-emerald-400">T(k·u)(綠箭頭)</span>與
              <span className="text-slate-200">白色圓環 k·T(u)</span>
              永遠重合 —— 先縮放或先變換,結果相同。
            </>
          )}
        </p>
      </div>

      {/* 自訂矩陣 */}
      <fieldset className="space-y-3">
        <legend className="mb-1 text-sm font-medium text-slate-300">
          變換矩陣 A(可自訂)
        </legend>
        <div className="grid w-fit grid-cols-2 gap-2">
          <NumberField label="a" value={m.a} onChange={(a) => setM({ ...m, a })} />
          <NumberField label="b" value={m.b} onChange={(b) => setM({ ...m, b })} />
          <NumberField label="c" value={m.c} onChange={(c) => setM({ ...m, c })} />
          <NumberField label="d" value={m.d} onChange={(d) => setM({ ...m, d })} />
        </div>
      </fieldset>

      {/* 數字對帳:兩條計算路徑(都在 Rust 算)各自攤開 */}
      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        {mode === 'add' ? (
          <>
            <Row label="路徑一:T(u+v)(先合成再變換)">
              <span className="text-emerald-300">
                ({fmt(tsum[0])}, {fmt(tsum[1])})
              </span>
            </Row>
            <Row label="路徑二:T(u)+T(v)(先變換再合成)">
              <span className="text-slate-200">
                ({fmt(tutv[0])}, {fmt(tutv[1])})
              </span>
            </Row>
          </>
        ) : (
          <>
            <Row label="路徑一:T(k·u)(先縮放再變換)">
              <span className="text-emerald-300">
                ({fmt(tku[0])}, {fmt(tku[1])})
              </span>
            </Row>
            <Row label="路徑二:k·T(u)(先變換再縮放)">
              <span className="text-slate-200">
                ({fmt(ktu[0])}, {fmt(ktu[1])})
              </span>
            </Row>
          </>
        )}

        <div className="border-t border-slate-800 pt-3">
          <Row label="core 的 verify_linearity(u, v, k)">
            <span className={linear ? 'text-emerald-400' : 'text-red-400'}>
              {linear ? '✓ 線性' : '✗ 非線性'}
            </span>
          </Row>
          <p className="mt-2 text-sm text-slate-400">
            這個 ✓ 不是前端寫死的 —— 每次拖動都把 (u, v, k) 丟回 Rust core 的{' '}
            <code className="text-slate-300">verify_linearity</code> 重新檢查。
            Theorem 2.7 保證:<span className="text-slate-200">矩陣誘導的轉換必為線性</span>,
            所以無論怎麼拖、怎麼換矩陣,它永遠是 ✓。
          </p>
        </div>

        <div className="border-t border-slate-800 pt-3">
          <Row label="行列式 det">
            <span className={collapsed ? 'text-amber-400' : 'text-slate-200'}>{fmt(det)}</span>
          </Row>
          {collapsed && (
            <p className="mt-2 text-sm text-slate-400">
              det = 0:平面被壓扁,這個轉換<span className="text-amber-300">不可逆</span> ——
              但守恆律照樣成立。<span className="text-slate-200">線性與可逆是兩回事</span>
              (投影、零轉換都是「不可逆的線性轉換」)。
            </p>
          )}
        </div>
      </div>
    </section>
  )
}
