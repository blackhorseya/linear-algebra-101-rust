import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg, type RuleKind } from '../lib/linalg'
import { SamplerCanvas } from '../components/SamplerCanvas'
import { Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Vec2 } from '../lib/canvas'

export const Route = createFileRoute('/standard-matrix')({
  component: StandardMatrixDemo,
})

/** 規則參數的滑桿規格(rotate 的值以「度」呈現,過 WASM 前轉弧度)。 */
interface ParamSpec {
  label: string
  unit?: string
  min: number
  max: number
  step: number
  def: number
}

// 單元 5-2 點名的幾何規則。每一條都「只有規則、沒有矩陣」——
// 矩陣留給 core 的 standard_matrix 取樣發現(Theorem 2.9 的工作流)。
const RULES: {
  id: RuleKind
  name: string
  desc: string
  param?: ParamSpec
}[] = [
  {
    id: 'rotate',
    name: '旋轉 θ',
    desc: '規則:(x, y) ↦ (x·cosθ − y·sinθ, x·sinθ + y·cosθ)。旋轉矩陣不用背 —— 問 e₁、e₂ 轉去哪就好。',
    param: { label: 'θ', unit: '°', min: -180, max: 180, step: 15, def: 90 },
  },
  {
    id: 'reflectX',
    name: 'x 軸反射',
    desc: '規則:x 不動、y 翻號 —— 練習 2 的主角,(x, y) ↦ (x, −y)。',
  },
  {
    id: 'reflectDiag',
    name: '對 y = x 反射',
    desc: '規則:x 與 y 互換,(x, y) ↦ (y, x) —— e₁ 與 e₂ 的影像剛好對調。',
  },
  {
    id: 'shearX',
    name: '剪切 k',
    desc: '規則:x 被 y 推 k 倍、y 不動,(x, y) ↦ (x + k·y, y)。注意 e₁ 在 x 軸上(y = 0),完全不動。',
    param: { label: 'k', min: -3, max: 3, step: 0.5, def: 1 },
  },
  {
    id: 'projectX',
    name: '投影到 x 軸',
    desc: '規則:丟掉 y,(x, y) ↦ (x, 0)。e₂ 被拍到原點 —— 矩陣第 2 行整行歸零(det = 0,不可逆但仍線性)。',
  },
  {
    id: 'scale',
    name: '等比縮放 k',
    desc: '規則:(x, y) ↦ (k·x, k·y)。取樣出 k·I;k 為負時整個平面對原點翻轉。',
    param: { label: 'k', min: -2, max: 2, step: 0.25, def: 1.5 },
  },
]

function StandardMatrixDemo() {
  // WASM 模組初次載入是唯一非同步點 → 交給 Query 管 loading / error。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:規則按鈕、滑桿、Canvas 共用這份 state(Canvas 全 controlled)。
  // 切規則時參數重設為該規則的預設值。
  const [ruleId, setRuleId] = useState<RuleKind>('rotate')
  const [param, setParam] = useState(90)
  const [v, setV] = useState<Vec2>({ x: 2, y: 1 })

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  const rule = RULES.find((r) => r.id === ruleId) ?? RULES[0]
  // 滑桿以「度」呈現旋轉角(UX),過 WASM 前轉弧度 —— 純單位換算
  //(沿 toScreen 的「呈現換算」性質),不是線代運算。
  const wasmParam = ruleId === 'rotate' ? (param * Math.PI) / 180 : param

  // 取樣!矩陣由 core 的 standard_matrix 對規則做 e₁、e₂ 取樣「發現」——
  // 頁面上沒有任何一個矩陣字面值。
  const sampled = linalg.sampleStandardMatrix(ruleId, wasmParam)
  const m = { a: sampled[0], b: sampled[1], c: sampled[2], d: sampled[3] }

  // 兩條影像路徑(都在 Rust 算):規則直接施作 vs 左乘取樣矩陣。
  const te1 = linalg.applyRule(ruleId, wasmParam, 1, 0)
  const te2 = linalg.applyRule(ruleId, wasmParam, 0, 1)
  const tv = linalg.applyRule(ruleId, wasmParam, v.x, v.y)
  const av = linalg.transformPoint(m.a, m.b, m.c, m.d, v.x, v.y)

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          標準矩陣取樣
        </h1>
        <p className="text-sm text-slate-400">
          Theorem 2.9:每個線性轉換都由唯一的矩陣誘導,而那個矩陣的第 j 行就是
          T(eⱼ)。下面每條幾何規則都「只有規則、沒有矩陣」—— 矩陣是 Rust core 的{' '}
          <code className="text-slate-300">standard_matrix</code> 當場取樣發現的。
        </p>
      </div>

      {/* 規則選擇:單元 5-2 點名的幾何轉換 */}
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {RULES.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => {
                setRuleId(r.id)
                setParam(r.param?.def ?? 0)
              }}
              className={`rounded-md border px-3 py-1.5 text-sm transition ${
                ruleId === r.id
                  ? 'border-violet-500 bg-violet-500/15 text-violet-200'
                  : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500'
              }`}
            >
              {r.name}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500">{rule.desc}</p>
        {rule.param && (
          <label className="flex items-center gap-3 text-sm text-slate-300">
            {rule.param.label} ={' '}
            <span className="w-14 font-mono text-slate-100">
              {fmt(param)}
              {rule.param.unit ?? ''}
            </span>
            <input
              type="range"
              min={rule.param.min}
              max={rule.param.max}
              step={rule.param.step}
              value={param}
              onChange={(e) => setParam(Number(e.target.value))}
              className="w-40 accent-violet-500"
            />
          </label>
        )}
      </div>

      <div className="space-y-3">
        <SamplerCanvas
          linalg={linalg}
          rule={ruleId}
          param={wasmParam}
          m={m}
          v={v}
          onChangeV={setV}
        />
        <p className="text-xs text-slate-500">
          <span className="text-violet-400">e₁</span>、
          <span className="text-sky-400">e₂</span>(細箭頭)是被取樣的標準基底;
          <span className="text-amber-400">T(e₁)</span>、
          <span className="text-rose-400">T(e₂)</span>(粗箭頭)是它們的影像 ——
          顏色對應下方矩陣的兩行。拖<span className="text-slate-200">
            {' '}
            v(白)
          </span>:
          <span className="text-emerald-400">T(v)(綠箭頭)</span>是規則直接算、
          <span className="text-slate-200">白色圓環</span>是左乘取樣矩陣的 A·v ——
          兩條路永遠會合。
        </p>
      </div>

      {/* 取樣結果:T(e_j) 直放成 A 的第 j 行 */}
      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <Row label="取樣 e₁ 的影像 T(e₁)(→ A 的第 1 行)">
          <span className="text-amber-300">
            ({fmt(te1[0])}, {fmt(te1[1])})
          </span>
        </Row>
        <Row label="取樣 e₂ 的影像 T(e₂)(→ A 的第 2 行)">
          <span className="text-rose-300">
            ({fmt(te2[0])}, {fmt(te2[1])})
          </span>
        </Row>

        <div className="flex items-center gap-4 border-t border-slate-800 pt-3">
          <span className="text-sm text-slate-400">
            直放成行,標準矩陣就「長」出來了:
          </span>
          <div className="flex items-center gap-2 font-mono">
            <span className="text-slate-500">A =</span>
            <div className="grid grid-cols-2 gap-x-5 gap-y-1 rounded-md border border-slate-700 px-4 py-2 text-right">
              <span className="text-amber-300">{fmt(m.a)}</span>
              <span className="text-rose-300">{fmt(m.b)}</span>
              <span className="text-amber-300">{fmt(m.c)}</span>
              <span className="text-rose-300">{fmt(m.d)}</span>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-800 pt-3">
          <Row label="路徑一:T(v)(規則直接算)">
            <span className="text-emerald-300">
              ({fmt(tv[0])}, {fmt(tv[1])})
            </span>
          </Row>
          <Row label="路徑二:A·v(左乘取樣出的矩陣)">
            <span className="text-slate-200">
              ({fmt(av[0])}, {fmt(av[1])})
            </span>
          </Row>
          <p className="mt-2 text-sm text-slate-400">
            無論 v 拖到哪,兩條路都會合 —— 這就是 Theorem 2.9:線性轉換在標準基底上的
            n 個取樣<span className="text-slate-200">決定了它在整個空間的行為</span>。
            n 個影像、一張矩陣,函數與資料一一對應。
          </p>
        </div>
      </div>
    </section>
  )
}
