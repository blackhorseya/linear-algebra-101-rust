import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { EigenvaluesCanvas, SNAP_EPSILON } from '../components/EigenvaluesCanvas'
import { NumberField, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Matrix2x2 } from '../lib/canvas'

export const Route = createFileRoute('/eigenvalues')({
  component: EigenvaluesDemo,
})

interface Frame {
  a: Matrix2x2 // 運算子 A(row-major a, b, c, d)
  lambda: number // 位移量 λ
}

// 每個 preset 的 λ 預設落在某個特徵值上,一進來就看到「塌縮 + 特徵向量現形」。
const PRESETS: { name: string; desc: string; frame: Frame }[] = [
  {
    name: '非均勻縮放 · λ=2',
    desc: 'A = diag(2, 3),特徵值 2 與 3。λ = 2 時 A−2I = diag(0, 1) 把 x 方向壓成 0 → 平行四邊形塌成一條直線,被壓到原點的方向 (1,0) 就是特徵值 2 的特徵向量。滑到 λ = 3 看另一條。',
    frame: { a: { a: 2, b: 0, c: 0, d: 3 }, lambda: 2 },
  },
  {
    name: '剪切 · λ=1(重根)',
    desc: 'A = [[1,1],[0,1]] 是水平剪切,特徵值只有一個(λ = 1,重根)且特徵空間只有一維 —— 只有 x 軸方向被保留。滑離 1 平行四邊形就鼓起來(det ≠ 0)。',
    frame: { a: { a: 1, b: 1, c: 0, d: 1 }, lambda: 1 },
  },
  {
    name: '反射 · λ=−1',
    desc: 'A = [[1,0],[0,−1]] 對 x 軸反射,特徵值 +1(x 軸不動)與 −1(y 軸翻向)。λ = −1 時塌縮方向是 (0,1):反射把它送到反向 = −1 倍自己。',
    frame: { a: { a: 1, b: 0, c: 0, d: -1 }, lambda: -1 },
  },
  {
    name: '旋轉 90° · 無實特徵值',
    desc: 'A = [[0,−1],[1,0]] 把每個方向都轉 90°,沒有任何方向被保留 —— det(A−λI) = λ²+1 對任何實 λ 都 > 0,平行四邊形永遠不塌。這就是「沒有實特徵值」的幾何長相。',
    frame: { a: { a: 0, b: -1, c: 1, d: 0 }, lambda: 0 },
  },
]

/** 2×2 矩陣顯示(row-major `[m11, m12, m21, m22]`)。 */
function Mat2({ m, accent }: { m: readonly number[]; accent?: string }) {
  return (
    <span
      className={`inline-grid grid-cols-2 gap-x-4 gap-y-0.5 rounded border border-slate-700 px-3 py-1.5 text-right font-mono text-sm ${accent ?? 'text-slate-100'}`}
    >
      <span>{fmt(m[0])}</span>
      <span>{fmt(m[1])}</span>
      <span>{fmt(m[2])}</span>
      <span>{fmt(m[3])}</span>
    </span>
  )
}

function EigenvaluesDemo() {
  const { data: linalg, isLoading, error } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:輸入框、preset、λ 滑桿、canvas 拖曳共用這份 state。
  const [frame, setFrame] = useState<Frame>(PRESETS[0].frame)

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  const { a, lambda } = frame
  const aFlat = [a.a, a.b, a.c, a.d]
  // 全部由 core 算:M = A−λI(characteristic_matrix)、det(A−λI)(determinant)、
  // Eλ 基底(eigenspace_basis → null_space_basis)、是否有實特徵值。
  const m = linalg.characteristicMatrix2d(aFlat, lambda)
  const detM = m.length === 4 ? linalg.determinant([m[0], m[1], m[2], m[3]], 2) : null
  const eig = linalg.eigenspaceBasis2d(aFlat, lambda, SNAP_EPSILON)
  const hasReal = linalg.hasRealEigenvalues2x2(aFlat)
  const eigenDim = eig.length / 2 // 0 / 1 / 2

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          特徵值:讓 A − λI 塌縮的那個 λ
        </h1>
        <p className="text-sm text-slate-400">
          特徵向量是被 <span className="text-amber-400">A</span> 作用後{' '}
          <span className="text-sky-400">只伸縮、方向不變</span>的向量(
          <code className="text-slate-300">A·v = λv</code>)。移項成{' '}
          <code className="text-slate-300">(A − λI)·v = 0</code> —— 特徵向量就是{' '}
          <code className="text-slate-300">A − λI</code> 的零空間。所以找特徵值 ={' '}
          找讓 <code className="text-slate-300">A − λI</code> <span className="text-slate-200">奇異</span>
          (壓扁、det = 0)的 λ。拖 <span className="text-slate-200">λ 滑桿</span>,看{' '}
          <code className="text-slate-300">A − λI</code> 把單位方塊送到的平行四邊形:
          <code className="text-slate-300">det(A − λI)</code> 滑到 0 時它{' '}
          <span className="text-sky-400">塌成一條線</span>,那一刻 λ 就是特徵值,被壓到原點的方向{' '}
          <span className="text-sky-400">v</span> 就是特徵向量(core 的{' '}
          <code className="text-slate-300">eigenspace_basis</code> 給;
          <span className="text-slate-50">白圈 Av = λv</span> 落回 v 的線上 = 它真的只被伸縮)。拖{' '}
          <span className="text-amber-400">î′</span> / <span className="text-orange-400">ĵ′</span>{' '}
          改 A 本身。
        </p>
      </div>

      {/* Preset */}
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              onClick={() => setFrame(p.frame)}
              className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-300 transition hover:border-sky-500/60 hover:text-sky-200"
            >
              {p.name}
            </button>
          ))}
        </div>
        {/* 輸入框:與 canvas 拖曳、λ 滑桿共用同一份 state */}
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="space-y-1">
            <span className="text-xs text-amber-400">運算子 A(行 = î′ / ĵ′)</span>
            <div className="flex gap-2">
              <NumberField label="a₁₁" value={a.a} onChange={(v) => setFrame({ ...frame, a: { ...a, a: v } })} />
              <NumberField label="a₁₂" value={a.b} onChange={(v) => setFrame({ ...frame, a: { ...a, b: v } })} />
              <NumberField label="a₂₁" value={a.c} onChange={(v) => setFrame({ ...frame, a: { ...a, c: v } })} />
              <NumberField label="a₂₂" value={a.d} onChange={(v) => setFrame({ ...frame, a: { ...a, d: v } })} />
            </div>
          </div>
          <NumberField label="λ" value={lambda} onChange={(v) => setFrame({ ...frame, lambda: v })} />
        </div>
        {/* λ 滑桿:主角。拖它找 det(A−λI) = 0 */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs text-sky-400">位移量 λ</span>
            <span className="font-mono text-sm text-slate-300">λ = {fmt(lambda)}</span>
          </div>
          <input
            type="range"
            min={-4}
            max={5}
            step={0.05}
            value={lambda}
            onChange={(e) => setFrame({ ...frame, lambda: Number(e.target.value) })}
            className="w-full max-w-xl accent-sky-400"
          />
        </div>
      </div>

      <div className="space-y-3">
        <EigenvaluesCanvas
          linalg={linalg}
          a={a}
          lambda={lambda}
          onChangeA={(mat) => setFrame({ ...frame, a: mat })}
        />
        <p className="text-xs text-slate-500">
          灰虛框 = 原始單位方塊;
          <span className="text-emerald-400">綠</span> /{' '}
          <span className="text-rose-400">紅</span>平行四邊形 ={' '}
          <code className="text-slate-300">A − λI</code> 把它送到的像(面積 = |det(A−λI)|,顏色 = 定向)——
          λ = 0 時正好貼著 <span className="text-amber-400">î′</span> /{' '}
          <span className="text-orange-400">ĵ′</span>,λ 一動就往內縮。塌成線時{' '}
          <span className="text-sky-400">藍虛線</span> = 特徵向量方向。
        </p>
      </div>

      {/* 對帳面板 */}
      <div className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
          <div className="space-y-1">
            <span className="text-xs text-amber-400">標準矩陣 A</span>
            <Mat2 m={aFlat} accent="text-amber-200" />
          </div>
          <div className="space-y-1">
            <span className="text-xs text-sky-400">A − λI(λ = {fmt(lambda)})</span>
            <div className="flex items-center gap-3">
              <Mat2 m={m.length === 4 ? [m[0], m[1], m[2], m[3]] : aFlat} accent="text-sky-200" />
              <span className="font-mono text-sm text-slate-400">
                det = {detM === null ? '—' : fmt(detM)}
              </span>
            </div>
          </div>
        </div>
        <div className="border-t border-slate-800 pt-3 text-sm">
          {!hasReal ? (
            <p className="text-rose-300">
              此矩陣<strong>沒有實特徵值</strong>:det(A − λI) 對任何實數 λ 都 &gt; 0,平行四邊形永遠不塌
              —— 純旋轉把每個方向都轉開,沒有方向被保留。
            </p>
          ) : eigenDim >= 2 ? (
            <p className="text-slate-400">
              <span className="text-sky-400">λ = {fmt(lambda)} 是特徵值</span>,且 Eλ ={' '}
              <span className="text-slate-200">整個平面</span>(純量矩陣 λI):A − λI = 0,
              <span className="text-slate-200">每個向量都是特徵向量</span>。
            </p>
          ) : eigenDim === 1 ? (
            <p className="text-slate-400">
              <span className="text-sky-400">λ = {fmt(lambda)} 是特徵值 ✓</span> —— A − λI 奇異
              (det ≈ 0),平行四邊形塌成一條線。被壓到原點的方向{' '}
              <span className="text-sky-400">v</span> 就是特徵向量,
              <span className="text-slate-50">A·v = λv</span>(白圈落回 v 的線上)。滑動 λ 找其他特徵值。
            </p>
          ) : (
            <p className="text-slate-400">
              λ = {fmt(lambda)}:<span className="font-mono">det(A − λI) = {detM === null ? '—' : fmt(detM)}</span>{' '}
              ≠ 0 —— A − λI 可逆、沒被壓扁,λ <strong>不是</strong>特徵值。把滑桿拖到 det 趨近 0
              的位置,特徵向量就會現形。
            </p>
          )}
        </div>
      </div>
    </section>
  )
}
