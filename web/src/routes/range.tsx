import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { RangeCanvas } from '../components/RangeCanvas'
import { NumberField, Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Matrix2x2, type Vec2 } from '../lib/canvas'

export const Route = createFileRoute('/range')({
  component: RangeDemo,
})

// 三種秩各給代表:值域 = 整個平面 / 直線 / 原點 —— Range(T) = Col(A),
// 行向量怎麼擺,值域就怎麼長。
const PRESETS: { name: string; desc: string; m: Matrix2x2 }[] = [
  {
    name: '可逆(rank 2)',
    desc: '行向量獨立 → 張滿整個 ℝ²:哪裡都到得了,T 映成(且每個 w 恰一個前身)。',
    m: { a: 2, b: 1, c: 1, d: 1 },
  },
  {
    name: '行共線(rank 1)',
    desc: 'a₂ = 2·a₁:兩支生成元素張不出平面,值域塌成直線 —— 線外全是不可達的 w。',
    m: { a: 1, b: 2, c: 2, d: 4 },
  },
  {
    name: '投影到 x 軸(rank 1)',
    desc: '第二行是零向量(對張成毫無貢獻):值域 = x 軸,基底只剩一支。',
    m: { a: 1, b: 0, c: 0, d: 0 },
  },
  {
    name: '零矩陣(rank 0)',
    desc: '一切都被吸進原點:值域 = {0},基底是空集合 —— 除了原點全不可達。',
    m: { a: 0, b: 0, c: 0, d: 0 },
  },
]

function RangeDemo() {
  // WASM 模組初次載入是唯一非同步點 → 交給 Query 管 loading / error。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:數字輸入框、preset 按鈕、Canvas 拖曳共用這份 state。
  const [matrix, setMatrix] = useState<Matrix2x2>(PRESETS[0].m)
  const [w, setW] = useState<Vec2>({ x: 3, y: 1 })

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  // 全部由 core 計算(經 WASM):各自獨立呼叫,不互相推導 —— 它們的一致
  // (基底支數 = rank、映成 ⟺ 無見證⋯)是 range 模組 laws 證過的定理。
  const { a, b, c, d } = matrix
  const basis = linalg.rangeBasis(a, b, c, d)
  const rank = basis.length / 2
  const onto = linalg.isOnto(a, b, c, d)
  const witness = linalg.unreachableVector(a, b, c, d)
  const reachable = linalg.rangeContains(a, b, c, d, w.x, w.y)
  const solved = linalg.solveForInput(a, b, c, d, w.x, w.y)

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          值域與映成
        </h1>
        <p className="text-sm text-slate-400">
          Range(T) = Col(A):值域就是行向量張成的空間。拖動{' '}
          <span className="text-amber-400">a₁</span>、
          <span className="text-rose-400">a₂</span>(矩陣的行)看值域從整個平面
          塌成直線、再塌進原點;拖 w 問「到得了嗎?」—— 可達性、基底、映成判定
          全部由 Rust core 的 <code className="text-slate-300">range</code>{' '}
          模組當場計算。
        </p>
      </div>

      {/* Preset:三種秩的代表 */}
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              onClick={() => setMatrix(p.m)}
              className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-300 transition hover:border-violet-500/60 hover:text-violet-200"
            >
              {p.name}
            </button>
          ))}
        </div>
        {/* 矩陣輸入:與 Canvas 拖曳共用同一份 state(雙向同步) */}
        <div className="flex items-end gap-4">
          <span className="pb-1 font-mono text-slate-500">A =</span>
          <div className="grid grid-cols-2 gap-2">
            <NumberField
              label="a(a₁ 的 x)"
              value={a}
              onChange={(v) => setMatrix({ ...matrix, a: v })}
            />
            <NumberField
              label="b(a₂ 的 x)"
              value={b}
              onChange={(v) => setMatrix({ ...matrix, b: v })}
            />
            <NumberField
              label="c(a₁ 的 y)"
              value={c}
              onChange={(v) => setMatrix({ ...matrix, c: v })}
            />
            <NumberField
              label="d(a₂ 的 y)"
              value={d}
              onChange={(v) => setMatrix({ ...matrix, d: v })}
            />
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <RangeCanvas
          linalg={linalg}
          m={matrix}
          w={w}
          onChangeM={setMatrix}
          onChangeW={setW}
        />
        <p className="text-xs text-slate-500">
          淡紫網格是整個平面的像(= 值域的覆蓋);
          <span className="text-violet-400">紫色粗線</span>
          是塌縮後的 Range 直線(由 core 的 range_basis 給方向)。拖{' '}
          <span className="text-slate-200">w</span>:
          <span className="text-emerald-400">綠 ✓ 可達</span> /{' '}
          <span className="text-red-400">紅 ✗ 不可達</span>由 core 的
          range_contains 即時判定;唯一解時虛線箭頭是 w 的前身 x,
          <span className="text-slate-200">白圓環</span>(A·x)必套住 w ——
          兩路會合。不映成時<span className="text-red-400">紅圈</span>
          標出被漏掉的 eᵢ(unreachable_vector 的見證)。
        </p>
      </div>

      {/* 判定面板:每個值各自由 core 算,一致性是定理(laws 證過),不是前端共用布林 */}
      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <Row label="生成集合(矩陣的行)">
          <span className="text-amber-300">
            a₁ = ({fmt(a)}, {fmt(c)})
          </span>
          <span className="text-slate-500">、</span>
          <span className="text-rose-300">
            a₂ = ({fmt(b)}, {fmt(d)})
          </span>
        </Row>
        <Row label="值域的基底(range_basis 蒸餾,支數 = rank)">
          {rank === 0 ? (
            <span className="text-slate-400">∅(值域 = {'{0}'})</span>
          ) : (
            <span className="text-violet-300">
              {basis.length >= 2 && `(${fmt(basis[0])}, ${fmt(basis[1])})`}
              {basis.length === 4 &&
                `、(${fmt(basis[2])}, ${fmt(basis[3])})`}
            </span>
          )}
        </Row>
        <Row label="dim Range(T) = rank(A)">
          <span className="text-slate-100">{rank}</span>
        </Row>
        <Row label="映成(Theorem 2.10:rank = m = 2)?">
          {onto ? (
            <span className="text-emerald-300">✓ 映成 —— Range = ℝ²,處處可達</span>
          ) : (
            <span className="text-red-300">
              ✗ 不映成 —— 見證:{witness[0] === 1 ? 'e₁' : 'e₂'} 不可達
            </span>
          )}
        </Row>

        <div className="border-t border-slate-800 pt-3">
          <Row label={`w = (${fmt(w.x)}, ${fmt(w.y)}) ∈ Range(T)?`}>
            {reachable ? (
              <span className="text-emerald-300">✓ 可達(Ax = w 相容)</span>
            ) : (
              <span className="text-red-300">✗ 不可達(Ax = w 無解)</span>
            )}
          </Row>
          <p className="mt-2 text-sm text-slate-400">
            {solved.kind === 'Unique' && solved.x && (
              <>
                唯一的前身:x = ({fmt(solved.x[0])}, {fmt(solved.x[1])}) ——
                左乘 A 後正好落在 w(圖上白圓環套住綠箭頭)。
              </>
            )}
            {solved.kind === 'Infinite' && (
              <>
                無限多前身 —— 行相依讓不同輸入擠進同一個像:到得了,
                但「誰到的」不唯一(這正是下一單元 one-to-one 要管的事)。
              </>
            )}
            {solved.kind === 'Inconsistent' && (
              <>
                沒有任何輸入到得了 w —— 增廣矩陣 [A | w] 化簡後冒出矛盾列,
                w 就是「值域沒蓋滿」的活見證。
              </>
            )}
          </p>
        </div>
      </div>
    </section>
  )
}
