import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { NullSpaceCanvas } from '../components/NullSpaceCanvas'
import { NumberField, Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Matrix2x2, type Vec2 } from '../lib/canvas'

export const Route = createFileRoute('/nullspace')({
  component: NullSpaceDemo,
})

// 三種秩各給代表 —— 與 /range 的 preset 對偶:那裡看「值域蓋住多少」,
// 這裡看「核壓扁多少」。rank + nullity = 2 在每個 preset 都成立。
const PRESETS: { name: string; desc: string; m: Matrix2x2 }[] = [
  {
    name: '可逆(rank 2)',
    desc: '核 = {0}:除了原點,沒有非零輸入被壓扁(nullity 0)。每個方向都保得住 → 同一個可逆性,在 /range 那端說的是「處處可達」。',
    m: { a: 2, b: 1, c: 1, d: 1 },
  },
  {
    name: '行共線(rank 1)',
    desc: 'a₂ = 2·a₁:一整條輸入方向被壓到原點 —— 核是過原點的直線(nullity 1)。沿核線的 v 全被 A 吃掉。',
    m: { a: 1, b: 2, c: 2, d: 4 },
  },
  {
    name: '投影到 x 軸(rank 1)',
    desc: '第二行是零(把高度丟掉):y 軸整條被壓扁,核 = y 軸(nullity 1)。x 軸方向則原封保留。',
    m: { a: 1, b: 0, c: 0, d: 0 },
  },
  {
    name: '零矩陣(rank 0)',
    desc: '一切都被壓進原點:核 = 整個 ℝ²(nullity 2)。值域只剩 {0} —— rank-nullity 的兩個極端同時發生。',
    m: { a: 0, b: 0, c: 0, d: 0 },
  },
]

function NullSpaceDemo() {
  // WASM 模組初次載入是唯一非同步點 → 交給 Query 管 loading / error。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:數字輸入框、preset 按鈕、Canvas 拖曳共用這份 state。
  // 預設 rank 1 配核方向上的 v —— 一進來就看到「v 在核裡、Av 塌到原點」。
  const [matrix, setMatrix] = useState<Matrix2x2>(PRESETS[1].m)
  const [v, setV] = useState<Vec2>({ x: 2, y: -1 })

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  // 全部由 core 計算(經 WASM)。nullity 與 rank 各自獨立算 —— 它們相加 = 2
  // 不是前端湊的,是 core 兩次獨立計算當場驗證的 rank-nullity 定理。
  const { a, b, c, d } = matrix
  const nul = linalg.nullity(a, b, c, d)
  const rank = linalg.rank(a, b, c, d)
  const inKernel = linalg.nullSpaceContains(a, b, c, d, v.x, v.y)
  const av = linalg.transformPoint(a, b, c, d, v.x, v.y)

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          零空間與 rank-nullity
        </h1>
        <p className="text-sm text-slate-400">
          Null A = {'{ v : Av = 0 }'}:被 A 壓到原點的輸入。這是{' '}
          <code className="text-slate-300">/range</code> 的{' '}
          <span className="text-slate-200">對偶</span> —— 那裡看輸出端的值域蓋住
          多少,這裡看輸入端被壓扁多少。拖{' '}
          <span className="text-sky-400">v</span>(domain 的輸入向量)看它的像{' '}
          <span className="text-amber-400">Av</span>;落到{' '}
          <span className="text-violet-400">核線</span>上時 Av 塌進原點,v 變綠
          —— 是否在核裡由 Rust core 的{' '}
          <code className="text-slate-300">null_space_contains</code> 當場判定。
        </p>
      </div>

      {/* Preset:三種秩的代表(與 /range 對偶) */}
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
        {/* 矩陣輸入:與 Canvas 共用同一份 state */}
        <div className="flex items-end gap-4">
          <span className="pb-1 font-mono text-slate-500">A =</span>
          <div className="grid grid-cols-2 gap-2">
            <NumberField
              label="a"
              value={a}
              onChange={(x) => setMatrix({ ...matrix, a: x })}
            />
            <NumberField
              label="b"
              value={b}
              onChange={(x) => setMatrix({ ...matrix, b: x })}
            />
            <NumberField
              label="c"
              value={c}
              onChange={(x) => setMatrix({ ...matrix, c: x })}
            />
            <NumberField
              label="d"
              value={d}
              onChange={(x) => setMatrix({ ...matrix, d: x })}
            />
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <NullSpaceCanvas linalg={linalg} m={matrix} v={v} onChangeV={setV} />
        <p className="text-xs text-slate-500">
          <span className="text-violet-400">紫色粗線</span>是核 Null A(nullity 1
          時的核線方向由 core 的像掃描出最被壓扁的方向);拖{' '}
          <span className="text-sky-400">v</span>:
          <span className="text-emerald-400">綠 ✓ 在核裡(Av = 0)</span> /{' '}
          <span className="text-sky-400">藍 ✗ 不在</span>由 core 的
          null_space_contains 即時判定。
          <span className="text-amber-400">橙虛線</span>是像 Av(transformPoint
          算);v 進核裡時 Av 塌到原點(綠圈)。零矩陣時整個平面鋪淡紫 ——
          核 = ℝ²。
        </p>
      </div>

      {/* 判定面板:nullity 與 rank 各自由 core 算,相加 = 2 是 rank-nullity 對帳 */}
      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <Row label="nullity = dim Null A(被壓扁的維度)">
          <span className="text-violet-300">{nul}</span>
          <span className="ml-2 text-slate-500">
            {nul === 0
              ? '核 = {0}(可逆,什麼都沒壓扁)'
              : nul === 1
                ? '核是一條過原點的直線'
                : '核 = 整個 ℝ²(零矩陣)'}
          </span>
        </Row>
        <Row label="rank = dim Col A(值域維度)">
          <span className="text-amber-300">{rank}</span>
        </Row>
        <Row label="rank-nullity 定理(domain 維度 = 2)">
          <span className="text-slate-100">
            rank {rank} + nullity {nul} = {rank + nul}
          </span>
          <span className="ml-2 text-emerald-400">✓</span>
        </Row>

        <div className="border-t border-slate-800 pt-3">
          <Row label={`v = (${fmt(v.x)}, ${fmt(v.y)}) ∈ Null A?`}>
            {inKernel ? (
              <span className="text-emerald-300">✓ 在核裡(Av = 0,被壓到原點)</span>
            ) : (
              <span className="text-sky-300">✗ 不在核裡(Av ≠ 0)</span>
            )}
          </Row>
          <p className="mt-2 text-sm text-slate-400">
            Av = ({fmt(av[0])}, {fmt(av[1])})。
            {inKernel
              ? ' v 沿核方向 → A 把它整個吃掉(像為零向量)。核是「資訊在 A 之下消失」的那些輸入。'
              : ' v 有非零的像 → 它帶的資訊沒被 A 抹掉。把 v 拖到紫色核線上,看 Av 塌進原點。'}
          </p>
        </div>
      </div>
    </section>
  )
}
