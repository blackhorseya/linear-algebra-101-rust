import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { RankCanvas } from '../components/RankCanvas'
import { NumberField, Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Matrix2x2 } from '../lib/canvas'

export const Route = createFileRoute('/rank')({
  component: RankDemo,
})

// 三種秩的代表。重點在 rank 1:Row A 與 Col A 是**不同方向**的兩條線,維度卻相等。
const PRESETS: { name: string; desc: string; m: Matrix2x2 }[] = [
  {
    name: '不同線、同維(rank 1)',
    desc: 'A = [[1,1],[2,2]]:列 (1,1)、(2,2) 共線 → Row A 是 (1,1) 方向的線;行 (1,2)、(1,2) 共線 → Col A 是 (1,2) 方向的線。兩條**不同**的線,但 dim 都是 1 —— rank(A) = rank(Aᵀ) 的精華。',
    m: { a: 1, b: 1, c: 2, d: 2 },
  },
  {
    name: '滿秩(rank 2)',
    desc: 'Row A 與 Col A 都是整個 ℝ²(兩面板都鋪滿):列獨立 ⟺ 行獨立,dim 都是 2。怎麼拖都無法讓一邊滿、另一邊不滿。',
    m: { a: 2, b: 1, c: 1, d: 1 },
  },
  {
    name: '零矩陣(rank 0)',
    desc: '列全為 0、行全為 0:Row A = Col A = {0},dim 都是 0。兩個極端在這裡會合。',
    m: { a: 0, b: 0, c: 0, d: 0 },
  },
]

/** 把攤平的基底(長度 0 / 2 / 4)印成可讀的向量列。 */
function fmtBasis(b: Float64Array): string {
  if (b.length === 0) return '∅(只有 0)'
  const vs: string[] = []
  for (let i = 0; i < b.length; i += 2) vs.push(`(${fmt(b[i])}, ${fmt(b[i + 1])})`)
  return vs.join(', ')
}

function RankDemo() {
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:數字輸入框、preset、兩個面板的拖曳共用這份 state。
  // 預設 rank 1 —— 一進來就看到「Row A 與 Col A 是不同的線,維度卻相等」。
  const [matrix, setMatrix] = useState<Matrix2x2>(PRESETS[0].m)

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  const { a, b, c, d } = matrix
  // 全部由 core 算。dim Col A = rank(A)、dim Row A = rank(Aᵀ),兩條獨立路徑 ——
  // 它們相等不是前端湊的,是 core 各跑一次消去法的結果。
  const rank = linalg.rank(a, b, c, d) // dim Col A
  const rankT = linalg.rankTranspose(a, b, c, d) // dim Row A,經 rank(Aᵀ)
  const rowBasis = linalg.rowSpaceBasis(a, b, c, d) // Row A 基底(RREF 列)
  const colBasis = linalg.rangeBasis(a, b, c, d) // Col A 基底(原始行)

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          行秩 = 列秩:Row A 與 Col A 同維
        </h1>
        <p className="text-sm text-slate-400">
          一個矩陣有兩個子空間:
          <span className="text-sky-400">Row A</span>(列張成,住在 domain)與{' '}
          <span className="text-amber-400">Col A</span>(行張成,住在 codomain)——
          這是 <code className="text-slate-300">/range</code> 看的{' '}
          <span className="text-amber-300">值域</span>。它們是{' '}
          <span className="text-slate-200">不同空間裡的不同子空間</span>,維度卻{' '}
          <span className="text-slate-200">永遠相等</span>:這就是定理{' '}
          <code className="text-slate-300">rank(A) = rank(Aᵀ)</code>。拖左面板的{' '}
          <span className="text-sky-400">列向量</span>或右面板的{' '}
          <span className="text-amber-400">行向量</span>,看兩邊的維度同進同退 ——
          維度由 Rust core 的 <code className="text-slate-300">rank</code> /{' '}
          <code className="text-slate-300">rank_transpose</code> 各自獨立算。
        </p>
      </div>

      {/* Preset:三種秩 */}
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
        {/* 矩陣輸入:與兩個面板共用同一份 state */}
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
        <RankCanvas linalg={linalg} m={matrix} onChangeM={setMatrix} />
        <p className="text-xs text-slate-500">
          左面板拖<span className="text-sky-400">列向量 r₁、r₂</span>(= A 的兩列),
          右面板拖<span className="text-amber-400">行向量 a₁、a₂</span>(= A 的兩行)——
          兩者拖的是<span className="text-slate-300">同一個 A</span>,所以互相連動。
          粗線 / 鋪面是 span(由 core 的{' '}
          <code className="text-slate-300">row_space_basis</code> /{' '}
          <code className="text-slate-300">range_basis</code>{' '}
          的維度決定):一條線 = dim 1、整面鋪滿 = dim 2、只剩原點 = dim 0。
        </p>
      </div>

      {/* 對帳面板:dim Row A 與 dim Col A 各自由 core 算,相等即 rank(A)=rank(Aᵀ) */}
      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        <Row label="dim Row A(domain,經 rank Aᵀ)">
          <span className="text-sky-300">{rankT}</span>
        </Row>
        <Row label="dim Col A(codomain,經 rank A)">
          <span className="text-amber-300">{rank}</span>
        </Row>
        <Row label="定理 rank(A) = rank(Aᵀ)">
          <span className="text-slate-100">
            {rankT} = {rank}
          </span>
          <span className="ml-2 text-emerald-400">✓</span>
        </Row>

        <div className="space-y-2 border-t border-slate-800 pt-3">
          <Row label="Row A 基底(RREF 非零列,canonical)">
            <span className="text-sky-300">{fmtBasis(rowBasis)}</span>
          </Row>
          <Row label="Col A 基底(pivot 的原始行)">
            <span className="text-amber-300">{fmtBasis(colBasis)}</span>
          </Row>
          <p className="mt-2 text-sm text-slate-400">
            兩組基底的<span className="text-slate-300">支數一樣</span>(都 = rank)——
            但向量本身不同,而且取法相反:
            <span className="text-sky-300">Row A 就地讀 RREF 的列</span>(列運算保留
            Row A),<span className="text-amber-300">Col A 回頭抓原始的行</span>
            (列運算破壞 Col A,RREF 的行已不在 Col A 裡)。同一台消去法,兩端各取所需。
          </p>
        </div>
      </div>
    </section>
  )
}
