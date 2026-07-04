import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import type { ComponentProps, ReactNode } from 'react'
import { loadLinalg } from '../lib/linalg'
import { EditableGrid, SizeToggle } from '../components/MatrixGrid'
import { NumberField, Row, Status } from '../components/ui'
import { resizeGrid, type Size } from '../lib/grid'
import { fmt } from '../lib/format'
import { useSharedMatrix } from '../lib/use-shared-matrix'

// 檔名 index.tsx → 對應路徑 '/'。新增 about.tsx 就會自動產生 '/about'。
export const Route = createFileRoute('/')({
  component: Home,
})

// ---------------------------------------------------------------------------
// 計算機:一顆 A(n×n,與 ?m= 共享)配上 B / v / λ,選運算看結果 ——
// 各章節做題目時的對答案工具。每個結果都由 core 經 WASM 計算,JS 只排版。
// ---------------------------------------------------------------------------

/** λ 與特徵值「多近算相等」的容差(做題目時 λ 通常是整數,1e-6 足夠嚴格)。 */
const EIGEN_EPSILON = 1e-6

/** 各尺寸的預設 A(可逆,消去後數字好讀)。 */
const DEFAULT_A: Record<Size, number[][]> = {
  2: [
    [2, 1],
    [1, 1],
  ],
  3: [
    [0, 2, 1],
    [4, 1, 0],
    [2, 1, 1],
  ],
  4: [
    [1, 2, 0, 1],
    [0, 1, 3, 0],
    [0, 0, 2, 2],
    [0, 0, 0, 3],
  ],
}

/** 第二顆矩陣 B 的預設(單位陣起步 —— A·B = A 一眼可驗)。 */
const identity = (n: number): number[][] =>
  Array.from({ length: n }, (_, r) =>
    Array.from({ length: n }, (_, c) => (r === c ? 1 : 0)),
  )

/** 向量隨尺寸增減:保留前段、新分量補 1(全 0 的 v 對多數運算沒戲唱)。 */
const resizeVec = (v: number[], n: number): number[] =>
  Array.from({ length: n }, (_, i) => v[i] ?? 1)

type OpId =
  | 'Av'
  | 'AB'
  | 'det'
  | 'inv'
  | 'rref'
  | 'solve'
  | 'subspace'
  | 'eigen'
  | 'coords'

/** 運算選單:need 決定顯示哪些輸入;only2 = 目前 binding 僅支援 2×2。 */
const OPS: {
  id: OpId
  name: string
  need: ('B' | 'v' | 'lambda')[]
  only2?: boolean
}[] = [
  { id: 'Av', name: 'A·v', need: ['v'] },
  { id: 'AB', name: 'A·B', need: ['B'] },
  { id: 'det', name: 'det(A)', need: [] },
  { id: 'inv', name: 'A⁻¹', need: [] },
  { id: 'rref', name: 'RREF·rank', need: [] },
  { id: 'solve', name: '解 Ax = b', need: ['v'] },
  { id: 'subspace', name: '子空間基底', need: [], only2: true },
  { id: 'eigen', name: '特徵檢定', need: ['lambda'], only2: true },
  { id: 'coords', name: '座標·[T]_B', need: ['B', 'v'], only2: true },
]

/** 唯讀矩陣(計算結果用,monospace 格子)。 */
function Mat({ grid }: { grid: number[][] }) {
  const cols = grid[0]?.length ?? 0
  return (
    <div
      className="inline-grid gap-1 align-middle"
      style={{ gridTemplateColumns: `repeat(${cols}, minmax(3rem, auto))` }}
    >
      {grid.map((row, r) =>
        row.map((v, c) => (
          <span
            key={`${r}-${c}`}
            className="flex h-9 items-center justify-center rounded bg-slate-800 px-2 font-mono text-xs tabular-nums text-slate-200"
          >
            {fmt(v)}
          </span>
        )),
      )}
    </div>
  )
}

/** 把攤平的基底(長度 0 / 2 / 4)印成向量列。 */
function fmtBasis(b: Float64Array): string {
  if (b.length === 0) return '∅(只有零向量)'
  const vs: string[] = []
  for (let i = 0; i < b.length; i += 2) vs.push(`(${fmt(b[i])}, ${fmt(b[i + 1])})`)
  return vs.join('、')
}

const fmtVec = (v: ArrayLike<number>): string =>
  `(${Array.from(v, fmt).join(', ')})`

// 深入各章的入口卡片,照 3Blue1Brown《Essence of Linear Algebra》的敘事
// 順序分節:向量 → 變換 → 合成 → 行列式 → 逆與子空間 → 換基底 → 特徵值,
// 每節標題附對應集數。`as const` 讓 `to` 保持字面量型別,Link 的路徑
// 打錯會直接編譯失敗。新增視覺化頁面時在對應章節補一筆即可。
const SECTIONS = [
  {
    title: '向量與張成',
    sub: 'Essence Ep.1–2 — 一切從向量、線性組合與 span 開始。',
    items: [
      {
        to: '/span',
        title: '張成 Span',
        desc: '兩個向量能「張成」多大的空間?拖到共線的瞬間,平面塌縮成一條線。',
      },
    ],
  },
  {
    title: '線性變換與矩陣',
    sub: 'Essence Ep.3 — 矩陣不是數字表,是把整個平面拉扯變形的動詞。',
    items: [
      {
        to: '/transform',
        title: '2D 線性變換',
        desc: '拖動向量、調整 2×2 矩陣,看整個平面如何被變形;行列式的幾何意義 = 面積縮放與翻面。',
      },
      {
        to: '/linearity',
        title: '線性轉換與守恆律',
        desc: '拖動 u、v 看 shear / 投影下的影像:T(u+v) 與 T(u)+T(v) 永遠在同一點會合 —— Theorem 2.7「矩陣誘導必線性」看得見。',
      },
      {
        to: '/standard-matrix',
        title: '標準矩陣取樣',
        desc: '選幾何規則(旋轉、反射、剪切⋯),看 e₁、e₂ 的影像被取樣、直放成矩陣的行 —— Theorem 2.9「線性轉換必由唯一矩陣誘導」現場上演。',
      },
    ],
  },
  {
    title: '乘法即合成',
    sub: 'Essence Ep.4 — 矩陣乘法就是「先做一個變換、再做下一個」。',
    items: [
      {
        to: '/multiply',
        title: '矩陣乘法',
        desc: '任意尺寸 (m×n)·(n×p) 的 row × col 互動展開:點 C 的任一格看它由哪一列點積哪一欄,並親手碰到「維度不合不能乘」。',
      },
      {
        to: '/composition',
        title: '合成與可逆性',
        desc: '拖 x 看「先 T 再 U」兩步路徑與「一步 BA」直達永遠會合(T_B ∘ T_A = T_BA);切到逆轉換模式看「變形 → 復原」,Summary Table 三燈由 core 點亮。',
      },
    ],
  },
  {
    title: '行列式',
    sub: 'Essence Ep.6 — 一個數字說完變換對面積做了什麼。',
    items: [
      {
        to: '/determinant',
        title: '行列式',
        desc: '拖 î′、ĵ′ 看單位正方形的像:|det| = 面積縮放、det < 0 = 翻面;切到 3×3 / 4×4 看 det 推廣為(超)體積,det 路與 rank 路對帳 Theorem 3.4(a)。',
      },
    ],
  },
  {
    title: '逆矩陣、秩與零空間',
    sub: 'Essence Ep.7 — 解方程組的幾何:回得去嗎?蓋得滿嗎?誰被壓扁了?',
    items: [
      {
        to: '/elimination',
        title: '高斯消去',
        desc: '逐步播放 forward / backward 消去,看矩陣化成 RREF,並判讀唯一解、無限多解或無解。',
      },
      {
        to: '/invertibility',
        title: '可逆矩陣',
        desc: 'Gauss-Jordan 累乘基本矩陣求 A⁻¹,並以 2D 變換看每個列運算的幾何意義(鏡射、伸縮、剪切)。',
      },
      {
        to: '/range',
        title: '值域與映成',
        desc: '拖動矩陣的行向量,看 Range(T) = Col(A) 從整個平面塌成直線;拖 w 問「到得了嗎?」—— 可達性與映成判定由 core 即時計算。',
      },
      {
        to: '/nullspace',
        title: '零空間與 rank-nullity',
        desc: '/range 的對偶:拖輸入 v 看它的像 Av,落到核線上時被壓到原點;nullity 與 rank 各自由 core 算,相加 = 2 當場驗證 rank-nullity 定理。',
      },
      {
        to: '/rank',
        title: '行秩 = 列秩',
        desc: 'Row A(domain)與 Col A(codomain)是不同空間的不同子空間,維度卻永遠相等。拖列向量或行向量,看兩邊的維度同進同退 —— rank(A) = rank(Aᵀ) 看得見。',
      },
    ],
  },
  {
    title: '基底變換與相似',
    sub: 'Essence Ep.13 — 換一把尺量同一個世界,矩陣跟著換臉。',
    items: [
      {
        to: '/coordinates',
        title: '座標系統',
        desc: '同一個點,換一組基底就有不同座標。拖斜格的 b₁、b₂ 換尺、拖 x 移點,看 [x]_B = (c₁, c₂) 即時變化 —— 座標是「沿 b₁、b₂ 各走幾步」的權重,由 core 解 B⁻¹x。',
      },
      {
        to: '/operator',
        title: '相似:運算子的矩陣表示',
        desc: '同一個運算子,換把尺就換個矩陣 [T]_B = B⁻¹AB。拖基底看 [T]_B 的四格變、但平行四邊形面積(det)鎖死不動 —— 相似矩陣共享 det,描述變了運算子沒變。',
      },
    ],
  },
  {
    title: '特徵值與特徵向量',
    sub: 'Essence Ep.14 — 變換翻天覆地,總有些方向只被伸縮、不被轉向。',
    items: [
      {
        to: '/eigenvalues',
        title: '特徵值與特徵向量',
        desc: '拖 λ 滑桿,看 A − λI 把單位方塊壓成的平行四邊形;det(A − λI) 滑到 0 時塌成一條線 —— 那一刻 λ 是特徵值,被壓到原點的方向就是特徵向量(A·v = λv)。',
      },
    ],
  },
] as const

const STACK = [
  ['React 19', 'UI 函式庫'],
  ['Vite', '開發伺服器與打包'],
  ['TypeScript', '型別安全'],
  ['Tailwind CSS v4', 'utility-first 樣式'],
  ['TanStack Router', '型別安全的 file-based 路由'],
  ['TanStack Query', 'server-state 與資料快取'],
] as const

function Home() {
  // WASM 模組初次載入是唯一非同步點 → 交給 Query 管 loading / error。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 共享優先:A(n×n)與 ?m= 同步,尺寸與內容都從共享矩陣還原。
  const { initial, publish } = useSharedMatrix()
  const [n, setN] = useState<Size>(() => (initial?.length as Size) ?? 2)
  const [gridA, setGridA] = useState<number[][]>(() => initial ?? DEFAULT_A[2])
  const [gridB, setGridB] = useState<number[][]>(() => identity(initial?.length ?? 2))
  const [vec, setVec] = useState<number[]>(() => resizeVec([3, 1], initial?.length ?? 2))
  const [lambda, setLambda] = useState(1)
  const [op, setOp] = useState<OpId>('Av')

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  // --- 互動(A 所有寫路徑收斂到 applyA:本地更新 + 整顆寫回共享) ---
  const applyA = (next: number[][]) => {
    setGridA(next)
    publish(next)
  }
  const setCellA = (r: number, c: number, value: number) =>
    applyA(gridA.map((row, ri) => (ri === r ? row.map((v, ci) => (ci === c ? value : v)) : row)))
  const setCellB = (r: number, c: number, value: number) =>
    setGridB((g) => g.map((row, ri) => (ri === r ? row.map((v, ci) => (ci === c ? value : v)) : row)))
  const changeSize = (next: Size) => {
    setN(next)
    applyA(resizeGrid(gridA, next))
    setGridB((g) => resizeGrid(g, next))
    setVec((v) => resizeVec(v, next))
    // 2×2 專用運算在大尺寸下沒有 binding,自動退回通用的 A·v。
    if (next !== 2 && OPS.find((o) => o.id === op)?.only2) setOp('Av')
  }

  const active = OPS.find((o) => o.id === op)!
  const opDisabled = (o: (typeof OPS)[number]) => (o.only2 ?? false) && n !== 2
  const flatA = gridA.flat()

  return (
    <section className="space-y-12">
      <div className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight text-slate-50">
          線性代數 101 · Web
        </h1>
        <p className="text-slate-400">
          matrix / vector 計算機 —— 各章做題目時的對答案工具:輸入{' '}
          <span className="text-amber-400">A</span>(必要時配上 B、v、λ),選運算看結果。
          每個數字都由 Rust core(WASM)計算,JS 只排版;A 會寫進網址,
          點進任何一章都帶著同一顆走。
        </p>
      </div>

      {/* 運算選單 */}
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {OPS.map((o) => (
            <button
              key={o.id}
              type="button"
              disabled={opDisabled(o)}
              onClick={() => setOp(o.id)}
              title={opDisabled(o) ? '目前僅支援 2×2(binding 限制)' : undefined}
              className={`rounded-md border px-3 py-1.5 text-sm transition ${
                op === o.id
                  ? 'border-violet-500 bg-violet-500/15 text-violet-200'
                  : opDisabled(o)
                    ? 'cursor-not-allowed border-slate-800 bg-slate-900/40 text-slate-600'
                    : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500'
              }`}
            >
              {o.name}
              {o.only2 && <span className="ml-1 text-[10px] text-slate-500">2×2</span>}
            </button>
          ))}
        </div>

        {/* 輸入區:A 永遠在;B / v / λ 依運算需要出現 */}
        <div className="flex flex-wrap items-start gap-x-10 gap-y-5">
          <fieldset className="space-y-2">
            <legend className="mb-1 flex items-center gap-4 text-sm font-medium text-slate-300">
              矩陣 A({n}×{n})
              <SizeToggle n={n} onChange={changeSize} />
            </legend>
            <EditableGrid grid={gridA} onCell={setCellA} />
          </fieldset>

          {active.need.includes('B') && (
            <fieldset className="space-y-2">
              <legend className="mb-1 text-sm font-medium text-slate-300">
                {op === 'coords' ? '基底矩陣 B(行 = b₁、b₂)' : `矩陣 B(${n}×${n})`}
              </legend>
              <EditableGrid grid={gridB} onCell={setCellB} />
            </fieldset>
          )}

          {active.need.includes('v') && (
            <fieldset className="space-y-2">
              <legend className="mb-1 text-sm font-medium text-slate-300">
                {op === 'solve' ? '右端向量 b' : op === 'coords' ? '點 x' : '向量 v'}
              </legend>
              <div className="flex gap-2">
                {vec.map((val, i) => (
                  <NumberField
                    key={i}
                    label={`${op === 'solve' ? 'b' : op === 'coords' ? 'x' : 'v'}${i + 1}`}
                    value={val}
                    onChange={(x) =>
                      setVec((v) => v.map((old, j) => (j === i ? x : old)))
                    }
                  />
                ))}
              </div>
            </fieldset>
          )}

          {active.need.includes('lambda') && (
            <fieldset className="space-y-2">
              <legend className="mb-1 text-sm font-medium text-slate-300">候選特徵值 λ</legend>
              <NumberField label="λ" value={lambda} onChange={setLambda} />
            </fieldset>
          )}
        </div>

        {/* 結果區 */}
        <CalcResult
          op={op}
          n={n}
          linalg={linalg}
          flatA={flatA}
          gridB={gridB}
          vec={vec}
          lambda={lambda}
        />
      </div>

      {/* 深入各章:照 3Blue1Brown《Essence of Linear Algebra》的敘事順序 */}
      <div className="space-y-8">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">深入各章</h2>
          <p className="text-xs text-slate-500">
            照 3Blue1Brown《Essence of Linear Algebra》的敘事走 —— 上面輸入的 A 會跟著你進每一頁。
          </p>
        </div>
        {SECTIONS.map(({ title, sub, items }) => (
          <div key={title} className="space-y-3">
            <div>
              <h3 className="font-semibold text-slate-200">{title}</h3>
              <p className="text-xs text-slate-500">{sub}</p>
            </div>
            <ul className="grid gap-3 sm:grid-cols-2">
              {items.map(({ to, title: cardTitle, desc }) => (
                <li key={to}>
                  <Link
                    to={to}
                    className="block h-full rounded-lg border border-slate-800 bg-slate-900/50 px-4 py-3 transition hover:border-violet-500/60 hover:bg-slate-900"
                  >
                    <p className="font-medium text-violet-300">{cardTitle} →</p>
                    <p className="mt-1 text-sm text-slate-400">{desc}</p>
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="space-y-4">
        <h2 className="text-sm font-semibold text-slate-500">技術棧</h2>
        <ul className="grid gap-2 sm:grid-cols-3">
          {STACK.map(([name, desc]) => (
            <li key={name} className="rounded-md border border-slate-800/60 px-3 py-2">
              <p className="text-sm text-slate-300">{name}</p>
              <p className="text-xs text-slate-500">{desc}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

/** 結果卡外框:標題 + 對應章節的「深入 →」(帶著共享矩陣走)。 */
function ResultCard({
  title,
  to,
  children,
}: {
  title: string
  to: ComponentProps<typeof Link>['to']
  children: ReactNode
}) {
  return (
    <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-medium text-slate-300">{title}</h2>
        <Link to={to} className="shrink-0 text-xs text-violet-400 transition hover:text-violet-300">
          深入圖解 →
        </Link>
      </div>
      {children}
    </div>
  )
}

/**
 * 依選中的運算呼叫對應 binding 並排版結果。全部同步呼叫(計算很便宜),
 * 每次輸入變動即時重算 —— 沒有「=」按鈕,答案永遠是新鮮的。
 */
function CalcResult({
  op,
  n,
  linalg,
  flatA,
  gridB,
  vec,
  lambda,
}: {
  op: OpId
  n: Size
  linalg: NonNullable<Awaited<ReturnType<typeof loadLinalg>>>
  flatA: number[]
  gridB: number[][]
  vec: number[]
  lambda: number
}) {
  switch (op) {
    case 'Av': {
      // n×n 的 A·v = A 右乘 n×1 行向量(multiply 通吃,transform_point 只管 2×2)。
      const out = linalg.multiplyExpand(flatA, n, n, vec, n, 1)
      return (
        <ResultCard title="A·v — 矩陣作用在向量上" to="/transform">
          <p className="font-mono text-lg text-amber-300">A·v = {fmtVec(out.c.map((r) => r[0]))}</p>
          <p className="text-xs text-slate-500">
            每個分量 = A 的一列與 v 的點積;2×2 時到「2D 線性變換」拖著看整個平面被變形。
          </p>
        </ResultCard>
      )
    }
    case 'AB': {
      const out = linalg.multiplyExpand(flatA, n, n, gridB.flat(), n, n)
      return (
        <ResultCard title="A·B — 矩陣乘法(合成)" to="/multiply">
          <div className="flex items-center gap-3">
            <span className="font-mono text-slate-400">A·B =</span>
            <Mat grid={out.c} />
          </div>
          <p className="text-xs text-slate-500">
            cᵢⱼ = A 第 i 列 · B 第 j 欄;到「矩陣乘法」逐格看展開項,或到「合成與可逆性」看它是兩個變換的接力。
          </p>
        </ResultCard>
      )
    }
    case 'det': {
      const det = linalg.determinant(flatA, n)
      const collapsed = Math.abs(det) < 1e-9
      return (
        <ResultCard title="det(A) — 有號體積" to="/determinant">
          <p
            className={`font-mono text-lg ${
              collapsed ? 'text-rose-400' : det < 0 ? 'text-amber-400' : 'text-emerald-400'
            }`}
          >
            det(A) = {fmt(det)}
          </p>
          <p className="text-xs text-slate-500">
            {collapsed
              ? 'det = 0:空間被壓扁,A 不可逆。'
              : det < 0
                ? `|det| = ${fmt(Math.abs(det))} 倍縮放,負號 = 定向反轉(翻面)。`
                : `單位方塊的體積被縮放 ${fmt(det)} 倍,定向不變。`}
          </p>
        </ResultCard>
      )
    }
    case 'inv': {
      // invert_trace 的終態 P = Eₖ⋯E₁ 即 A⁻¹(可逆時);IMT 旗標一併取得。
      const trace = linalg.invertTrace(flatA, n)
      const last = trace.steps[trace.steps.length - 1]
      return (
        <ResultCard title="A⁻¹ — Gauss-Jordan 求逆" to="/invertibility">
          {trace.invertible ? (
            <div className="flex items-center gap-3">
              <span className="font-mono text-slate-400">A⁻¹ =</span>
              <Mat grid={last.p} />
            </div>
          ) : (
            <p className="font-mono text-lg text-rose-400">A⁻¹ 不存在(RREF 有零列)</p>
          )}
          <Row label="IMT 對帳(各自獨立計算)">
            <span className="font-mono text-xs text-slate-300">
              rank = {trace.rank} · nullity = {trace.nullity} · 行獨立{' '}
              {trace.colsIndependent ? '✓' : '✗'}
            </span>
          </Row>
          <p className="text-xs text-slate-500">
            到「可逆矩陣」逐步播放消去過程,看每一步基本矩陣的幾何動作。
          </p>
        </ResultCard>
      )
    }
    case 'rref': {
      const trace = linalg.eliminate(flatA, n, n, -1)
      const last = trace.steps[trace.steps.length - 1]
      return (
        <ResultCard title="RREF(A) 與 rank" to="/elimination">
          <div className="flex items-center gap-3">
            <span className="font-mono text-slate-400">RREF =</span>
            <Mat grid={last.matrix} />
          </div>
          <Row label="rank / nullity(rank-nullity 對帳)">
            <span className="font-mono text-xs text-slate-300">
              rank = {trace.rank} · nullity = {n - trace.rank} · 相加 = {n}
            </span>
          </Row>
          <Row label="pivot 行 / 自由行">
            <span className="font-mono text-xs text-slate-300">
              pivot:{trace.pivotColumns.length ? trace.pivotColumns.map((c) => c + 1).join(', ') : '無'}
              {' · '}自由:{trace.freeColumns.length ? trace.freeColumns.map((c) => c + 1).join(', ') : '無'}
            </span>
          </Row>
        </ResultCard>
      )
    }
    case 'solve': {
      // 增廣 [A | b] 消去:每列尾端接上 bᵢ,augCol = n。
      const data = Array.from({ length: n }, (_, r) => [
        ...flatA.slice(r * n, r * n + n),
        vec[r],
      ]).flat()
      const trace = linalg.eliminate(data, n, n + 1, n)
      const last = trace.steps[trace.steps.length - 1]
      return (
        <ResultCard title="解 Ax = b — 增廣消去" to="/elimination">
          <div className="flex items-center gap-3">
            <span className="font-mono text-slate-400">RREF[A|b] =</span>
            <Mat grid={last.matrix} />
          </div>
          {trace.solutionKind === 'Unique' && (
            <p className="font-mono text-lg text-emerald-300">
              唯一解:x = {fmtVec(last.matrix.map((row) => row[n]))}
            </p>
          )}
          {trace.solutionKind === 'Infinite' && (
            <p className="text-sm text-sky-300">
              無限多解 —— 自由變數:{trace.freeColumns.map((c) => `x${c + 1}`).join('、')}
              (rank {trace.rank} &lt; n = {n})。
            </p>
          )}
          {trace.solutionKind === 'Inconsistent' && (
            <p className="text-sm text-rose-400">無解 —— 增廣欄冒出矛盾列(0 = 非零)。</p>
          )}
          <p className="text-xs text-slate-500">到「高斯消去」逐步播放這條消去路。</p>
        </ResultCard>
      )
    }
    case 'subspace': {
      // 2×2 專用:三大子空間基底各自由 core 算,支數 = 維度。
      const [a, b, c, d] = flatA
      const col = linalg.rangeBasis(a, b, c, d)
      const row = linalg.rowSpaceBasis(a, b, c, d)
      const nul = linalg.eigenspaceBasis2d(flatA, 0, EIGEN_EPSILON) // Null A = E₀
      const rank = linalg.rank(a, b, c, d)
      return (
        <ResultCard title="子空間基底 — Col / Row / Null" to="/nullspace">
          <Row label={`Col A 基底(dim = ${col.length / 2})`}>
            <span className="font-mono text-sm text-amber-300">{fmtBasis(col)}</span>
          </Row>
          <Row label={`Row A 基底(dim = ${row.length / 2})`}>
            <span className="font-mono text-sm text-sky-300">{fmtBasis(row)}</span>
          </Row>
          <Row label={`Null A 基底(dim = ${nul.length / 2})`}>
            <span className="font-mono text-sm text-violet-300">{fmtBasis(nul)}</span>
          </Row>
          <p className="text-xs text-slate-500">
            rank = {rank}:dim Col = dim Row(行秩 = 列秩),rank + dim Null = 2(rank-nullity)——
            三個基底各自獨立算,維度關係是定理。
          </p>
        </ResultCard>
      )
    }
    case 'eigen': {
      // 2×2 專用:λ 是不是特徵值 = A − λI 塌不塌(det 歸零)。
      const cm = linalg.characteristicMatrix2d(flatA, lambda)
      const detM = linalg.determinant([cm[0], cm[1], cm[2], cm[3]], 2)
      const eig = linalg.eigenspaceBasis2d(flatA, lambda, EIGEN_EPSILON)
      const hasReal = linalg.hasRealEigenvalues2x2(flatA)
      const isEigen = eig.length > 0
      return (
        <ResultCard title={`特徵檢定 — λ = ${fmt(lambda)} 是特徵值嗎?`} to="/eigenvalues">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-mono text-slate-400">A − λI =</span>
            <Mat
              grid={[
                [cm[0], cm[1]],
                [cm[2], cm[3]],
              ]}
            />
            <span className="font-mono text-sm text-slate-400">det = {fmt(detM)}</span>
          </div>
          {isEigen ? (
            <p className="font-mono text-sm text-emerald-300">
              ✓ λ = {fmt(lambda)} 是特徵值 —— E_λ 基底:{fmtBasis(eig)}
            </p>
          ) : (
            <p className="font-mono text-sm text-slate-400">
              ✗ 不是:det(A − λI) ≠ 0,A − λI 沒塌。
              {hasReal ? '換個 λ 試試。' : '(這顆 A 根本沒有實特徵值 —— 如純旋轉。)'}
            </p>
          )}
          <p className="text-xs text-slate-500">
            到「特徵值與特徵向量」拖 λ 滑桿,看 det(A−λI) 歸零時平行四邊形塌成線。
          </p>
        </ResultCard>
      )
    }
    case 'coords': {
      // 2×2 專用:B 的行當基底 {b₁, b₂},問 x 的 B-座標與 A 的 B-表示。
      const b1: [number, number] = [gridB[0][0], gridB[1][0]]
      const b2: [number, number] = [gridB[0][1], gridB[1][1]]
      const coord = linalg.coordinates2d(b1[0], b1[1], b2[0], b2[1], vec[0], vec[1])
      const tb = linalg.bMatrix2d(flatA, b1[0], b1[1], b2[0], b2[1])
      const degenerate = coord.length === 0
      return (
        <ResultCard title="座標與 [T]_B — 換把尺再量一次" to="/coordinates">
          {degenerate ? (
            <p className="text-sm text-rose-400">
              b₁ ∥ b₂:B 的行不是 ℝ² 的基底,座標未定義 —— 換一顆行獨立的 B。
            </p>
          ) : (
            <>
              <Row label="x 的 B-座標 [x]_B(x = c₁·b₁ + c₂·b₂)">
                <span className="font-mono text-sm text-sky-300">{fmtVec(coord)}</span>
              </Row>
              <Row label="A 的 B-表示 [T]_B = B⁻¹AB">
                {tb.length === 4 ? (
                  <Mat
                    grid={[
                      [tb[0], tb[1]],
                      [tb[2], tb[3]],
                    ]}
                  />
                ) : (
                  <span className="text-sm text-rose-400">未定義</span>
                )}
              </Row>
              <p className="text-xs text-slate-500">
                到「座標系統」拖斜格的尺、到「相似」看 [T]_B 四格變而 det 鎖死。
              </p>
            </>
          )}
        </ResultCard>
      )
    }
  }
}
