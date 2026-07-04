import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import type { ComponentProps, ReactNode } from 'react'
import { loadLinalg } from '../lib/linalg'
import { TransformCanvas } from '../components/TransformCanvas'
import { NumberField, Status } from '../components/ui'
import { SharedMatrixNote } from '../components/SharedMatrixNote'
import { fmt } from '../lib/format'
import { patchTopLeft, toMatrix2x2 } from '../lib/shared-matrix'
import { useSharedMatrix } from '../lib/use-shared-matrix'
import { type Matrix2x2, type Vec2 } from '../lib/canvas'

// 檔名 index.tsx → 對應路徑 '/'。新增 about.tsx 就會自動產生 '/about'。
export const Route = createFileRoute('/')({
  component: Home,
})

// 總覽儀表板的 preset:一顆矩陣一種命運,切換後六個面板同時翻轉 ——
// 「所有章節都在講同一顆矩陣」的最快示範。
const PRESETS: { name: string; desc: string; m: Matrix2x2 }[] = [
  {
    name: '可逆',
    desc: 'det ≠ 0、rank 2、核 = {0}、有 A⁻¹、有實特徵值 —— 全部的燈一起亮。',
    m: { a: 2, b: 1, c: 1, d: 1 },
  },
  {
    name: '剪切',
    desc: 'det = 1(面積不變)但特徵值只剩重根 1 —— 可逆與「方向被保留多少」是兩回事。',
    m: { a: 1, b: 1, c: 0, d: 1 },
  },
  {
    name: '旋轉 90°',
    desc: '可逆、保面積,卻沒有任何實特徵方向 —— 每個向量都被轉開。',
    m: { a: 0, b: -1, c: 1, d: 0 },
  },
  {
    name: '奇異(rank 1)',
    desc: 'det = 0:值域塌成一條線、核浮現一條線、A⁻¹ 消失 —— 所有面板一起翻臉。',
    m: { a: 1, b: 2, c: 2, d: 4 },
  },
  {
    name: '零矩陣',
    desc: '一切被吸進原點:rank 0、核 = 整個平面 —— 兩個極端同時發生。',
    m: { a: 0, b: 0, c: 0, d: 0 },
  },
]

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

  // 單一真相:preset、矩陣輸入框、Canvas 拖曳、六個面板全吃這份 state。
  // 共享優先:URL 有共享矩陣就採其左上 2×2,離開首頁也帶著同一顆走。
  const { initial, publish } = useSharedMatrix()
  const [m, setM] = useState<Matrix2x2>(() => toMatrix2x2(initial) ?? PRESETS[0].m)
  const [v, setV] = useState<Vec2>({ x: 1, y: 1 })

  // 矩陣所有寫路徑的唯一出口:本地更新 + patch 左上寫回共享(v 不共享)。
  const applyM = (next: Matrix2x2) => {
    setM(next)
    publish(patchTopLeft(initial, next))
  }

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  // 全貌 = 對同一顆 A 問六個問題,每一問都是 core 的獨立計算(非彼此推導):
  // 它們的一致性(det≠0 ⟺ 可逆 ⟺ rank 滿⋯)是定理,不是前端共用布林。
  const { a, b, c, d } = m
  const av = linalg.transformPoint(a, b, c, d, v.x, v.y)
  const det = linalg.determinant([a, b, c, d], 2)
  const rank = linalg.rank(a, b, c, d)
  const nul = linalg.nullity(a, b, c, d)
  const basis = linalg.rangeBasis(a, b, c, d)
  const onto = linalg.isOnto(a, b, c, d)
  const inv = linalg.inverseMatrix(a, b, c, d)
  const report = linalg.transformationReport(a, b, c, d)
  const hasReal = linalg.hasRealEigenvalues2x2([a, b, c, d])
  const vIsZero = v.x === 0 && v.y === 0
  const onEigen = !vIsZero && linalg.areParallel(v.x, v.y, av[0], av[1])

  const detCollapsed = Math.abs(det) < 1e-9
  const activePreset = PRESETS.find(
    (p) => p.m.a === a && p.m.b === b && p.m.c === c && p.m.d === d,
  )

  return (
    <section className="space-y-12">
      <div className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight text-slate-50">
          線性代數 101 · Web
        </h1>
        <p className="text-slate-400">
          整個線性代數都在講同一件事:<span className="text-slate-200">一顆矩陣對向量做了什麼</span>。
          調整 <span className="text-amber-400">A</span>、拖動{' '}
          <span className="text-violet-400">v</span>,看變換、行列式、秩、零空間、
          可逆性、特徵方向同時連動 —— 每一格都由 Rust(WASM)獨立計算,
          它們的一致是定理。矩陣會寫進網址,點進任何一章都帶著同一顆走。
        </p>
        <SharedMatrixNote n={initial?.length} />
      </div>

      {/* 控制列:preset + A 與 v 的輸入框(與 Canvas 拖曳共用同一份 state) */}
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              onClick={() => applyM(p.m)}
              className={`rounded-md border px-3 py-1.5 text-sm transition ${
                activePreset?.name === p.name
                  ? 'border-violet-500 bg-violet-500/15 text-violet-200'
                  : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500'
              }`}
            >
              {p.name}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500">
          {activePreset?.desc ?? '自訂矩陣 —— 拖 î′、ĵ′ 或直接改數字,六個面板即時翻轉。'}
        </p>
        <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
          <div className="flex items-end gap-3">
            <span className="pb-1 font-mono text-amber-400/90">A =</span>
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="a" value={a} onChange={(x) => applyM({ ...m, a: x })} />
              <NumberField label="b" value={b} onChange={(x) => applyM({ ...m, b: x })} />
              <NumberField label="c" value={c} onChange={(x) => applyM({ ...m, c: x })} />
              <NumberField label="d" value={d} onChange={(x) => applyM({ ...m, d: x })} />
            </div>
          </div>
          <div className="flex items-end gap-3">
            <span className="pb-1 font-mono text-violet-400/90">v =</span>
            <div className="flex gap-2">
              <NumberField label="x" value={v.x} onChange={(x) => setV({ ...v, x })} />
              <NumberField label="y" value={v.y} onChange={(y) => setV({ ...v, y })} />
            </div>
          </div>
        </div>
      </div>

      {/* 主畫布 + 全貌面板:左看幾何、右看六問(lg 以上並排) */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-3">
          <TransformCanvas
            linalg={linalg}
            m={m}
            v={v}
            onChangeMatrix={applyM}
            onChangeV={setV}
          />
          <p className="text-xs text-slate-500">
            拖<span className="text-emerald-400"> î′</span>、
            <span className="text-red-400">ĵ′</span>(矩陣的行)變形整個平面;拖
            <span className="text-violet-400"> v</span> 看它的像
            <span className="text-amber-400"> A·v</span>。v 與 A·v 共線時兩箭頭加粗
            —— 你正踩在特徵方向上。
          </p>
        </div>

        <div className="grid content-start gap-3 sm:grid-cols-2">
          <Panel title="變換 A·v" to="/transform">
            <p className="font-mono text-sm text-amber-300">
              A·v = ({fmt(av[0])}, {fmt(av[1])})
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {vIsZero
                ? '0 永遠被送到 0 —— 線性變換釘住原點。'
                : onEigen
                  ? 'v 與 A·v 共線:v 是特徵向量(A·v = λv)!'
                  : '矩陣是函數:吃進 v、吐出 A·v。'}
            </p>
          </Panel>

          <Panel title="行列式" to="/determinant">
            <p
              className={`font-mono text-sm ${
                detCollapsed ? 'text-rose-400' : det < 0 ? 'text-amber-400' : 'text-emerald-400'
              }`}
            >
              det(A) = {fmt(det)}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {detCollapsed
                ? '面積歸零 —— 平面被壓扁,回不去了。'
                : det < 0
                  ? `面積 ×${fmt(Math.abs(det))},且平面翻面(定向反轉)。`
                  : `單位方格的面積被縮放 ${fmt(det)} 倍。`}
            </p>
          </Panel>

          <Panel title="值域 Col A" to="/range">
            <p className="font-mono text-sm text-slate-200">
              rank = {rank}
              {onto ? '(映成 ℝ²)' : rank === 1 ? '(塌成一條線)' : '(只剩原點)'}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {rank === 0
                ? '基底 = ∅ —— 除了原點哪裡都到不了。'
                : `基底:(${fmt(basis[0])}, ${fmt(basis[1])})${
                    basis.length === 4 ? `、(${fmt(basis[2])}, ${fmt(basis[3])})` : ''
                  }`}
            </p>
          </Panel>

          <Panel title="零空間 Null A" to="/nullspace">
            <p className="font-mono text-sm text-slate-200">
              nullity = {nul}
              <span className="text-slate-500">(rank + nullity = {rank + nul})</span>
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {nul === 0
                ? '核 = {0}:沒有非零向量被壓到原點。'
                : nul === 1
                  ? '一整條方向被 A 吃掉 —— 核是過原點的直線。'
                  : '整個平面都被吸進原點。'}
            </p>
          </Panel>

          <Panel title="可逆性" to="/invertibility">
            {inv ? (
              <p className="font-mono text-sm text-emerald-300">
                A⁻¹ = [[{fmt(inv[0])}, {fmt(inv[1])}], [{fmt(inv[2])}, {fmt(inv[3])}]]
              </p>
            ) : (
              <p className="font-mono text-sm text-rose-400">A⁻¹ 不存在</p>
            )}
            <p className="mt-1 text-xs text-slate-500">
              一對一 {report.isOneToOne ? '✓' : '✗'} · 映成 {report.isOnto ? '✓' : '✗'} · 可逆{' '}
              {report.isInvertible ? '✓' : '✗'} —— 三燈永遠同進退(2×2 方陣)。
            </p>
          </Panel>

          <Panel title="特徵方向" to="/eigenvalues">
            <p
              className={`font-mono text-sm ${hasReal ? 'text-sky-300' : 'text-rose-400'}`}
            >
              {hasReal ? '存在實特徵值 ✓' : '沒有實特徵值'}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {hasReal
                ? onEigen
                  ? 'v 現在就踩在特徵方向上 —— 只被伸縮、不被轉向。'
                  : '拖 v 找找看:哪個方向的 v 與 A·v 會共線?'
                : '每個方向都被轉開(如純旋轉)—— det(A−λI) 永不歸零。'}
            </p>
          </Panel>
        </div>
      </div>

      {/* 深入各章:照 3Blue1Brown《Essence of Linear Algebra》的敘事順序 */}
      <div className="space-y-8">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">深入各章</h2>
          <p className="text-xs text-slate-500">
            照 3Blue1Brown《Essence of Linear Algebra》的敘事走 —— 上面調好的矩陣會跟著你進每一頁。
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
            <li
              key={name}
              className="rounded-md border border-slate-800/60 px-3 py-2"
            >
              <p className="text-sm text-slate-300">{name}</p>
              <p className="text-xs text-slate-500">{desc}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

/** 全貌面板的一格:一章的縮影 + 深入連結(帶著共享矩陣走)。 */
function Panel({
  title,
  to,
  children,
}: {
  title: string
  to: ComponentProps<typeof Link>['to']
  children: ReactNode
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-medium text-slate-300">{title}</h3>
        <Link
          to={to}
          className="shrink-0 text-xs text-violet-400 transition hover:text-violet-300"
        >
          深入 →
        </Link>
      </div>
      {children}
    </div>
  )
}
