import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { CompositionCanvas } from '../components/CompositionCanvas'
import { NumberField, Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Matrix2x2, type Vec2 } from '../lib/canvas'

export const Route = createFileRoute('/composition')({
  component: CompositionDemo,
})

type Mode = 'compose' | 'inverse'

// preset 的共同形狀:逆轉換模式不帶 u(外層由 core 的 T⁻¹ 接管)。
interface Preset {
  name: string
  desc: string
  t: Matrix2x2
  u?: Matrix2x2
}

// 合成模式的代表組合:不可交換的經典對、幾何上好讀的串接、rank 傳染。
const COMPOSE_PRESETS: Preset[] = [
  {
    name: '旋轉 ∘ 反射',
    desc: '先 x 軸反射、再旋轉 90° = 對角鏡射;交換順序答案就不同 —— 合成不可交換(BA ≠ AB)。',
    u: { a: 0, b: -1, c: 1, d: 0 },
    t: { a: 1, b: 0, c: 0, d: -1 },
  },
  {
    name: '放大 ∘ 剪切',
    desc: '先剪切再等比放大:兩個可逆的合成仍可逆 —— Summary Table 三燈全亮。',
    u: { a: 1.5, b: 0, c: 0, d: 1.5 },
    t: { a: 1, b: 1, c: 0, d: 1 },
  },
  {
    name: '投影 ∘ 可逆(rank 傳染)',
    desc: 'T 可逆,但外層投影把一切拍到 x 軸 —— 鏈上有一環塌縮,整條合成就不可逆。',
    u: { a: 1, b: 0, c: 0, d: 0 },
    t: { a: 2, b: 1, c: 1, d: 1 },
  },
]

// 逆轉換模式的代表:可逆(看「復原」)與奇異(看「回不去」)。
const INVERSE_PRESETS: Preset[] = [
  {
    name: '剪切(可逆)',
    desc: '推 2 的剪切,逆是推 −2 —— 怎麼變形就怎麼推回去,網格像與參考網格重合。',
    t: { a: 1, b: 2, c: 0, d: 1 },
  },
  {
    name: '一般可逆',
    desc: 'det = 1 的可逆矩陣:T⁻¹ 由 core 的 Gauss-Jordan 當場求出。',
    t: { a: 2, b: 1, c: 1, d: 1 },
  },
  {
    name: '投影(不可逆)',
    desc: 'y 分量被抹掉(不一對一):整條鉛直線擠進同一點,「逆」不知道該回哪 —— T⁻¹ 不存在。',
    t: { a: 1, b: 0, c: 0, d: 0 },
  },
]

function CompositionDemo() {
  // WASM 模組初次載入是唯一非同步點 → 交給 Query 管 loading / error。
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:輸入框、preset、Canvas 拖曳共用這份 state。
  const [mode, setMode] = useState<Mode>('compose')
  const [t, setT] = useState<Matrix2x2>(COMPOSE_PRESETS[0].t)
  // u 在 Preset 型別上是 optional(逆轉換 preset 不帶),fallback 給 identity。
  const [u, setU] = useState<Matrix2x2>(
    COMPOSE_PRESETS[0].u ?? { a: 1, b: 0, c: 0, d: 1 },
  )
  const [x, setX] = useState<Vec2>({ x: 3, y: 1 })

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  // 逆轉換模式的外層 = T⁻¹(core 的 inverse;null = 不可逆,沒有第二步)。
  const inv = linalg.inverseMatrix(t.a, t.b, t.c, t.d)
  const outer: Matrix2x2 | null =
    mode === 'inverse'
      ? inv && { a: inv[0], b: inv[1], c: inv[2], d: inv[3] }
      : u

  // 合成矩陣 BA 與 Summary Table 全由 core 計算:合成模式報告 U∘T 的三燈
  // (合成後整體的性質),逆轉換模式報告 T 自己的三燈(可不可逆是 T 的事)。
  const composed = outer
    ? linalg.composeMatrix(outer.a, outer.b, outer.c, outer.d, t.a, t.b, t.c, t.d)
    : null
  const report =
    mode === 'compose' && composed
      ? linalg.transformationReport(composed[0], composed[1], composed[2], composed[3])
      : linalg.transformationReport(t.a, t.b, t.c, t.d)
  const reportSubject = mode === 'compose' ? 'U ∘ T' : 'T'

  const presets = mode === 'compose' ? COMPOSE_PRESETS : INVERSE_PRESETS

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          合成與可逆性
        </h1>
        <p className="text-sm text-slate-400">
          T_B ∘ T_A = T_BA:「先後施作兩個轉換」與「先乘好矩陣、一步走完」是同一個映射。
          拖 <span className="text-slate-200">x</span> 看兩步路徑(
          <span className="text-amber-400">① T</span> →{' '}
          <span className="text-rose-400">② U</span>)與一步直達(白圓環)永遠會合;
          切到逆轉換模式,U 鎖成 core 求出的 T⁻¹ —— 「變形 → 復原」回到原地。
          合成、求逆、判定全由 Rust core 的{' '}
          <code className="text-slate-300">composition</code> 模組當場計算。
        </p>
      </div>

      {/* 模式切換 + preset */}
      <div className="space-y-3">
        <div className="flex gap-2">
          {(
            [
              ['compose', '合成 U ∘ T'],
              ['inverse', '逆轉換 U = T⁻¹'],
            ] as const
          ).map(([m, text]) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`rounded-md border px-3 py-1.5 text-sm transition ${
                mode === m
                  ? 'border-violet-500/60 bg-violet-500/10 text-violet-200'
                  : 'border-slate-700 bg-slate-900 text-slate-400 hover:text-slate-200'
              }`}
            >
              {text}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          {presets.map((p) => (
            <button
              key={p.name}
              type="button"
              onClick={() => {
                setT(p.t)
                if (p.u) setU(p.u)
              }}
              title={p.desc}
              className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-300 transition hover:border-violet-500/60 hover:text-violet-200"
            >
              {p.name}
            </button>
          ))}
        </div>

        {/* 矩陣輸入:T 一律可調;U 只在合成模式開放(逆轉換模式由 core 接管) */}
        <div className="flex flex-wrap items-end gap-6">
          <div className="flex items-end gap-3">
            <span className="pb-1 font-mono text-slate-500">A(T)=</span>
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="a" value={t.a} onChange={(v) => setT({ ...t, a: v })} />
              <NumberField label="b" value={t.b} onChange={(v) => setT({ ...t, b: v })} />
              <NumberField label="c" value={t.c} onChange={(v) => setT({ ...t, c: v })} />
              <NumberField label="d" value={t.d} onChange={(v) => setT({ ...t, d: v })} />
            </div>
          </div>
          {mode === 'compose' ? (
            <div className="flex items-end gap-3">
              <span className="pb-1 font-mono text-slate-500">B(U)=</span>
              <div className="grid grid-cols-2 gap-2">
                <NumberField label="a" value={u.a} onChange={(v) => setU({ ...u, a: v })} />
                <NumberField label="b" value={u.b} onChange={(v) => setU({ ...u, b: v })} />
                <NumberField label="c" value={u.c} onChange={(v) => setU({ ...u, c: v })} />
                <NumberField label="d" value={u.d} onChange={(v) => setU({ ...u, d: v })} />
              </div>
            </div>
          ) : (
            <p className="pb-1 font-mono text-sm text-slate-400">
              A⁻¹ ={' '}
              {inv ? (
                <span className="text-violet-300">
                  [[{fmt(inv[0])}, {fmt(inv[1])}], [{fmt(inv[2])}, {fmt(inv[3])}]]
                </span>
              ) : (
                <span className="text-red-300">不存在(T 不可逆)</span>
              )}
            </p>
          )}
        </div>
      </div>

      <div className="space-y-3">
        <CompositionCanvas
          linalg={linalg}
          t={t}
          u={outer}
          outerLabel={mode === 'compose' ? 'U' : 'T⁻¹'}
          x={x}
          onChangeX={setX}
        />
        <p className="text-xs text-slate-500">
          虛線是兩步路徑:<span className="text-slate-200">x</span> 先到{' '}
          <span className="text-amber-400">T(x)</span>、再到{' '}
          <span className="text-rose-400">
            {mode === 'compose' ? 'U(T(x))' : 'T⁻¹(T(x))'}
          </span>
          ;<span className="text-slate-200">白圓環</span>是 x 左乘合成矩陣 B·A
          的一步直達 —— 永遠套住終點(T_B ∘ T_A = T_BA)。淡紫網格是合成後的
          世界{mode === 'inverse' && '(可逆時 T⁻¹·A = I,網格像與參考網格重合 —— 復原)'}。
        </p>
      </div>

      {/* 判定面板:全部由 core 計算,JS 純讀 */}
      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        {mode === 'compose' && composed && (
          <Row label="合成的標準矩陣 B·A(core 的 compose)">
            <span className="font-mono text-violet-300">
              [[{fmt(composed[0])}, {fmt(composed[1])}], [{fmt(composed[2])},{' '}
              {fmt(composed[3])}]]
            </span>
          </Row>
        )}
        {mode === 'inverse' && (
          <Row label="逆轉換 T⁻¹(Theorem 2.13:標準矩陣 = A⁻¹)">
            {inv ? (
              <span className="font-mono text-violet-300">
                [[{fmt(inv[0])}, {fmt(inv[1])}], [{fmt(inv[2])}, {fmt(inv[3])}]]
              </span>
            ) : (
              <span className="text-red-300">
                ✗ 不存在 —— 塌縮的方向回不去(NotInvertible)
              </span>
            )}
          </Row>
        )}
        <Row label={`Summary Table(${reportSubject} 的 report,core 一次點亮)`}>
          {(
            [
              [report.isOneToOne, '一對一'],
              [report.isOnto, '映成'],
              [report.isInvertible, '可逆'],
            ] as const
          ).map(([on, name]) => (
            <span key={name} className={on ? 'text-emerald-300' : 'text-red-300'}>
              {on ? '✓' : '✗'} {name}
            </span>
          ))}
        </Row>
        <p className="text-sm text-slate-400">
          {mode === 'compose' &&
            (report.isInvertible ? (
              <>
                兩個可逆的合成仍可逆,且 (U∘T)⁻¹ = T⁻¹ ∘ U⁻¹ ——
                解開的順序與穿上相反(襪子鞋子定理)。注意三燈永遠同步:2×2
                方陣的 1-1 / 映成 / 可逆是同一句話(IMT 三位一體)。
              </>
            ) : (
              <>
                鏈上只要有一環塌縮(rank 不足),整條合成就跟著塌 ——
                rank(BA) ≤ min(rank(B), rank(A)):資訊一旦在某一步丟失,
                後面的步驟再可逆也救不回來。
              </>
            ))}
          {mode === 'inverse' &&
            (report.isInvertible ? (
              <>
                T⁻¹ ∘ T = I:x 被送出去又被接回原地(白圓環套住 x 自己),
                合成後的網格與參考網格重合 —— 「可逆」就是存在一個把一切復原的轉換,
                而它的標準矩陣正是第四單元 Gauss-Jordan 求出的 A⁻¹。
              </>
            ) : (
              <>
                不一對一的轉換沒有逆:多個輸入擠進同一個輸出,「復原」不知道
                該回哪一個 —— 路徑停在 T(x),第二步不存在。這正是 Theorem 2.12:
                可逆 ⟺ 一對一且映成,缺一即死。
              </>
            ))}
        </p>
      </div>
    </section>
  )
}
