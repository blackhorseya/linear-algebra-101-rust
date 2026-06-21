import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { OperatorCanvas } from '../components/OperatorCanvas'
import { NumberField, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Matrix2x2, type Vec2 } from '../lib/canvas'

export const Route = createFileRoute('/operator')({
  component: OperatorDemo,
})

interface Frame {
  a: Matrix2x2 // 運算子 A(row-major a,b,c,d)
  b1: Vec2
  b2: Vec2
}

const COS30 = Math.sqrt(3) / 2 // ≈ 0.866

// 每個 preset 同時換「運算子 A」與「基底 B」,演不同的相似情境;det 都標在說明裡。
const PRESETS: { name: string; desc: string; frame: Frame }[] = [
  {
    name: '旋轉 90° · 傾斜尺',
    desc: 'A 是旋轉 90°(det = 1)。換到傾斜基底 B={(2,1),(−1,1)},[T]_B 的四格全變了,但有號面積 det 仍是 1 —— 同一個旋轉,只是用斜尺描述。',
    frame: { a: { a: 0, b: -1, c: 1, d: 0 }, b1: { x: 2, y: 1 }, b2: { x: -1, y: 1 } },
  },
  {
    name: '反射 · 對角尺',
    desc: 'A 是 x 軸反射(det = −1,翻面)。在對角基底 B={(1,1),(1,−1)} 下,[T]_B 變成交換矩陣 [[0,1],[1,0]] —— 與 A 不相等卻相似,det 同為 −1。',
    frame: { a: { a: 1, b: 0, c: 0, d: -1 }, b1: { x: 1, y: 1 }, b2: { x: 1, y: -1 } },
  },
  {
    name: '剪切 · 標準尺',
    desc: 'A 是水平剪切(det = 1)。基底就是標準軸 B={e₁,e₂},此時「換座標」是 identity,故 [T]_B = A —— 標準基底下表示不變。',
    frame: { a: { a: 1, b: 1, c: 0, d: 1 }, b1: { x: 1, y: 0 }, b2: { x: 0, y: 1 } },
  },
  {
    name: '縮放 · 旋轉尺',
    desc: 'A 是非均勻縮放 diag(2,3)(det = 6)。換到旋轉 30° 的正交基底,[T]_B 不再是對角矩陣,但 det 仍鎖在 6。',
    frame: {
      a: { a: 2, b: 0, c: 0, d: 3 },
      b1: { x: COS30, y: 0.5 },
      b2: { x: -0.5, y: COS30 },
    },
  },
]

/** 2×2 矩陣顯示(row-major `[m11,m12,m21,m22]`)。 */
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

function OperatorDemo() {
  const { data: linalg, isLoading, error } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:輸入框、preset、canvas 拖曳共用這份 state。預設「旋轉 90° · 傾斜尺」——
  // 一進來就看到「同一個運算子,斜尺下矩陣不一樣,但 det 不變」。
  const [frame, setFrame] = useState<Frame>(PRESETS[0].frame)

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  const { a, b1, b2 } = frame
  const aFlat = [a.a, a.b, a.c, a.d]
  // 全部由 core 算:[T]_B 走 b_matrix(對基底取像、求座標、組 column),det 走既有
  // determinant binding。基底退化(b₁ ∥ b₂)時 b_matrix 回空 —— [T]_B 未定義。
  const tb = linalg.bMatrix2d(aFlat, b1.x, b1.y, b2.x, b2.y)
  const valid = tb.length === 4
  const detA = linalg.determinant(aFlat, 2)
  const detTB = valid ? linalg.determinant([...tb], 2) : null
  const detLocked = detTB !== null && Math.abs(detA - detTB) < 1e-6
  const equalsA = valid && tb.every((t, i) => Math.abs(t - aFlat[i]) < 1e-6)

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          相似:同一個運算子,換把尺就換個矩陣
        </h1>
        <p className="text-sm text-slate-400">
          一個線性運算子 <span className="text-slate-200">T</span>(這裡用它的標準矩陣{' '}
          <span className="text-amber-400">A</span> 表示)是抽象的「動作」;把它放到不同基底{' '}
          <span className="text-sky-400">b₁</span>、<span className="text-violet-400">b₂</span>{' '}
          底下量,就得到不同的矩陣{' '}
          <code className="text-slate-300">[T]_B = B⁻¹AB</code>(Theorem 4.12)。這些矩陣彼此{' '}
          <span className="text-slate-200">相似</span> —— 描述變了,運算子沒變。最直接的見證:{' '}
          <code className="text-slate-300">det[T]_B</code> 永遠等於{' '}
          <code className="text-slate-300">det A</code>(圖上那塊平行四邊形的有號面積)。拖{' '}
          <span className="text-sky-400">b₁</span> / <span className="text-violet-400">b₂</span>{' '}
          換尺看 <code className="text-slate-300">[T]_B</code> 變、面積不動;拖{' '}
          <span className="text-amber-400">î′</span> / <span className="text-orange-400">ĵ′</span>{' '}
          改 A 本身。<code className="text-slate-300">[T]_B</code> 由 Rust core 的{' '}
          <code className="text-slate-300">b_matrix</code> 計算。
        </p>
      </div>

      {/* Preset:四種運算子 × 基底組合 */}
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              onClick={() => setFrame(p.frame)}
              className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-300 transition hover:border-violet-500/60 hover:text-violet-200"
            >
              {p.name}
            </button>
          ))}
        </div>
        {/* 輸入框:與 canvas 拖曳共用同一份 state */}
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
          <div className="space-y-1">
            <span className="text-xs text-sky-400">基底 b₁</span>
            <div className="flex gap-2">
              <NumberField label="x" value={b1.x} onChange={(v) => setFrame({ ...frame, b1: { ...b1, x: v } })} />
              <NumberField label="y" value={b1.y} onChange={(v) => setFrame({ ...frame, b1: { ...b1, y: v } })} />
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-violet-400">基底 b₂</span>
            <div className="flex gap-2">
              <NumberField label="x" value={b2.x} onChange={(v) => setFrame({ ...frame, b2: { ...b2, x: v } })} />
              <NumberField label="y" value={b2.y} onChange={(v) => setFrame({ ...frame, b2: { ...b2, y: v } })} />
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <OperatorCanvas
          linalg={linalg}
          a={a}
          b1={b1}
          b2={b2}
          onChangeA={(m) => setFrame({ ...frame, a: m })}
          onChangeB1={(v) => setFrame({ ...frame, b1: v })}
          onChangeB2={(v) => setFrame({ ...frame, b2: v })}
        />
        <p className="text-xs text-slate-500">
          灰底方格 = 標準座標;<span className="text-sky-400">藍</span> /{' '}
          <span className="text-violet-400">紫</span> 斜格 = 基底 B 這把尺;暖色平行四邊形 ={' '}
          <span className="text-amber-400">A</span> 把單位方塊送到的像(面積 = |det A|,
          <span className="text-emerald-400">綠</span>保持定向 /{' '}
          <span className="text-rose-400">紅</span>翻面)。拖基底時這塊面積紋風不動。
        </p>
      </div>

      {/* 對帳面板:A 與 [T]_B 並排,det 鎖死相等 = 相似不變量 */}
      <div className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        {valid ? (
          <>
            <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
              <div className="space-y-1">
                <span className="text-xs text-amber-400">標準矩陣 A</span>
                <div className="flex items-center gap-3">
                  <Mat2 m={aFlat} accent="text-amber-200" />
                  <span className="font-mono text-sm text-slate-400">det = {fmt(detA)}</span>
                </div>
              </div>
              <div className="space-y-1">
                <span className="text-xs text-slate-300">
                  B-矩陣 [T]_B = B⁻¹AB {equalsA && <span className="text-emerald-400">(= A)</span>}
                </span>
                <div className="flex items-center gap-3">
                  <Mat2 m={[...tb]} accent="text-slate-100" />
                  <span className="font-mono text-sm text-slate-400">det = {fmt(detTB!)}</span>
                </div>
              </div>
            </div>
            <div className="border-t border-slate-800 pt-3 text-sm">
              {detLocked ? (
                <p className="text-slate-400">
                  <span className="text-emerald-400">det [T]_B = det A ✓</span> —— 相似矩陣共享
                  行列式(也共享跡、特徵值)。換尺改變的是<span className="text-slate-200">表示</span>,
                  不是<span className="text-slate-200">運算子本身</span>:拖 b₁ / b₂ 看 [T]_B 的四格
                  怎麼變,det 永遠釘在 {fmt(detA)}。
                </p>
              ) : (
                <p className="text-rose-300">det 對帳異常(數值誤差?):det A = {fmt(detA)}、det [T]_B = {fmt(detTB!)}</p>
              )}
            </div>
          </>
        ) : (
          <p className="text-sm text-rose-300">
            b₁ ∥ b₂ —— 兩個向量共線,張不出整個平面、<strong>不是基底</strong>,[T]_B 未定義。把
            b₁ 或 b₂ 拖離共線方向(基底向量必須線性獨立)即可恢復。
          </p>
        )}
      </div>
    </section>
  )
}
