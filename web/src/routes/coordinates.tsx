import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { loadLinalg } from '../lib/linalg'
import { CoordinatesCanvas } from '../components/CoordinatesCanvas'
import { NumberField, Row, Status } from '../components/ui'
import { fmt } from '../lib/format'
import { type Vec2 } from '../lib/canvas'

export const Route = createFileRoute('/coordinates')({
  component: CoordinatesDemo,
})

interface Frame {
  b1: Vec2
  b2: Vec2
  x: Vec2
}

const SQRT1_2 = Math.SQRT1_2 // √2/2 ≈ 0.7071

// 四種基底,演不同的「換尺」。傾斜:座標歪斜;旋轉:剛體運動(保長度);重縮放:軸變長
// 短;標準:座標映射 = identity([x]_E = x)。
const PRESETS: { name: string; desc: string; frame: Frame }[] = [
  {
    name: '傾斜基底',
    desc: 'b₁=(2,1)、b₂=(−1,1):格子歪成平行四邊形。同一個 x=(3,3),在這把斜尺下座標是 (2,1) —— 沿 b₁ 走 2 步、沿 b₂ 走 1 步剛好到 x。',
    frame: { b1: { x: 2, y: 1 }, b2: { x: -1, y: 1 }, x: { x: 3, y: 3 } },
  },
  {
    name: '旋轉 45°',
    desc: '正交基底 b₁=(√2/2,√2/2)、b₂=(−√2/2,√2/2):格子只是轉了 45°、沒拉伸。換到這個 frame 是剛體運動 —— 座標向量的長度與 x 相同(練習 4 的保長度)。',
    frame: {
      b1: { x: SQRT1_2, y: SQRT1_2 },
      b2: { x: -SQRT1_2, y: SQRT1_2 },
      x: { x: 1, y: 0 },
    },
  },
  {
    name: '重縮放軸',
    desc: 'b₁=(2,0)、b₂=(0,3):軸還是水平 / 垂直,但變長了。x=(4,3) 的座標是 (2,1) —— 每個分量除以該軸的長度。',
    frame: { b1: { x: 2, y: 0 }, b2: { x: 0, y: 3 }, x: { x: 4, y: 3 } },
  },
  {
    name: '標準基底',
    desc: 'b₁=(1,0)、b₂=(0,1):斜格與灰底方格重合。此時座標映射就是 identity —— [x]_E 恰好是 x 的分量本身。',
    frame: { b1: { x: 1, y: 0 }, b2: { x: 0, y: 1 }, x: { x: 3, y: 2 } },
  },
]

function CoordinatesDemo() {
  const {
    data: linalg,
    isLoading,
    error,
  } = useQuery({ queryKey: ['linalg'], queryFn: loadLinalg })

  // 單一真相:數字輸入框、preset、canvas 拖曳共用這份 state。預設傾斜基底 ——
  // 一進來就看到「同一個點,斜尺下座標不是分量本身」。
  const [frame, setFrame] = useState<Frame>(PRESETS[0].frame)

  if (isLoading) return <Status>載入 WASM 模組中…</Status>
  if (error || !linalg) return <Status>WASM 載入失敗:{String(error)}</Status>

  const { b1, b2, x } = frame
  // 全部由 core 算:[x]_B 解 RREF、重建走 from_coordinates。座標是否定義(基底退化)
  // 也由 core 判 —— 空陣列 = b₁ ∥ b₂、不是基底。
  const coords = linalg.coordinates2d(b1.x, b1.y, b2.x, b2.y, x.x, x.y)
  const valid = coords.length === 2
  const back = valid
    ? linalg.fromCoordinates2d(b1.x, b1.y, b2.x, b2.y, coords[0], coords[1])
    : null

  return (
    <section className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-50">
          座標系統:同一個點,換把尺
        </h1>
        <p className="text-sm text-slate-400">
          一個點 <span className="text-amber-400">x</span> 的座標不是絕對的數字,而是{' '}
          <span className="text-slate-200">相對於某組基底的權重</span>。換一組基底{' '}
          <span className="text-sky-400">b₁</span>、<span className="text-violet-400">b₂</span>
          (一把斜尺),同一個 x 就有不同的座標{' '}
          <code className="text-slate-300">[x]_B = (c₁, c₂)</code> —— 也就是「沿{' '}
          <span className="text-sky-400">b₁</span> 走 c₁ 步、沿{' '}
          <span className="text-violet-400">b₂</span> 走 c₂ 步」會到 x(圖上的平行四邊形)。拖{' '}
          <span className="text-sky-400">b₁</span> / <span className="text-violet-400">b₂</span>{' '}
          換尺、拖 <span className="text-amber-400">x</span> 移點,座標由 Rust core 的{' '}
          <code className="text-slate-300">coordinates</code> 即時解出;白圈是{' '}
          <code className="text-slate-300">from_coordinates</code> 由座標重建的落點,永遠套住 x。
        </p>
      </div>

      {/* Preset:四種基底 */}
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
        {/* 基底與點的輸入:與 canvas 拖曳共用同一份 state */}
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="space-y-1">
            <span className="text-xs text-sky-400">基底 b₁</span>
            <div className="flex gap-2">
              <NumberField
                label="x"
                value={b1.x}
                onChange={(v) => setFrame({ ...frame, b1: { ...b1, x: v } })}
              />
              <NumberField
                label="y"
                value={b1.y}
                onChange={(v) => setFrame({ ...frame, b1: { ...b1, y: v } })}
              />
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-violet-400">基底 b₂</span>
            <div className="flex gap-2">
              <NumberField
                label="x"
                value={b2.x}
                onChange={(v) => setFrame({ ...frame, b2: { ...b2, x: v } })}
              />
              <NumberField
                label="y"
                value={b2.y}
                onChange={(v) => setFrame({ ...frame, b2: { ...b2, y: v } })}
              />
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-amber-400">點 x</span>
            <div className="flex gap-2">
              <NumberField
                label="x"
                value={x.x}
                onChange={(v) => setFrame({ ...frame, x: { ...x, x: v } })}
              />
              <NumberField
                label="y"
                value={x.y}
                onChange={(v) => setFrame({ ...frame, x: { ...x, y: v } })}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <CoordinatesCanvas
          linalg={linalg}
          b1={b1}
          b2={b2}
          x={x}
          onChangeB1={(v) => setFrame({ ...frame, b1: v })}
          onChangeB2={(v) => setFrame({ ...frame, b2: v })}
          onChangeX={(v) => setFrame({ ...frame, x: v })}
        />
        <p className="text-xs text-slate-500">
          灰底方格 = 標準座標;<span className="text-sky-400">藍</span> /{' '}
          <span className="text-violet-400">紫</span> 斜格 = 基底 B 這把尺。實線是「先沿 b₁、再沿
          b₂」到 x 的路徑(兩段長度就是座標 c₁、c₂),虛線補完平行四邊形。
        </p>
      </div>

      {/* 對帳面板:[x]_B 由 core 解、重建驗證雙射、與標準座標對照 */}
      <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-5">
        {valid ? (
          <>
            <Row label="標準座標(灰格)x">
              <span className="text-amber-300">
                ({fmt(x.x)}, {fmt(x.y)})
              </span>
            </Row>
            <Row label="基底座標 [x]_B = (c₁, c₂)">
              <span className="text-sky-300">
                ({fmt(coords[0])}, {fmt(coords[1])})
              </span>
            </Row>
            <div className="border-t border-slate-800 pt-3">
              <Row label="重建 c₁·b₁ + c₂·b₂">
                <span className="text-slate-100">
                  ({fmt(back![0])}, {fmt(back![1])})
                </span>
                <span className="ml-2 text-emerald-400">= x ✓</span>
              </Row>
              <p className="mt-2 text-sm text-slate-400">
                座標 = 「相對於基底的權重」:沿 <span className="text-sky-300">b₁</span> 走{' '}
                <span className="text-sky-300">{fmt(coords[0])}</span> 步、沿{' '}
                <span className="text-violet-300">b₂</span> 走{' '}
                <span className="text-violet-300">{fmt(coords[1])}</span> 步,恰好到 x。基底是
                方陣時,這就是 <code className="text-slate-300">[x]_B = B⁻¹x</code>(Theorem
                4.11);換成標準基底則 [x]_E = x。
              </p>
            </div>
          </>
        ) : (
          <p className="text-sm text-rose-300">
            b₁ ∥ b₂ —— 兩個向量共線,張不出整個平面、<strong>不是基底</strong>,座標未定義。把
            b₁ 或 b₂ 拖離共線方向(基底向量必須線性獨立)即可恢復。
          </p>
        )}
      </div>
    </section>
  )
}
