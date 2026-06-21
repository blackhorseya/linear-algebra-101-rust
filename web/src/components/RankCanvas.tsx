import { useLayoutEffect, useRef } from 'react'
import type { Linalg } from '../lib/linalg'
import {
  beginFrame,
  dot,
  drawArrow,
  drawReferenceGrid,
  HIT_PX,
  hitTest,
  label,
  makeViewport,
  strokeLine,
  useSquareSize,
  type Handle,
  type Matrix2x2,
  type Vec2,
  type Viewport,
} from '../lib/canvas'

interface RankCanvasProps {
  linalg: Linalg
  /** 2×2 矩陣 —— 左面板拖它的「列」、右面板拖它的「行」,共用同一份 state。 */
  m: Matrix2x2
  onChangeM: (m: Matrix2x2) => void
}

// /rank 頁專屬色票(結構色在 canvas.ts 的 BASE_COLORS)。
// domain 側(Row A)走 sky/cyan,codomain 側(Col A)走 amber/rose(沿 /range 的行色)。
// 兩個 span 的線 / 鋪面只在 rank 改變時出現或消失 —— 它們同進同退就是定理。
const COLORS = {
  r1: '#38bdf8', // sky-400:列向量 r₁ =(a, b)
  r2: '#22d3ee', // cyan-400:列向量 r₂ =(c, d)
  rowLine: 'rgba(56, 189, 248, 0.5)', // Row A 直線(rank 1)
  rowFill: 'rgba(56, 189, 248, 0.10)', // Row A = ℝ²(rank 2)鋪面
  a1: '#fbbf24', // amber-400:行向量 a₁ =(a, c)
  a2: '#fb7185', // rose-400:行向量 a₂ =(b, d)
  colLine: 'rgba(251, 146, 60, 0.55)', // orange-400:Col A 直線
  colFill: 'rgba(251, 146, 60, 0.10)', // Col A = ℝ² 鋪面
  origin: '#94a3b8', // slate-400:子空間 = {0} 的點
} as const

type HandleId = 'v1' | 'v2'

// span 的形狀完全由 core 的 basis 攤平長度決定(= 2·dim):
// 4 → 鋪滿平面(dim 2)、2 → 過原點的直線(dim 1)、0 → 只剩原點(dim 0)。
// 不是 JS 判維度,是直接畫 core 算出來的基底。
function drawSpan(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  size: number,
  basis: Float64Array,
  lineColor: string,
  fillColor: string,
) {
  const S = vp.toScreen
  if (basis.length === 4) {
    ctx.save()
    ctx.fillStyle = fillColor
    ctx.fillRect(0, 0, size, size)
    ctx.restore()
  } else if (basis.length === 2) {
    const [bx, by] = basis
    const t = 100 / Math.hypot(bx, by) // 基底非零(零行成不了 pivot);兩端伸出視野
    ctx.save()
    ctx.strokeStyle = lineColor
    ctx.lineWidth = 7
    ctx.lineCap = 'round'
    strokeLine(ctx, S(-t * bx, -t * by), S(t * bx, t * by))
    ctx.restore()
  } else {
    dot(ctx, S(0, 0), COLORS.origin)
  }
}

interface PanelProps {
  /** 兩支可拖箭頭(domain 是列對、codomain 是行對)。 */
  v1: Vec2
  v2: Vec2
  label1: string
  label2: string
  color1: string
  color2: string
  /** core 算出的 span 基底(rowSpaceBasis / rangeBasis),決定線 / 面 / 點。 */
  basis: Float64Array
  spanLine: string
  spanFill: string
  onDrag1: (w: Vec2) => void
  onDrag2: (w: Vec2) => void
}

// 單一 ℝ² 面板:參考網格 + span(core 的基底)+ 兩支可拖向量。domain / codomain
// 各 render 一個,差別只在傳進來的向量對與基底 —— 結構鏡像 RangeCanvas 的拖曳慣例。
function Panel({
  v1,
  v2,
  label1,
  label2,
  color1,
  color2,
  basis,
  spanLine,
  spanFill,
  onDrag1,
  onDrag2,
}: PanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef<HandleId | null>(null)
  const size = useSquareSize(containerRef)

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size <= 0) return
    const ctx = beginFrame(canvas, size)
    if (!ctx) return
    const vp = makeViewport(size)
    drawReferenceGrid(ctx, vp)
    drawSpan(ctx, vp, size, basis, spanLine, spanFill)

    const S = vp.toScreen
    const origin = S(0, 0)
    const t1 = S(v1.x, v1.y)
    const t2 = S(v2.x, v2.y)
    drawArrow(ctx, origin, t1, color1, 3)
    drawArrow(ctx, origin, t2, color2, 3)
    dot(ctx, t1, color1) // dot 保證塌縮 / 零向量時仍抓得到 handle
    dot(ctx, t2, color2)
    label(ctx, label1, t1, color1)
    label(ctx, label2, t2, color2)
  }, [v1, v2, label1, label2, color1, color2, basis, spanLine, spanFill, size])

  const buildHandles = (vp: Viewport): Handle<HandleId>[] => {
    const [x1, y1] = vp.toScreen(v1.x, v1.y)
    const [x2, y2] = vp.toScreen(v2.x, v2.y)
    return [
      { id: 'v1', sx: x1, sy: y1, priority: 0 },
      { id: 'v2', sx: x2, sy: y2, priority: 1 },
    ]
  }

  const pointerPos = (
    e: React.PointerEvent<HTMLCanvasElement>,
  ): [number, number] => {
    const rect = e.currentTarget.getBoundingClientRect()
    return [e.clientX - rect.left, e.clientY - rect.top]
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (size <= 0) return
    const [px, py] = pointerPos(e)
    const hit = hitTest(px, py, buildHandles(makeViewport(size)), HIT_PX)
    if (!hit) return
    e.currentTarget.setPointerCapture(e.pointerId)
    draggingRef.current = hit
    e.currentTarget.style.cursor = 'grabbing'
    e.preventDefault()
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (size <= 0) return
    const vp = makeViewport(size)
    const [px, py] = pointerPos(e)
    const dragging = draggingRef.current
    if (!dragging) {
      const hit = hitTest(px, py, buildHandles(vp), HIT_PX)
      e.currentTarget.style.cursor = hit ? 'grab' : 'crosshair'
      return
    }
    const [wx, wy] = vp.toWorld(px, py)
    if (dragging === 'v1') onDrag1({ x: wx, y: wy })
    else onDrag2({ x: wx, y: wy })
  }

  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current) return
    e.currentTarget.releasePointerCapture(e.pointerId)
    draggingRef.current = null
    e.currentTarget.style.cursor = 'crosshair'
  }

  return (
    <div ref={containerRef} className="aspect-square w-full">
      <canvas
        ref={canvasRef}
        className="touch-none rounded-lg border border-slate-800"
        style={{ cursor: 'crosshair' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onLostPointerCapture={() => {
          draggingRef.current = null
        }}
      />
    </div>
  )
}

export function RankCanvas({ linalg, m, onChangeM }: RankCanvasProps) {
  // 兩個 span 各由 core 獨立算:Row A 用 rowSpaceBasis(RREF 列)、Col A 用
  // rangeBasis(原始行)。它們的維度恆相等(rank(A) = rank(Aᵀ))——
  // 拖任一面板把秩拉到 1,兩條線同時出現;拉回 2,兩面同時鋪滿。
  const rowBasis = linalg.rowSpaceBasis(m.a, m.b, m.c, m.d)
  const colBasis = linalg.rangeBasis(m.a, m.b, m.c, m.d)

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-sky-300">
          domain ℝ²(輸入端)· Row A
        </p>
        <Panel
          v1={{ x: m.a, y: m.b }}
          v2={{ x: m.c, y: m.d }}
          label1="r₁"
          label2="r₂"
          color1={COLORS.r1}
          color2={COLORS.r2}
          basis={rowBasis}
          spanLine={COLORS.rowLine}
          spanFill={COLORS.rowFill}
          onDrag1={(w) => onChangeM({ ...m, a: w.x, b: w.y })} // r₁ = A 的第 1 列
          onDrag2={(w) => onChangeM({ ...m, c: w.x, d: w.y })} // r₂ = A 的第 2 列
        />
      </div>
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-amber-300">
          codomain ℝ²(輸出端)· Col A
        </p>
        <Panel
          v1={{ x: m.a, y: m.c }}
          v2={{ x: m.b, y: m.d }}
          label1="a₁"
          label2="a₂"
          color1={COLORS.a1}
          color2={COLORS.a2}
          basis={colBasis}
          spanLine={COLORS.colLine}
          spanFill={COLORS.colFill}
          onDrag1={(w) => onChangeM({ ...m, a: w.x, c: w.y })} // a₁ = A 的第 1 行
          onDrag2={(w) => onChangeM({ ...m, b: w.x, d: w.y })} // a₂ = A 的第 2 行
        />
      </div>
    </div>
  )
}
