import { useLayoutEffect, useRef } from 'react'
import type { Linalg } from '../lib/linalg'
import {
  beginFrame,
  dot,
  drawArrow,
  drawReferenceGrid,
  GRID_N,
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

interface OperatorCanvasProps {
  linalg: Linalg
  /** 運算子 A(row-major a,b,c,d)。它的兩個 column 是 î′ = A·e₁、ĵ′ = A·e₂(可拖)。 */
  a: Matrix2x2
  /** 基底 B = {b₁, b₂}(可拖 —— 拖它們 = 換尺,[T]_B 就變,但 det 不變)。 */
  b1: Vec2
  b2: Vec2
  onChangeA: (m: Matrix2x2) => void
  onChangeB1: (v: Vec2) => void
  onChangeB2: (v: Vec2) => void
}

// operator 頁專屬色票(結構色在 canvas.ts 的 BASE_COLORS)。
// 暖色 = 運算子 A(它的單位方塊像);冷色斜格 = 基底 B 這把「尺」。det 的正負以填色
// 區分定向(綠 = 保持、紅 = 翻面)—— 拖基底時方塊不動(det 鎖死),拖 î′/ĵ′ 才改 A。
const COLORS = {
  latticeB1: 'rgba(56, 189, 248, 0.20)', // sky-400:平行 b₁ 的格線
  latticeB2: 'rgba(167, 139, 250, 0.20)', // violet-400:平行 b₂ 的格線
  latticeB1Axis: 'rgba(56, 189, 248, 0.55)', // 過原點的 b₁ 軸(較亮)
  latticeB2Axis: 'rgba(167, 139, 250, 0.55)', // 過原點的 b₂ 軸
  b1: '#38bdf8', // sky-400
  b2: '#a78bfa', // violet-400
  ihat: '#fbbf24', // amber-400:A·e₁(A 第一行)
  jhat: '#fb923c', // orange-400:A·e₂(A 第二行)
  fillPos: 'rgba(52, 211, 153, 0.16)', // emerald:det > 0(定向保持)
  fillNeg: 'rgba(248, 113, 113, 0.16)', // rose:det < 0(翻面)
  degenerate: '#f87171', // red-400:b₁ ∥ b₂,不是基底
} as const

type HandleId = 'ihat' | 'jhat' | 'b1' | 'b2'

// 沿基底方向鋪滿視野的斜格點陣(同 CoordinatesCanvas):平行 b₁ 的線穿過各 k·b₂、
// 平行 b₂ 的線穿過各 k·b₁;k = 0 是過原點的 B 軸,畫亮一點。
function drawLattice(ctx: CanvasRenderingContext2D, vp: Viewport, b1: Vec2, b2: Vec2) {
  const S = vp.toScreen
  const T = 60
  ctx.save()
  ctx.lineWidth = 1
  for (let k = -GRID_N; k <= GRID_N; k++) {
    ctx.strokeStyle = k === 0 ? COLORS.latticeB1Axis : COLORS.latticeB1
    strokeLine(
      ctx,
      S(k * b2.x - T * b1.x, k * b2.y - T * b1.y),
      S(k * b2.x + T * b1.x, k * b2.y + T * b1.y),
    )
    ctx.strokeStyle = k === 0 ? COLORS.latticeB2Axis : COLORS.latticeB2
    strokeLine(
      ctx,
      S(k * b1.x - T * b2.x, k * b1.y - T * b2.y),
      S(k * b1.x + T * b2.x, k * b1.y + T * b2.y),
    )
  }
  ctx.restore()
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  a: Matrix2x2,
  b1: Vec2,
  b2: Vec2,
) {
  const S = vp.toScreen
  const origin = S(0, 0)

  // 基底是否合法由 core 判:b₁ ∥ b₂ 時 bMatrix2d 回空陣列([T]_B 未定義)。
  const tb = linalg.bMatrix2d([a.a, a.b, a.c, a.d], b1.x, b1.y, b2.x, b2.y)
  const valid = tb.length === 4

  // 基底合法才畫斜格(這把「尺」);退化時只留標準方格,凸顯「沒有格子可量 [T]_B」。
  if (valid) drawLattice(ctx, vp, b1, b2)

  // 運算子 A 對單位方塊的像:平行四邊形 O → î′ → î′+ĵ′ → ĵ′。填色由 core 算的 det
  // 正負決定定向(綠保持 / 紅翻面)。這塊面積 = |det A| —— 拖基底時它紋風不動。
  const detA = linalg.determinant([a.a, a.b, a.c, a.d], 2)
  const iTip = S(a.a, a.c) // î′ = A·e₁ =(a11, a21)
  const jTip = S(a.b, a.d) // ĵ′ = A·e₂ =(a12, a22)
  const diag = S(a.a + a.b, a.c + a.d) // î′ + ĵ′
  ctx.save()
  ctx.fillStyle = detA < 0 ? COLORS.fillNeg : COLORS.fillPos
  ctx.beginPath()
  ctx.moveTo(origin[0], origin[1])
  ctx.lineTo(iTip[0], iTip[1])
  ctx.lineTo(diag[0], diag[1])
  ctx.lineTo(jTip[0], jTip[1])
  ctx.closePath()
  ctx.fill()
  ctx.restore()

  // 運算子的兩個 column(可拖):î′ = A 第一行、ĵ′ = A 第二行。
  drawArrow(ctx, origin, iTip, COLORS.ihat, 2.5)
  drawArrow(ctx, origin, jTip, COLORS.jhat, 2.5)
  dot(ctx, iTip, COLORS.ihat)
  dot(ctx, jTip, COLORS.jhat)
  label(ctx, 'î′ = A·e₁', iTip, COLORS.ihat)
  label(ctx, 'ĵ′ = A·e₂', jTip, COLORS.jhat)

  // 基底箭頭 b₁、b₂(可拖):退化時轉紅提示「不是基底」。
  const b1Tip = S(b1.x, b1.y)
  const b2Tip = S(b2.x, b2.y)
  const b1Color = valid ? COLORS.b1 : COLORS.degenerate
  const b2Color = valid ? COLORS.b2 : COLORS.degenerate
  drawArrow(ctx, origin, b1Tip, b1Color, 3)
  drawArrow(ctx, origin, b2Tip, b2Color, 3)
  dot(ctx, b1Tip, b1Color)
  dot(ctx, b2Tip, b2Color)
  label(ctx, valid ? 'b₁' : 'b₁ ∥ b₂(非基底)', b1Tip, b1Color)
  label(ctx, 'b₂', b2Tip, b2Color)
}

export function OperatorCanvas({
  linalg,
  a,
  b1,
  b2,
  onChangeA,
  onChangeB1,
  onChangeB2,
}: OperatorCanvasProps) {
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
    drawScene(ctx, vp, linalg, a, b1, b2)
  }, [linalg, a, b1, b2, size])

  // 四個 handle:î′ / ĵ′(運算子的兩行,拖 = 改 A)、b₁ / b₂(基底,拖 = 換尺)。
  // 基底優先(主互動是換尺,常與運算子箭頭重疊)。
  const buildHandles = (vp: Viewport): Handle<HandleId>[] => {
    const [ix, iy] = vp.toScreen(a.a, a.c)
    const [jx, jy] = vp.toScreen(a.b, a.d)
    const [b1x, b1y] = vp.toScreen(b1.x, b1.y)
    const [b2x, b2y] = vp.toScreen(b2.x, b2.y)
    return [
      { id: 'b1', sx: b1x, sy: b1y, priority: 0 },
      { id: 'b2', sx: b2x, sy: b2y, priority: 1 },
      { id: 'ihat', sx: ix, sy: iy, priority: 2 },
      { id: 'jhat', sx: jx, sy: jy, priority: 3 },
    ]
  }

  const pointerPos = (e: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
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
    if (dragging === 'ihat') {
      onChangeA({ ...a, a: wx, c: wy }) // î′ = A 第一行 =(a11, a21)
    } else if (dragging === 'jhat') {
      onChangeA({ ...a, b: wx, d: wy }) // ĵ′ = A 第二行 =(a12, a22)
    } else if (dragging === 'b1') {
      onChangeB1({ x: wx, y: wy })
    } else {
      onChangeB2({ x: wx, y: wy })
    }
  }

  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current) return
    e.currentTarget.releasePointerCapture(e.pointerId)
    draggingRef.current = null
    e.currentTarget.style.cursor = 'crosshair'
  }

  return (
    <div ref={containerRef} className="aspect-square w-full max-w-xl">
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
