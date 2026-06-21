import { useLayoutEffect, useRef } from 'react'
import type { Linalg } from '../lib/linalg'
import {
  beginFrame,
  dot,
  drawArrow,
  drawReferenceGrid,
  drawRing,
  GRID_N,
  HIT_PX,
  hitTest,
  label,
  makeViewport,
  strokeLine,
  useSquareSize,
  type Handle,
  type Vec2,
  type Viewport,
} from '../lib/canvas'

interface CoordinatesCanvasProps {
  linalg: Linalg
  /** 基底向量 b₁、b₂(可在圖上直接拖 —— 拖它們 = 換尺,同一個點座標就變)。 */
  b1: Vec2
  b2: Vec2
  /** 被測量的點 x(可拖移點)。 */
  x: Vec2
  onChangeB1: (v: Vec2) => void
  onChangeB2: (v: Vec2) => void
  onChangeX: (v: Vec2) => void
}

// coordinates 頁專屬色票(結構色在 canvas.ts 的 BASE_COLORS)。
// b₁ / b₂ 的斜格點陣 = 基底 B 這把「尺」;step 與基底同色,演出「沿 b₁ 走 c₁ 步、
// 沿 b₂ 走 c₂ 步」;x 用亮色凸顯;ring 是 core 重建落點(套住 x = 雙射兩路會合)。
const COLORS = {
  latticeB1: 'rgba(56, 189, 248, 0.22)', // sky-400:平行 b₁ 的格線
  latticeB2: 'rgba(167, 139, 250, 0.22)', // violet-400:平行 b₂ 的格線
  latticeB1Axis: 'rgba(56, 189, 248, 0.55)', // 過原點的 b₁ 軸(較亮)
  latticeB2Axis: 'rgba(167, 139, 250, 0.55)', // 過原點的 b₂ 軸
  b1: '#38bdf8', // sky-400
  b2: '#a78bfa', // violet-400
  x: '#fbbf24', // amber-400:目標點
  ring: '#f8fafc', // slate-50:from_coordinates 重建落點
  degenerate: '#f87171', // red-400:b₁ ∥ b₂,不是基底
} as const

type HandleId = 'b1' | 'b2' | 'x'

// 沿基底方向鋪滿視野的斜格點陣:平行 b₁ 的線(穿過各 k·b₂)與平行 b₂ 的線(穿過各 k·b₁)。
// T 取夠大讓線段兩端伸出 ±WORLD_HALF 視窗;k = 0 的線是過原點的 B 軸,畫亮一點。
function drawLattice(ctx: CanvasRenderingContext2D, vp: Viewport, b1: Vec2, b2: Vec2) {
  const S = vp.toScreen
  const T = 60
  ctx.save()
  ctx.lineWidth = 1
  for (let k = -GRID_N; k <= GRID_N; k++) {
    // 平行 b₁、穿過 k·b₂ 的線
    ctx.strokeStyle = k === 0 ? COLORS.latticeB1Axis : COLORS.latticeB1
    strokeLine(
      ctx,
      S(k * b2.x - T * b1.x, k * b2.y - T * b1.y),
      S(k * b2.x + T * b1.x, k * b2.y + T * b1.y),
    )
    // 平行 b₂、穿過 k·b₁ 的線
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
  b1: Vec2,
  b2: Vec2,
  x: Vec2,
) {
  const S = vp.toScreen
  const origin = S(0, 0)

  // 座標是否定義由 core 判:b₁ ∥ b₂ 時 coordinates_2d 回空陣列(非基底)。
  const coords = linalg.coordinates2d(b1.x, b1.y, b2.x, b2.y, x.x, x.y)
  const valid = coords.length === 2

  // 基底有效才畫斜格點陣(這把「尺」);退化時只留標準方格,凸顯「沒有格子可讀座標」。
  if (valid) drawLattice(ctx, vp, b1, b2)

  if (valid) {
    const [c1, c2] = coords
    const stepA = S(c1 * b1.x, c1 * b1.y) // 原點 → c₁·b₁
    const xTip = S(x.x, x.y)
    const stepB = S(c2 * b2.x, c2 * b2.y) // 原點 → c₂·b₂

    // 平行四邊形分解 x = c₁b₁ + c₂b₂:實線走「先沿 b₁ 走 c₁ 步、再沿 b₂ 走 c₂ 步」,
    // 虛線補完另兩邊(先 b₂ 再 b₁),兩條路都到 x —— 座標就是這兩個步數。
    ctx.save()
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    ctx.strokeStyle = COLORS.b1
    strokeLine(ctx, origin, stepA)
    ctx.strokeStyle = COLORS.b2
    strokeLine(ctx, stepA, xTip)
    ctx.setLineDash([5, 5])
    ctx.lineWidth = 1.5
    ctx.strokeStyle = COLORS.b2
    strokeLine(ctx, origin, stepB)
    ctx.strokeStyle = COLORS.b1
    strokeLine(ctx, stepB, xTip)
    ctx.restore()

    // core 由座標重建的落點:圓環必套住拖曳中的 x(from_coordinates ∘ coordinates = id)。
    const back = linalg.fromCoordinates2d(b1.x, b1.y, b2.x, b2.y, c1, c2)
    drawRing(ctx, S(back[0], back[1]), COLORS.ring)
  }

  // 基底箭頭 b₁、b₂(可拖):dot 保證退化 / 短向量時仍抓得到。
  const b1Tip = S(b1.x, b1.y)
  const b2Tip = S(b2.x, b2.y)
  drawArrow(ctx, origin, b1Tip, COLORS.b1, 3)
  drawArrow(ctx, origin, b2Tip, COLORS.b2, 3)
  dot(ctx, b1Tip, COLORS.b1)
  dot(ctx, b2Tip, COLORS.b2)
  label(ctx, 'b₁', b1Tip, COLORS.b1)
  label(ctx, 'b₂', b2Tip, COLORS.b2)

  // 目標點 x:座標標在點旁。退化時 x 轉紅、提示「不是基底,座標未定義」。
  const xTip = S(x.x, x.y)
  const xColor = valid ? COLORS.x : COLORS.degenerate
  dot(ctx, xTip, xColor)
  if (valid) {
    label(ctx, `x · [x]_B = (${fmtNum(coords[0])}, ${fmtNum(coords[1])})`, xTip, xColor)
  } else {
    label(ctx, 'x · b₁ ∥ b₂(非基底)', xTip, xColor)
  }
}

/** canvas 標籤用的短格式(最多 2 位,去尾零)。 */
function fmtNum(n: number): string {
  return Number(n.toFixed(2)).toString()
}

export function CoordinatesCanvas({
  linalg,
  b1,
  b2,
  x,
  onChangeB1,
  onChangeB2,
  onChangeX,
}: CoordinatesCanvasProps) {
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
    drawScene(ctx, vp, linalg, b1, b2, x)
  }, [linalg, b1, b2, x, size])

  // 三個 handle 都可拖:b₁、b₂ 是基底(拖 = 換尺),x 是目標點。x 優先(常與基底重疊)。
  const buildHandles = (vp: Viewport): Handle<HandleId>[] => {
    const [b1x, b1y] = vp.toScreen(b1.x, b1.y)
    const [b2x, b2y] = vp.toScreen(b2.x, b2.y)
    const [xx, xy] = vp.toScreen(x.x, x.y)
    return [
      { id: 'x', sx: xx, sy: xy, priority: 0 },
      { id: 'b1', sx: b1x, sy: b1y, priority: 1 },
      { id: 'b2', sx: b2x, sy: b2y, priority: 2 },
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
    if (dragging === 'x') {
      onChangeX({ x: wx, y: wy })
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
