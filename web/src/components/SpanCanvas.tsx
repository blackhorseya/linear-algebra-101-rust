import { useLayoutEffect, useRef } from 'react'
import type { Linalg } from '../lib/linalg'
import {
  beginFrame,
  dot,
  drawArrow,
  drawImageSegment,
  drawReferenceGrid,
  GRID_N,
  HIT_PX,
  hitTest,
  label,
  makeViewport,
  ORIGIN_EPS,
  strokeLine,
  useSquareSize,
  WORLD_HALF,
  type Handle,
  type Vec2,
  type Viewport,
} from '../lib/canvas'

interface SpanCanvasProps {
  linalg: Linalg
  v: Vec2
  w: Vec2
  onChangeV: (v: Vec2) => void
  onChangeW: (w: Vec2) => void
}

// span 專屬色票
const COLORS = {
  grid: '#2f5d63', // 張成網格(teal 偏暗)
  gridAxis: '#4d8a93', // v / w 方向那兩條線(亮一點)
  line: '#5eead4', // 相依時的張成直線(teal-300)
  v: '#a78bfa', // violet-400
  w: '#22d3ee', // cyan-400
} as const

function drawSpan(ctx: CanvasRenderingContext2D, vp: Viewport, linalg: Linalg, v: Vec2, w: Vec2) {
  const S = vp.toScreen
  const origin = S(0, 0)
  const vLen = Math.hypot(v.x, v.y)
  const wLen = Math.hypot(w.x, w.y)
  const bothZero = vLen < ORIGIN_EPS && wLen < ORIGIN_EPS
  // 相依 = 兩向量平行(線性相依);span 因此只塌成一條線(或一點)。
  const parallel = !bothZero && linalg.areParallel(v.x, v.y, w.x, w.y)

  if (!bothZero && !parallel) {
    // 線性獨立 → 張成整個平面:畫由 v、w 張出的網格(= 矩陣 [v w] 作用在標準網格的像)。
    const m = { a: v.x, b: w.x, c: v.y, d: w.y } // 兩欄就是 v 和 w
    ctx.save()
    for (let k = -GRID_N; k <= GRID_N; k++) {
      ctx.strokeStyle = k === 0 ? COLORS.gridAxis : COLORS.grid
      ctx.lineWidth = k === 0 ? 1.8 : 1
      drawImageSegment(ctx, linalg.transformPoint, m, vp, k, -GRID_N, k, GRID_N)
      drawImageSegment(ctx, linalg.transformPoint, m, vp, -GRID_N, k, GRID_N, k)
    }
    ctx.restore()
  } else if (parallel) {
    // 線性相依 → 張成一條過原點的直線(沿較長的非零向量方向)。
    const useV = vLen >= wLen
    const len = useV ? vLen : wLen
    const ux = (useV ? v.x : w.x) / len
    const uy = (useV ? v.y : w.y) / len
    ctx.save()
    ctx.strokeStyle = COLORS.line
    ctx.lineWidth = 2.5
    strokeLine(ctx, S(-WORLD_HALF * ux, -WORLD_HALF * uy), S(WORLD_HALF * ux, WORLD_HALF * uy))
    ctx.restore()
  }

  // 兩個向量箭頭(端點即拖曳 handle)
  const vTip = S(v.x, v.y)
  const wTip = S(w.x, w.y)
  if (vLen >= ORIGIN_EPS) {
    drawArrow(ctx, origin, vTip, COLORS.v, 2.5)
    label(ctx, 'v', vTip, COLORS.v)
  }
  if (wLen >= ORIGIN_EPS) {
    drawArrow(ctx, origin, wTip, COLORS.w, 2.5)
    label(ctx, 'w', wTip, COLORS.w)
  }
  dot(ctx, vTip, COLORS.v)
  dot(ctx, wTip, COLORS.w)
}

export function SpanCanvas({ linalg, v, w, onChangeV, onChangeW }: SpanCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef<'v' | 'w' | null>(null)
  const size = useSquareSize(containerRef)

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size <= 0) return
    const ctx = beginFrame(canvas, size)
    if (!ctx) return
    const vp = makeViewport(size)
    drawReferenceGrid(ctx, vp)
    drawSpan(ctx, vp, linalg, v, w)
  }, [linalg, v, w, size])

  const buildHandles = (vp: Viewport): Handle<'v' | 'w'>[] => {
    const [vx, vy] = vp.toScreen(v.x, v.y)
    const [wx, wy] = vp.toScreen(w.x, w.y)
    return [
      { id: 'v', sx: vx, sy: vy, priority: 0 },
      { id: 'w', sx: wx, sy: wy, priority: 1 },
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
    if (!draggingRef.current) {
      const hit = hitTest(px, py, buildHandles(vp), HIT_PX)
      e.currentTarget.style.cursor = hit ? 'grab' : 'crosshair'
      return
    }
    const [rx, ry] = vp.toWorld(px, py)
    const x = rx === 0 ? 0 : rx
    const y = ry === 0 ? 0 : ry
    if (draggingRef.current === 'v') onChangeV({ x, y })
    else onChangeW({ x, y })
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
