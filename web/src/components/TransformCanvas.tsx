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
  type Matrix2x2,
  type Vec2,
  type Viewport,
} from '../lib/canvas'

interface TransformCanvasProps {
  linalg: Linalg
  m: Matrix2x2
  v: Vec2
  onChangeMatrix: (m: Matrix2x2) => void
  onChangeV: (v: Vec2) => void
}

// transform 專屬色票(結構色在 canvas.ts 的 BASE_COLORS)
const COLORS = {
  tGrid: '#4c4368', // 變換後網格(violet 偏暗)
  tGridAxis: '#7c6db0', // 變換後座標軸的像(亮一點)
  iHat: '#34d399', // emerald-400
  jHat: '#f87171', // red-400
  v: '#a78bfa', // violet-400
  av: '#fbbf24', // amber-400
  eigen: '#34d399', // emerald 虛線(特徵方向)
  squarePos: 'rgba(167,139,250,0.12)', // det>0:violet(定向不變)
  squareNeg: 'rgba(251,191,36,0.12)', // det<0:amber(平面翻面)
} as const

function drawTransform(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  m: Matrix2x2,
  v: Vec2,
) {
  const S = vp.toScreen
  const origin = S(0, 0)

  // 變換後網格(每條線的像都靠 WASM transformPoint 算端點)
  ctx.save()
  for (let k = -GRID_N; k <= GRID_N; k++) {
    ctx.strokeStyle = k === 0 ? COLORS.tGridAxis : COLORS.tGrid
    ctx.lineWidth = k === 0 ? 1.8 : 1
    drawImageSegment(ctx, linalg.transformPoint, m, vp, k, -GRID_N, k, GRID_N) // x=k 的像
    drawImageSegment(ctx, linalg.transformPoint, m, vp, -GRID_N, k, GRID_N, k) // y=k 的像
  }
  ctx.restore()

  // 單位方格的像(平行四邊形)。遠角 î'+ĵ' 由 WASM 取得,JS 不做加法。
  // 面積 = |det|;det<0 代表翻面,用顏色區分(det 也由 WASM 算)。
  const iTip = S(m.a, m.c)
  const jTip = S(m.b, m.d)
  const farW = linalg.transformPoint(m.a, m.b, m.c, m.d, 1, 1)
  if ([m.a, m.c, m.b, m.d, farW[0], farW[1]].every(Number.isFinite)) {
    const far = S(farW[0], farW[1])
    const det = linalg.determinant(m.a, m.b, m.c, m.d)
    ctx.save()
    ctx.fillStyle = det < 0 ? COLORS.squareNeg : COLORS.squarePos
    ctx.beginPath()
    ctx.moveTo(origin[0], origin[1])
    ctx.lineTo(iTip[0], iTip[1])
    ctx.lineTo(far[0], far[1])
    ctx.lineTo(jTip[0], jTip[1])
    ctx.closePath()
    ctx.fill()
    ctx.restore()
  }

  // 基底箭頭 î'、ĵ'(端點即拖曳 handle)
  drawArrow(ctx, origin, iTip, COLORS.iHat, 2.5)
  drawArrow(ctx, origin, jTip, COLORS.jHat, 2.5)
  dot(ctx, iTip, COLORS.iHat)
  dot(ctx, jTip, COLORS.jHat)
  label(ctx, "î'", iTip, COLORS.iHat)
  label(ctx, "ĵ'", jTip, COLORS.jHat)

  // v 與 A·v(A·v 由 WASM 算;平行則高亮特徵方向)
  const vLen = Math.hypot(v.x, v.y)
  const vTip = S(v.x, v.y)
  const av = linalg.transformPoint(m.a, m.b, m.c, m.d, v.x, v.y)
  const avTip = S(av[0], av[1])
  const parallel = vLen >= ORIGIN_EPS && linalg.areParallel(v.x, v.y, av[0], av[1])

  if (parallel) {
    ctx.save()
    ctx.strokeStyle = COLORS.eigen
    ctx.globalAlpha = 0.4
    ctx.setLineDash([6, 5])
    const ux = v.x / vLen
    const uy = v.y / vLen
    strokeLine(ctx, S(-WORLD_HALF * ux, -WORLD_HALF * uy), S(WORLD_HALF * ux, WORLD_HALF * uy))
    ctx.restore()
  }

  if (vLen >= ORIGIN_EPS) {
    drawArrow(ctx, origin, avTip, COLORS.av, parallel ? 4 : 2.5)
    drawArrow(ctx, origin, vTip, COLORS.v, parallel ? 4 : 2.5)
    label(ctx, 'A·v', avTip, COLORS.av)
    label(ctx, 'v', vTip, COLORS.v)
  }
  dot(ctx, vTip, COLORS.v) // 近原點時只剩這顆淡點,仍可被抓回來
}

export function TransformCanvas({ linalg, m, v, onChangeMatrix, onChangeV }: TransformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef<'iHat' | 'jHat' | 'v' | null>(null)
  const size = useSquareSize(containerRef)

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size <= 0) return
    const ctx = beginFrame(canvas, size)
    if (!ctx) return
    const vp = makeViewport(size)
    drawReferenceGrid(ctx, vp)
    drawTransform(ctx, vp, linalg, m, v)
  }, [linalg, m, v, size])

  // handlers 由 React 每次 render 重建,必看到最新 m/v(無 stale closure)
  const buildHandles = (vp: Viewport): Handle<'iHat' | 'jHat' | 'v'>[] => {
    const [vx, vy] = vp.toScreen(v.x, v.y)
    const [ix, iy] = vp.toScreen(m.a, m.c)
    const [jx, jy] = vp.toScreen(m.b, m.d)
    return [
      { id: 'v', sx: vx, sy: vy, priority: 0 },
      { id: 'iHat', sx: ix, sy: iy, priority: 1 },
      { id: 'jHat', sx: jx, sy: jy, priority: 2 },
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
    const [wx, wy] = vp.toWorld(px, py)
    const x = wx === 0 ? 0 : wx
    const y = wy === 0 ? 0 : wy
    switch (draggingRef.current) {
      case 'v':
        onChangeV({ x, y })
        break
      case 'iHat':
        onChangeMatrix({ ...m, a: x, c: y })
        break
      case 'jHat':
        onChangeMatrix({ ...m, b: x, d: y })
        break
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
