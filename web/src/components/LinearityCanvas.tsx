import { useLayoutEffect, useRef } from 'react'
import type { Linalg } from '../lib/linalg'
import {
  beginFrame,
  dot,
  drawArrow,
  drawImageSegment,
  drawReferenceGrid,
  drawRing,
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
  type Pt,
  type Vec2,
  type Viewport,
} from '../lib/canvas'

/** 守恆律展示模式:加法 T(u+v) = T(u)+T(v),或純量乘 T(ku) = k·T(u)。 */
export type LinearityMode = 'add' | 'scale'

interface LinearityCanvasProps {
  linalg: Linalg
  m: Matrix2x2
  u: Vec2
  v: Vec2
  k: number
  mode: LinearityMode
  onChangeU: (u: Vec2) => void
  onChangeV: (v: Vec2) => void
}

// linearity 專屬色票(結構色在 canvas.ts 的 BASE_COLORS)
const COLORS = {
  tGrid: '#3b3554', // 變換後網格(比 transform 頁更暗:主角是向量,不是網格)
  tGridAxis: '#5b5286', // 變換後座標軸的像
  u: '#a78bfa', // violet-400
  v: '#38bdf8', // sky-400
  sum: '#e2e8f0', // slate-200(u+v / k·u:輸入側的「合成」向量)
  tu: '#fbbf24', // amber-400(T(u))
  tv: '#fb7185', // rose-400(T(v))
  tImage: '#34d399', // emerald-400(T(u+v) / T(k·u):左邊先合成、再送進 T)
  guide: '#64748b', // slate-500(平行四邊形虛線導引)
  ring: '#f8fafc', // slate-50(T(u)+T(v) / k·T(u):右邊各自過 T、再合成 —— 守恆律 = 圓環套住箭頭)
} as const

/** 虛線輔助線(平行四邊形的另外兩邊)。 */
function dashedLine(ctx: CanvasRenderingContext2D, p0: Pt, p1: Pt, color: string) {
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = 1
  ctx.setLineDash([5, 4])
  strokeLine(ctx, p0, p1)
  ctx.restore()
}

function drawLinearity(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  m: Matrix2x2,
  u: Vec2,
  v: Vec2,
  k: number,
  mode: LinearityMode,
) {
  const S = vp.toScreen
  const origin = S(0, 0)
  // T(x) 的唯一算法:WASM 的 transformPoint(core 的 multiply_vector)。
  const T = (x: number, y: number) => linalg.transformPoint(m.a, m.b, m.c, m.d, x, y)
  const finite = (p: Float64Array) => [p[0], p[1]].every(Number.isFinite)

  // 變換後網格(淡):shear 斜掉、投影塌成一條線,給「整個平面怎麼動」的背景感。
  ctx.save()
  for (let g = -GRID_N; g <= GRID_N; g++) {
    ctx.strokeStyle = g === 0 ? COLORS.tGridAxis : COLORS.tGrid
    ctx.lineWidth = g === 0 ? 1.6 : 1
    drawImageSegment(ctx, linalg.transformPoint, m, vp, g, -GRID_N, g, GRID_N)
    drawImageSegment(ctx, linalg.transformPoint, m, vp, -GRID_N, g, GRID_N, g)
  }
  ctx.restore()

  const uTip = S(u.x, u.y)
  const tu = T(u.x, u.y)

  if (mode === 'add') {
    // ---- 輸入側:u、v 與平行四邊形對角線 u+v(u+v 由 WASM 算,JS 不做加法)----
    const sum = linalg.addVectors(u.x, u.y, v.x, v.y)
    const vTip = S(v.x, v.y)
    const sumTip = S(sum[0], sum[1])
    dashedLine(ctx, uTip, sumTip, COLORS.guide)
    dashedLine(ctx, vTip, sumTip, COLORS.guide)
    drawArrow(ctx, origin, sumTip, COLORS.sum, 2)
    drawArrow(ctx, origin, uTip, COLORS.u, 2.5)
    drawArrow(ctx, origin, vTip, COLORS.v, 2.5)
    label(ctx, 'u', uTip, COLORS.u)
    label(ctx, 'v', vTip, COLORS.v)
    label(ctx, 'u+v', sumTip, COLORS.sum)

    // ---- 影像側:兩條獨立計算路徑 ----
    // 路徑一(emerald 箭頭):先合成 u+v,再送進 T → T(u+v)。
    // 路徑二(白色圓環):u、v 各自過 T,再合成 → T(u)+T(v)。
    // 兩者重合 = 加法守恆;每一步都由 WASM 算,JS 只畫圖。
    const tv = T(v.x, v.y)
    const tsum = T(sum[0], sum[1])
    const tutv = linalg.addVectors(tu[0], tu[1], tv[0], tv[1])
    if ([tu, tv, tsum, tutv].every(finite)) {
      const tuTip = S(tu[0], tu[1])
      const tvTip = S(tv[0], tv[1])
      const tsumTip = S(tsum[0], tsum[1])
      dashedLine(ctx, tuTip, S(tutv[0], tutv[1]), COLORS.guide)
      dashedLine(ctx, tvTip, S(tutv[0], tutv[1]), COLORS.guide)
      drawArrow(ctx, origin, tuTip, COLORS.tu, 2.5)
      drawArrow(ctx, origin, tvTip, COLORS.tv, 2.5)
      drawArrow(ctx, origin, tsumTip, COLORS.tImage, 3)
      drawRing(ctx, S(tutv[0], tutv[1]), COLORS.ring)
      label(ctx, 'T(u)', tuTip, COLORS.tu)
      label(ctx, 'T(v)', tvTip, COLORS.tv)
      label(ctx, 'T(u+v)', tsumTip, COLORS.tImage)
    }
    dot(ctx, vTip, COLORS.v) // 近原點時仍可被抓回來
  } else {
    // ---- 輸入側:u 與 k·u(k·u 由 WASM 算)----
    const ku = linalg.scaleVector(u.x, u.y, k)
    const kuTip = S(ku[0], ku[1])
    // u 的方向線(span):k·u 永遠落在這條線上 —— 純量乘不離開 span。
    const uLen = Math.hypot(u.x, u.y)
    if (uLen >= ORIGIN_EPS) {
      const dx = u.x / uLen
      const dy = u.y / uLen
      dashedLine(
        ctx,
        S(-WORLD_HALF * dx, -WORLD_HALF * dy),
        S(WORLD_HALF * dx, WORLD_HALF * dy),
        COLORS.guide,
      )
    }
    drawArrow(ctx, origin, kuTip, COLORS.sum, 2)
    drawArrow(ctx, origin, uTip, COLORS.u, 2.5)
    label(ctx, 'u', uTip, COLORS.u)
    label(ctx, 'k·u', kuTip, COLORS.sum)

    // ---- 影像側:T(k·u)(emerald 箭頭)vs k·T(u)(白色圓環)----
    const tku = T(ku[0], ku[1])
    const ktu = linalg.scaleVector(tu[0], tu[1], k)
    if ([tu, tku, ktu].every(finite)) {
      const tuTip = S(tu[0], tu[1])
      drawArrow(ctx, origin, tuTip, COLORS.tu, 2.5)
      drawArrow(ctx, origin, S(tku[0], tku[1]), COLORS.tImage, 3)
      drawRing(ctx, S(ktu[0], ktu[1]), COLORS.ring)
      label(ctx, 'T(u)', tuTip, COLORS.tu)
      label(ctx, 'T(k·u)', S(tku[0], tku[1]), COLORS.tImage)
    }
  }
  dot(ctx, uTip, COLORS.u) // 近原點時仍可被抓回來
}

export function LinearityCanvas({
  linalg,
  m,
  u,
  v,
  k,
  mode,
  onChangeU,
  onChangeV,
}: LinearityCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef<'u' | 'v' | null>(null)
  const size = useSquareSize(containerRef)

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size <= 0) return
    const ctx = beginFrame(canvas, size)
    if (!ctx) return
    const vp = makeViewport(size)
    drawReferenceGrid(ctx, vp)
    drawLinearity(ctx, vp, linalg, m, u, v, k, mode)
  }, [linalg, m, u, v, k, mode, size])

  // handlers 由 React 每次 render 重建,必看到最新 u/v(無 stale closure)。
  // 純量乘模式只拖 u(v 不在畫面上)。
  const buildHandles = (vp: Viewport): Handle<'u' | 'v'>[] => {
    const [ux, uy] = vp.toScreen(u.x, u.y)
    const handles: Handle<'u' | 'v'>[] = [{ id: 'u', sx: ux, sy: uy, priority: 0 }]
    if (mode === 'add') {
      const [vx, vy] = vp.toScreen(v.x, v.y)
      handles.push({ id: 'v', sx: vx, sy: vy, priority: 1 })
    }
    return handles
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
    if (draggingRef.current === 'u') onChangeU({ x: wx, y: wy })
    else onChangeV({ x: wx, y: wy })
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
