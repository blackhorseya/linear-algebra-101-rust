import { useLayoutEffect, useRef } from 'react'
import type { Linalg, RuleKind } from '../lib/linalg'
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
  useSquareSize,
  type Handle,
  type Matrix2x2,
  type Vec2,
  type Viewport,
} from '../lib/canvas'

interface SamplerCanvasProps {
  linalg: Linalg
  /** 幾何規則與其參數(rotate 收弧度)—— T 的「規則側」。 */
  rule: RuleKind
  param: number
  /** 取樣出的標準矩陣(頁面經 sampleStandardMatrix 算好傳入,單一真相共用)。 */
  m: Matrix2x2
  /** 可拖曳的測試向量 v:看兩條路徑(規則 vs 矩陣)在 T(v) 會合。 */
  v: Vec2
  onChangeV: (v: Vec2) => void
}

// standard-matrix 專屬色票(結構色在 canvas.ts 的 BASE_COLORS)。
// T(e₁) / T(e₂) 的顏色刻意對齊矩陣面板的行色:箭頭 = 矩陣的行,看一眼就接上。
const COLORS = {
  tGrid: '#3b3554', // 變換後網格(主角是取樣現場,網格退為背景)
  tGridAxis: '#5b5286', // 變換後座標軸的像
  e1: '#a78bfa', // violet-400:e₁
  e2: '#38bdf8', // sky-400:e₂
  te1: '#fbbf24', // amber-400:T(e₁) = A 的第 1 行
  te2: '#fb7185', // rose-400:T(e₂) = A 的第 2 行
  v: '#e2e8f0', // slate-200:測試向量 v
  tv: '#34d399', // emerald-400:T(v),規則直接算(規則路徑)
  ring: '#f8fafc', // slate-50:A·v,左乘取樣矩陣(矩陣路徑)—— 圓環套住箭頭 = Theorem 2.9
} as const

function drawSampler(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  rule: RuleKind,
  param: number,
  m: Matrix2x2,
  v: Vec2,
) {
  const S = vp.toScreen
  const origin = S(0, 0)
  const finite = (p: Float64Array) => [p[0], p[1]].every(Number.isFinite)

  // 變換後網格(淡):整個平面的像,由「矩陣路徑」(取樣矩陣 + core 的
  // multiply_vector)畫 —— 與規則路徑一致正是 Theorem 2.9 本身。
  ctx.save()
  for (let g = -GRID_N; g <= GRID_N; g++) {
    ctx.strokeStyle = g === 0 ? COLORS.tGridAxis : COLORS.tGrid
    ctx.lineWidth = g === 0 ? 1.6 : 1
    drawImageSegment(ctx, linalg.transformPoint, m, vp, g, -GRID_N, g, GRID_N)
    drawImageSegment(ctx, linalg.transformPoint, m, vp, -GRID_N, g, GRID_N, g)
  }
  ctx.restore()

  // ---- 取樣現場:e₁、e₂(細箭頭)與它們的影像(粗箭頭,規則直接算)----
  // T(eⱼ) 的座標就是矩陣第 j 行 —— 顏色與矩陣面板的行色一一對應。
  const te1 = linalg.applyRule(rule, param, 1, 0)
  const te2 = linalg.applyRule(rule, param, 0, 1)
  const e1Tip = S(1, 0)
  const e2Tip = S(0, 1)
  drawArrow(ctx, origin, e1Tip, COLORS.e1, 2)
  drawArrow(ctx, origin, e2Tip, COLORS.e2, 2)
  label(ctx, 'e₁', e1Tip, COLORS.e1)
  label(ctx, 'e₂', e2Tip, COLORS.e2)
  if (finite(te1)) {
    const tip = S(te1[0], te1[1])
    drawArrow(ctx, origin, tip, COLORS.te1, 3)
    label(ctx, 'T(e₁)', tip, COLORS.te1)
  }
  if (finite(te2)) {
    const tip = S(te2[0], te2[1])
    drawArrow(ctx, origin, tip, COLORS.te2, 3)
    label(ctx, 'T(e₂)', tip, COLORS.te2)
  }

  // ---- 測試向量 v:規則路徑 T(v)(綠箭頭)vs 矩陣路徑 A·v(白圓環)----
  // 兩者都在 Rust 算;重合 = 取樣出的 A 真的代表了規則(Theorem 2.9)。
  const vTip = S(v.x, v.y)
  const tv = linalg.applyRule(rule, param, v.x, v.y)
  const av = linalg.transformPoint(m.a, m.b, m.c, m.d, v.x, v.y)
  drawArrow(ctx, origin, vTip, COLORS.v, 2)
  label(ctx, 'v', vTip, COLORS.v)
  if (finite(tv) && finite(av)) {
    const tvTip = S(tv[0], tv[1])
    drawArrow(ctx, origin, tvTip, COLORS.tv, 2.5)
    drawRing(ctx, S(av[0], av[1]), COLORS.ring)
    label(ctx, 'T(v)', tvTip, COLORS.tv)
  }
  dot(ctx, vTip, COLORS.v) // 近原點時仍可被抓回來
}

export function SamplerCanvas({
  linalg,
  rule,
  param,
  m,
  v,
  onChangeV,
}: SamplerCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)
  const size = useSquareSize(containerRef)

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size <= 0) return
    const ctx = beginFrame(canvas, size)
    if (!ctx) return
    const vp = makeViewport(size)
    drawReferenceGrid(ctx, vp)
    drawSampler(ctx, vp, linalg, rule, param, m, v)
  }, [linalg, rule, param, m, v, size])

  // 只有測試向量 v 可拖 —— e₁、e₂ 是標準基底,被釘死正是「standard」的意思。
  const buildHandles = (vp: Viewport): Handle<'v'>[] => {
    const [sx, sy] = vp.toScreen(v.x, v.y)
    return [{ id: 'v', sx, sy, priority: 0 }]
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
    draggingRef.current = true
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
    onChangeV({ x: wx, y: wy })
  }

  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current) return
    e.currentTarget.releasePointerCapture(e.pointerId)
    draggingRef.current = false
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
          draggingRef.current = false
        }}
      />
    </div>
  )
}
