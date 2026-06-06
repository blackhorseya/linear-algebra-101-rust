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
  useSquareSize,
  type Handle,
  type Matrix2x2,
  type Vec2,
  type Viewport,
} from '../lib/canvas'

interface RangeCanvasProps {
  linalg: Linalg
  /** 2×2 標準矩陣 —— 行向量 a₁ = (a, c)、a₂ = (b, d) 可在圖上直接拖。 */
  m: Matrix2x2
  /** codomain 的測試向量 w:拖著問「到得了嗎?」(可達性由 core 即時判定)。 */
  w: Vec2
  onChangeM: (m: Matrix2x2) => void
  onChangeW: (w: Vec2) => void
}

// range 頁專屬色票(結構色在 canvas.ts 的 BASE_COLORS)。
// a₁ / a₂ 沿 standard-matrix 的「行色」(amber / rose)—— 行向量 = 值域的生成元素;
// w 的綠 / 紅不是寫死的:每幀由 core 的 range_contains 判定後選色。
const COLORS = {
  tGrid: '#3b3554', // 變換後網格(整個平面的像 —— Range 本人)
  tGridAxis: '#5b5286', // 變換後座標軸的像
  rangeLine: 'rgba(167, 139, 250, 0.45)', // violet-400 @ 45%:塌縮後的 Range 直線
  a1: '#fbbf24', // amber-400:行 1
  a2: '#fb7185', // rose-400:行 2
  reachable: '#34d399', // emerald-400:w ∈ Range
  unreachable: '#f87171', // red-400:w ∉ Range(與不可達見證同色,語意一致)
  x: '#94a3b8', // slate-400:唯一輸入 x(虛線 —— 它住在 domain,不是像)
  ring: '#f8fafc', // slate-50:A·x 的落點(矩陣路徑)—— 圓環套住 w = 兩路會合
} as const

type HandleId = 'a1' | 'a2' | 'w'

function drawRangeScene(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  m: Matrix2x2,
  w: Vec2,
) {
  const S = vp.toScreen
  const origin = S(0, 0)

  // 變換後網格(淡):整個平面的像。rank 2 蓋滿、rank 1 塌成一條線、
  // rank 0 縮成原點 —— 「Range 蓋住多少」不用另外畫,網格的像就是答案。
  ctx.save()
  for (let g = -GRID_N; g <= GRID_N; g++) {
    ctx.strokeStyle = g === 0 ? COLORS.tGridAxis : COLORS.tGrid
    ctx.lineWidth = g === 0 ? 1.6 : 1
    drawImageSegment(ctx, linalg.transformPoint, m, vp, g, -GRID_N, g, GRID_N)
    drawImageSegment(ctx, linalg.transformPoint, m, vp, -GRID_N, g, GRID_N, g)
  }
  ctx.restore()

  // Range 的形狀由 core 的 range_basis 決定(攤平長度 = 2·rank):
  // 2 → 直線(沿基底向量無限延伸);0 → 原點;4 → ℝ²(網格已蓋滿,不必再畫)。
  const basis = linalg.rangeBasis(m.a, m.b, m.c, m.d)
  if (basis.length === 2) {
    const [bx, by] = basis
    const t = 100 / Math.hypot(bx, by) // 基底非零(零行成不了 pivot);兩端伸出視野
    ctx.save()
    ctx.strokeStyle = COLORS.rangeLine
    ctx.lineWidth = 7
    ctx.lineCap = 'round'
    const p0 = S(-t * bx, -t * by)
    const p1 = S(t * bx, t * by)
    ctx.beginPath()
    ctx.moveTo(p0[0], p0[1])
    ctx.lineTo(p1[0], p1[1])
    ctx.stroke()
    ctx.restore()
  } else if (basis.length === 0) {
    dot(ctx, origin, COLORS.rangeLine) // 零轉換:值域只剩原點
  }

  // 不可達見證(core 的 unreachable_vector,標準基底掃描):
  // 不映成時,被值域漏掉的那支 eᵢ —— 紅圈標出「這裡到不了」。
  const witness = linalg.unreachableVector(m.a, m.b, m.c, m.d)
  if (witness.length === 2) {
    const p = S(witness[0], witness[1])
    drawRing(ctx, p, COLORS.unreachable)
    label(ctx, `${witness[0] === 1 ? 'e₁' : 'e₂'} ∉ Range`, p, COLORS.unreachable)
  }

  // 行向量 a₁、a₂(可拖,= 值域的生成集合):dot 保證塌縮 / 零矩陣時仍抓得到。
  const a1Tip = S(m.a, m.c)
  const a2Tip = S(m.b, m.d)
  drawArrow(ctx, origin, a1Tip, COLORS.a1, 3)
  drawArrow(ctx, origin, a2Tip, COLORS.a2, 3)
  dot(ctx, a1Tip, COLORS.a1)
  dot(ctx, a2Tip, COLORS.a2)
  label(ctx, 'a₁', a1Tip, COLORS.a1)
  label(ctx, 'a₂', a2Tip, COLORS.a2)

  // 測試向量 w:綠 / 紅由 core 的 range_contains 當場判,不是 JS 的條件式著色。
  const reachable = linalg.rangeContains(m.a, m.b, m.c, m.d, w.x, w.y)
  const wColor = reachable ? COLORS.reachable : COLORS.unreachable
  const wTip = S(w.x, w.y)
  drawArrow(ctx, origin, wTip, wColor, 2.5)
  dot(ctx, wTip, wColor)
  label(ctx, reachable ? 'w ✓' : 'w ✗', wTip, wColor)

  // 唯一解時把「到得了」的見證 x 畫出來(虛線:它住在 domain、是 w 的前身),
  // 再左乘 A 走矩陣路徑 —— 白圓環必套住 w 的箭頭尖端(A·x = w,兩路會合)。
  const solved = linalg.solveForInput(m.a, m.b, m.c, m.d, w.x, w.y)
  if (solved.x) {
    const [xx, xy] = solved.x
    const xTip = S(xx, xy)
    ctx.save()
    ctx.setLineDash([6, 4])
    drawArrow(ctx, origin, xTip, COLORS.x, 2)
    ctx.restore()
    label(ctx, 'x', xTip, COLORS.x)
    const ax = linalg.transformPoint(m.a, m.b, m.c, m.d, xx, xy)
    drawRing(ctx, S(ax[0], ax[1]), COLORS.ring)
  }
}

export function RangeCanvas({
  linalg,
  m,
  w,
  onChangeM,
  onChangeW,
}: RangeCanvasProps) {
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
    drawRangeScene(ctx, vp, linalg, m, w)
  }, [linalg, m, w, size])

  // 三個 handle 都可拖:a₁、a₂ 直接改矩陣(行向量「就是」矩陣 —— 拖行向量
  // = 改 A = 改值域),w 是 codomain 的測試點。w 優先:它常與行向量重疊。
  const buildHandles = (vp: Viewport): Handle<HandleId>[] => {
    const [a1x, a1y] = vp.toScreen(m.a, m.c)
    const [a2x, a2y] = vp.toScreen(m.b, m.d)
    const [wx, wy] = vp.toScreen(w.x, w.y)
    return [
      { id: 'w', sx: wx, sy: wy, priority: 0 },
      { id: 'a1', sx: a1x, sy: a1y, priority: 1 },
      { id: 'a2', sx: a2x, sy: a2y, priority: 2 },
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
    if (dragging === 'w') {
      onChangeW({ x: wx, y: wy })
    } else if (dragging === 'a1') {
      onChangeM({ ...m, a: wx, c: wy }) // a₁ = (a, c):第 1 行
    } else {
      onChangeM({ ...m, b: wx, d: wy }) // a₂ = (b, d):第 2 行
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
