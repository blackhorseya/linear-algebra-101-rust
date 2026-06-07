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
  useSquareSize,
  type Handle,
  type Matrix2x2,
  type Viewport,
} from '../lib/canvas'
import { fmt } from '../lib/format'

interface DeterminantCanvasProps {
  linalg: Linalg
  m: Matrix2x2
  onChangeMatrix: (m: Matrix2x2) => void
}

// determinant 專屬色票(結構色在 canvas.ts 的 BASE_COLORS;
// î/ĵ 與正負定向的配色沿 TransformCanvas,跨頁一致)
const COLORS = {
  tGrid: '#4c4368', // 變換後網格(violet 偏暗)
  tGridAxis: '#7c6db0', // 變換後座標軸的像(亮一點)
  iHat: '#34d399', // emerald-400
  jHat: '#f87171', // red-400
  unitSquare: '#94a3b8', // slate-400:變換前的單位正方形(虛線輪廓)
  squarePos: 'rgba(167,139,250,0.22)', // det>0:violet(定向不變)
  squareNeg: 'rgba(251,191,36,0.22)', // det<0:amber(翻面)
  outlinePos: '#a78bfa', // violet-400
  outlineNeg: '#fbbf24', // amber-400
  collapsed: '#fb7185', // rose-400:塌縮成線(det = 0)
} as const

function drawDeterminant(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  m: Matrix2x2,
) {
  const S = vp.toScreen
  const origin = S(0, 0)
  const flat = [m.a, m.b, m.c, m.d]

  // 變換後網格(每條線的像都靠 WASM transformPoint 算端點)
  ctx.save()
  for (let k = -GRID_N; k <= GRID_N; k++) {
    ctx.strokeStyle = k === 0 ? COLORS.tGridAxis : COLORS.tGrid
    ctx.lineWidth = k === 0 ? 1.8 : 1
    drawImageSegment(ctx, linalg.transformPoint, m, vp, k, -GRID_N, k, GRID_N) // x=k 的像
    drawImageSegment(ctx, linalg.transformPoint, m, vp, -GRID_N, k, GRID_N, k) // y=k 的像
  }
  ctx.restore()

  // 「變換前」:單位正方形(虛線輪廓,面積恆為 1 的基準)
  ctx.save()
  ctx.strokeStyle = COLORS.unitSquare
  ctx.lineWidth = 1.5
  ctx.setLineDash([5, 4])
  ctx.beginPath()
  ctx.moveTo(origin[0], origin[1])
  const sq10 = S(1, 0)
  const sq11 = S(1, 1)
  const sq01 = S(0, 1)
  ctx.lineTo(sq10[0], sq10[1])
  ctx.lineTo(sq11[0], sq11[1])
  ctx.lineTo(sq01[0], sq01[1])
  ctx.closePath()
  ctx.stroke()
  ctx.restore()

  // 「變換後」:單位正方形的像(平行四邊形)。遠角 î'+ĵ' 由 WASM 取得,
  // JS 不做向量加法;det(面積與定向)與塌縮判定也都由 WASM 算 ——
  // 顏色走 det 路(正負定向),塌縮樣式走 is_invertible 的 rank 路。
  const iTip = S(m.a, m.c)
  const jTip = S(m.b, m.d)
  const farW = linalg.transformPoint(m.a, m.b, m.c, m.d, 1, 1)
  if ([m.a, m.c, m.b, m.d, farW[0], farW[1]].every(Number.isFinite)) {
    const far = S(farW[0], farW[1])
    const det = linalg.determinant(flat, 2)
    const collapsed = !linalg.isInvertible(flat, 2)
    ctx.save()
    ctx.beginPath()
    ctx.moveTo(origin[0], origin[1])
    ctx.lineTo(iTip[0], iTip[1])
    ctx.lineTo(far[0], far[1])
    ctx.lineTo(jTip[0], jTip[1])
    ctx.closePath()
    if (collapsed) {
      // 塌縮:平行四邊形退化成線段,只描邊(rose)讓「面積 = 0」看得見
      ctx.strokeStyle = COLORS.collapsed
      ctx.lineWidth = 2.5
      ctx.stroke()
    } else {
      ctx.fillStyle = det < 0 ? COLORS.squareNeg : COLORS.squarePos
      ctx.fill()
      ctx.strokeStyle = det < 0 ? COLORS.outlineNeg : COLORS.outlinePos
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
    ctx.restore()

    // 面積標籤放在原點與遠角的螢幕中點(純排版定位,非線代計算)
    const mid: [number, number] = [
      (origin[0] + far[0]) / 2,
      (origin[1] + far[1]) / 2,
    ]
    const tag = collapsed ? '面積 = 0' : `|det| = ${fmt(Math.abs(det))}`
    label(ctx, tag, mid, collapsed ? COLORS.collapsed : COLORS.unitSquare)
  }

  // 基底箭頭 î'、ĵ'(端點即拖曳 handle)
  drawArrow(ctx, origin, iTip, COLORS.iHat, 2.5)
  drawArrow(ctx, origin, jTip, COLORS.jHat, 2.5)
  dot(ctx, iTip, COLORS.iHat)
  dot(ctx, jTip, COLORS.jHat)
  label(ctx, "î'", iTip, COLORS.iHat)
  label(ctx, "ĵ'", jTip, COLORS.jHat)

  // 單位正方形的「1」基準標籤(虛線框中心;調暗避免搶走像的視覺焦點)
  const unitMid = S(0.5, 0.5)
  ctx.save()
  ctx.globalAlpha = 0.7
  label(ctx, '面積 1', unitMid, COLORS.unitSquare)
  ctx.restore()
}

export function DeterminantCanvas({
  linalg,
  m,
  onChangeMatrix,
}: DeterminantCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef<'iHat' | 'jHat' | null>(null)
  const size = useSquareSize(containerRef)

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size <= 0) return
    const ctx = beginFrame(canvas, size)
    if (!ctx) return
    const vp = makeViewport(size)
    drawReferenceGrid(ctx, vp)
    drawDeterminant(ctx, vp, linalg, m)
  }, [linalg, m, size])

  // handlers 由 React 每次 render 重建,必看到最新 m(無 stale closure)
  const buildHandles = (vp: Viewport): Handle<'iHat' | 'jHat'>[] => {
    const [ix, iy] = vp.toScreen(m.a, m.c)
    const [jx, jy] = vp.toScreen(m.b, m.d)
    return [
      { id: 'iHat', sx: ix, sy: iy, priority: 0 },
      { id: 'jHat', sx: jx, sy: jy, priority: 1 },
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
    if (!draggingRef.current) {
      const hit = hitTest(px, py, buildHandles(vp), HIT_PX)
      e.currentTarget.style.cursor = hit ? 'grab' : 'crosshair'
      return
    }
    const [x, y] = vp.toWorld(px, py)
    if (draggingRef.current === 'iHat') {
      onChangeMatrix({ ...m, a: x, c: y })
    } else {
      onChangeMatrix({ ...m, b: x, d: y })
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
