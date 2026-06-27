import { useLayoutEffect, useRef } from 'react'
import type { Linalg } from '../lib/linalg'
import {
  beginFrame,
  dot,
  drawArrow,
  drawReferenceGrid,
  drawRing,
  HIT_PX,
  hitTest,
  label,
  makeViewport,
  strokeLine,
  useSquareSize,
  WORLD_HALF,
  type Handle,
  type Matrix2x2,
  type Pt,
  type Viewport,
} from '../lib/canvas'

// 「λ 多接近特徵值,特徵向量才浮現」的吸附範圍 —— 即 core eigenspace_basis 的 RREF
// 算零門檻。調這個值改吸附手感,計算仍全在 Rust。與 λ 滑桿 step(0.05)搭配:
// 落在特徵值 ±0.1 內即現形(整數特徵值的 preset 很好對到)。route 的資訊面板共用此值。
export const SNAP_EPSILON = 0.1

interface EigenvaluesCanvasProps {
  linalg: Linalg
  /** 運算子 A(row-major a, b, c, d);拖 î' / ĵ'(A 的兩行)即改 A。 */
  a: Matrix2x2
  /** 位移量 λ:畫的是 A − λI 的作用。 */
  lambda: number
  onChangeA: (m: Matrix2x2) => void
}

// eigenvalues 頁專屬色票(結構色在 canvas.ts 的 BASE_COLORS)。
const COLORS = {
  unit: 'rgba(148, 163, 184, 0.35)', // slate-400:原始單位方塊(參考)
  ihat: '#fbbf24', // amber-400:A 的第一行 î'(= A·e₁)
  jhat: '#fb923c', // orange-400:A 的第二行 ĵ'(= A·e₂)
  imagePosFill: 'rgba(52, 211, 153, 0.20)', // emerald-400:det(A−λI) ≥ 0
  imageNegFill: 'rgba(251, 113, 133, 0.20)', // rose-400:det(A−λI) < 0
  imagePos: '#34d399',
  imageNeg: '#fb7185',
  eigen: '#38bdf8', // sky-400:特徵向量(A 的不變方向)
  eigenImage: '#f8fafc', // slate-50:A·v 落點(= λv,證明只被伸縮、留在自己的線上)
} as const

type HandleId = 'ihat' | 'jhat'

/** 原始單位方塊外框(0,0)-(1,0)-(1,1)-(0,1),當作「塌縮多少」的尺規。 */
function drawUnitSquare(ctx: CanvasRenderingContext2D, vp: Viewport) {
  const S = vp.toScreen
  ctx.save()
  ctx.strokeStyle = COLORS.unit
  ctx.lineWidth = 1
  ctx.setLineDash([4, 4])
  ctx.beginPath()
  ctx.moveTo(...S(0, 0))
  ctx.lineTo(...S(1, 0))
  ctx.lineTo(...S(1, 1))
  ctx.lineTo(...S(0, 1))
  ctx.closePath()
  ctx.stroke()
  ctx.restore()
}

/** 填滿四邊形(平行四邊形 = M 把單位方塊送到的像),依 det 正負上色。 */
function fillParallelogram(ctx: CanvasRenderingContext2D, pts: Pt[], fill: string, stroke: string) {
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(pts[0][0], pts[0][1])
  for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1])
  ctx.closePath()
  ctx.fillStyle = fill
  ctx.fill()
  ctx.strokeStyle = stroke
  ctx.lineWidth = 2
  ctx.stroke()
  ctx.restore()
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  a: Matrix2x2,
  lambda: number,
) {
  const S = vp.toScreen
  const origin = S(0, 0)
  const aFlat = [a.a, a.b, a.c, a.d]

  drawUnitSquare(ctx, vp)

  // M = A − λI 把單位方塊送到的平行四邊形。端點一律由 core 的 transformPoint 算
  // (JS 不做向量乘法);M 本身也是 core 的 characteristic_matrix 算的。
  const m = linalg.characteristicMatrix2d(aFlat, lambda)
  if (m.length === 4) {
    const img = (x: number, y: number): Pt => {
      const p = linalg.transformPoint(m[0], m[1], m[2], m[3], x, y)
      return S(p[0], p[1])
    }
    const detM = linalg.determinant([m[0], m[1], m[2], m[3]], 2)
    const fill = detM >= 0 ? COLORS.imagePosFill : COLORS.imageNegFill
    const stroke = detM >= 0 ? COLORS.imagePos : COLORS.imageNeg
    // 像的四角:M·(0,0)=原點(線性)、M·(1,0)、M·(1,1)、M·(0,1)。
    fillParallelogram(ctx, [origin, img(1, 0), img(1, 1), img(0, 1)], fill, stroke)
  }

  // 特徵向量:λ 夠接近特徵值時 core 回非空(Eλ = Null(A−λI))。畫不變方向(過原點長線),
  // 再用 A·v 落點驗證 A 只把 v 伸縮 λ 倍、留在同一條線上(A·v = λv,兩條 core 路會合)。
  const eig = linalg.eigenspaceBasis2d(aFlat, lambda, SNAP_EPSILON)
  for (let k = 0; k + 1 < eig.length; k += 2) {
    const len = Math.hypot(eig[k], eig[k + 1]) || 1
    const ux = eig[k] / len
    const uy = eig[k + 1] / len
    ctx.save()
    ctx.strokeStyle = COLORS.eigen
    ctx.lineWidth = 1.5
    ctx.setLineDash([6, 4])
    const T = WORLD_HALF * 1.5
    strokeLine(ctx, S(-T * ux, -T * uy), S(T * ux, T * uy))
    ctx.restore()
    // v 取固定顯示長度 2.5;A·v = transformPoint(A, v) = λ·v,落在同一條線上。
    const vx = ux * 2.5
    const vy = uy * 2.5
    const av = linalg.transformPoint(a.a, a.b, a.c, a.d, vx, vy)
    drawArrow(ctx, origin, S(vx, vy), COLORS.eigen, 3)
    drawArrow(ctx, origin, S(av[0], av[1]), COLORS.eigenImage, 1.5)
    drawRing(ctx, S(av[0], av[1]), COLORS.eigenImage)
    label(ctx, 'v', S(vx, vy), COLORS.eigen)
  }

  // A 的兩行 î'、ĵ'(可拖 = 改 A)。λ=0 時平行四邊形的兩邊正好落在這兩個箭頭上;
  // λ 一動,平行四邊形(A−λI)就從這裡往內縮,縮到塌 = λ 命中特徵值。
  const ihat = S(a.a, a.c)
  const jhat = S(a.b, a.d)
  drawArrow(ctx, origin, ihat, COLORS.ihat, 2)
  drawArrow(ctx, origin, jhat, COLORS.jhat, 2)
  dot(ctx, ihat, COLORS.ihat)
  dot(ctx, jhat, COLORS.jhat)
  label(ctx, "î'", ihat, COLORS.ihat)
  label(ctx, "ĵ'", jhat, COLORS.jhat)
}

export function EigenvaluesCanvas({ linalg, a, lambda, onChangeA }: EigenvaluesCanvasProps) {
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
    drawScene(ctx, vp, linalg, a, lambda)
  }, [linalg, a, lambda, size])

  const buildHandles = (vp: Viewport): Handle<HandleId>[] => {
    const [ix, iy] = vp.toScreen(a.a, a.c)
    const [jx, jy] = vp.toScreen(a.b, a.d)
    return [
      { id: 'ihat', sx: ix, sy: iy, priority: 0 },
      { id: 'jhat', sx: jx, sy: jy, priority: 1 },
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
      onChangeA({ ...a, a: wx, c: wy }) // î' = A 的第一行 = (a, c)
    } else {
      onChangeA({ ...a, b: wx, d: wy }) // ĵ' = A 的第二行 = (b, d)
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
