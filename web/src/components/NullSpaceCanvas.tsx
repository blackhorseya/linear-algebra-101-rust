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
  type Vec2,
  type Viewport,
} from '../lib/canvas'

interface NullSpaceCanvasProps {
  linalg: Linalg
  /** 2×2 矩陣(用 preset / 數字框改;這是 domain 視圖,行向量住在 codomain 不在此拖)。 */
  m: Matrix2x2
  /** domain 的輸入向量 v:拖著問「會不會被壓到原點?」(成員資格由 core 判定)。 */
  v: Vec2
  onChangeV: (v: Vec2) => void
}

// nullspace 頁專屬色票(結構色在 canvas.ts 的 BASE_COLORS)。與 /range 對偶:
// 那裡畫輸出端的值域覆蓋,這裡畫輸入端被壓扁的核。v 的藍 / 綠不是寫死的:
// 每幀由 core 的 null_space_contains 判定後選色(在核裡 = 綠)。
const COLORS = {
  kernelLine: 'rgba(167, 139, 250, 0.5)', // violet-400 @ 50%:核 Null A(一條過原點的線)
  kernelFill: 'rgba(167, 139, 250, 0.12)', // 整個平面都是核(零矩陣)時的淡紫覆蓋
  v: '#38bdf8', // sky-400:輸入向量 v(不在核裡)
  vKernel: '#34d399', // emerald-400:v ∈ Null A(被壓到原點)
  image: '#fbbf24', // amber-400:像 Av(虛線 —— 它住在 codomain)
} as const

type HandleId = 'v'

/** 掃描單位半圓找「被 A 壓得最扁」的方向 —— nullity = 1 時的核線方向。
 *  純繪圖輔助:每個方向的像 Av 由 core 的 transformPoint 算,JS 只挑 |Av| 最小者
 *  來畫線;「某向量到底在不在核裡」的權威判定一律走 null_space_contains。 */
function findKernelDirection(linalg: Linalg, m: Matrix2x2): Pt {
  const STEPS = 720 // 0.25° 解析度
  let best: Pt = [1, 0]
  let bestNorm = Infinity
  for (let i = 0; i < STEPS; i++) {
    const t = (Math.PI * i) / STEPS
    const dx = Math.cos(t)
    const dy = Math.sin(t)
    const av = linalg.transformPoint(m.a, m.b, m.c, m.d, dx, dy)
    const norm = Math.hypot(av[0], av[1])
    if (norm < bestNorm) {
      bestNorm = norm
      best = [dx, dy]
    }
  }
  return best
}

function drawNullScene(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  m: Matrix2x2,
  v: Vec2,
) {
  const S = vp.toScreen
  const origin = S(0, 0)
  const { a, b, c, d } = m

  // 核 Null A 的形狀由 core 的 nullity 決定:
  // 2 → 整個 domain 都被壓扁(淡紫鋪滿);1 → 一條過原點的核線;0 → 只剩原點。
  const nul = linalg.nullity(a, b, c, d)
  if (nul === 2) {
    ctx.save()
    ctx.fillStyle = COLORS.kernelFill
    const tl = S(-WORLD_HALF, WORLD_HALF)
    const br = S(WORLD_HALF, -WORLD_HALF)
    ctx.fillRect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
    ctx.restore()
  } else if (nul === 1) {
    const [kx, ky] = findKernelDirection(linalg, m)
    ctx.save()
    ctx.strokeStyle = COLORS.kernelLine
    ctx.lineWidth = 7
    ctx.lineCap = 'round'
    strokeLine(ctx, S(-100 * kx, -100 * ky), S(100 * kx, 100 * ky))
    ctx.restore()
  }

  // 像 Av(橙虛線,先畫在底層):由 core 的 transformPoint 算,JS 不做乘法。
  // v 落在核裡時 Av ≈ 0 → 箭頭塌進原點,改用圓環圈住原點點出「壓到這裡」。
  const av = linalg.transformPoint(a, b, c, d, v.x, v.y)
  const inKernel = linalg.nullSpaceContains(a, b, c, d, v.x, v.y)
  if (!inKernel) {
    const avTip = S(av[0], av[1])
    ctx.save()
    ctx.setLineDash([5, 4])
    drawArrow(ctx, origin, avTip, COLORS.image, 2)
    ctx.restore()
    label(ctx, 'Av', avTip, COLORS.image)
  }

  // 輸入向量 v(可拖):綠 / 藍由 core 的 null_space_contains 當場判,非 JS 條件著色。
  const vColor = inKernel ? COLORS.vKernel : COLORS.v
  const vTip = S(v.x, v.y)
  drawArrow(ctx, origin, vTip, vColor, 3)
  dot(ctx, vTip, vColor)
  label(ctx, inKernel ? 'v ∈ Null A' : 'v', vTip, vColor)

  // v 在核裡 → Av 塌到原點:圓環圈住原點,與 v 同綠,點出「這支被壓沒了」。
  if (inKernel) drawRing(ctx, origin, COLORS.vKernel)
}

export function NullSpaceCanvas({
  linalg,
  m,
  v,
  onChangeV,
}: NullSpaceCanvasProps) {
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
    drawNullScene(ctx, vp, linalg, m, v)
  }, [linalg, m, v, size])

  // 只有 v 可拖:它住在 domain(這張圖的主角)。矩陣由 preset / 數字框改 ——
  // 行向量是 codomain 的東西,擺進 domain 視圖拖會混淆兩個空間。
  const buildHandles = (vp: Viewport): Handle<HandleId>[] => {
    const [vx, vy] = vp.toScreen(v.x, v.y)
    return [{ id: 'v', sx: vx, sy: vy, priority: 0 }]
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
    onChangeV({ x: wx, y: wy })
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
