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

interface CompositionCanvasProps {
  linalg: Linalg
  /** 內層轉換 T(先施)的標準矩陣 A。 */
  t: Matrix2x2
  /**
   * 外層轉換 U(後施)的標準矩陣 B;`null` = 沒有第二步
   * (逆轉換模式下 T 不可逆 → T⁻¹ 不存在,只畫到 T(x) 為止)。
   */
  u: Matrix2x2 | null
  /** 外層轉換在圖上的名字(合成模式 'U',逆轉換模式 'T⁻¹')。 */
  outerLabel: string
  /** domain 的測試向量 x:拖著看兩條路徑跟著走。 */
  x: Vec2
  onChangeX: (x: Vec2) => void
}

// composition 頁專屬色票(結構色在 canvas.ts 的 BASE_COLORS)。
// 路徑三站沿「出發 → 中繼 → 終點」漸進:slate → amber → rose;
// 一步直達的白圓環沿全站慣例(「另一條計算路徑」的落點)。
const COLORS = {
  tGrid: '#3b3554', // 合成後網格(BA 一步走完的世界)
  tGridAxis: '#5b5286', // 合成後座標軸的像
  x: '#94a3b8', // slate-400:出發點 x(domain,可拖)
  mid: '#fbbf24', // amber-400:中繼站 T(x)
  end: '#fb7185', // rose-400:終點 U(T(x))
  ring: '#f8fafc', // slate-50:(BA)x 的落點 —— 圓環套住終點 = 兩路會合
} as const

type HandleId = 'x'

function drawCompositionScene(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  t: Matrix2x2,
  u: Matrix2x2 | null,
  outerLabel: string,
  x: Vec2,
) {
  const S = vp.toScreen
  const origin = S(0, 0)

  // 合成矩陣 BA 由 core 的 compose 算(u 缺席時退回只看 T)——
  // 變換後網格畫「一步走完的世界」:逆轉換模式 BA = T⁻¹·A = I,
  // 網格像與參考網格重合,「復原」直接看得見。
  const composed: Matrix2x2 | null = u
    ? (() => {
        const ba = linalg.composeMatrix(u.a, u.b, u.c, u.d, t.a, t.b, t.c, t.d)
        return { a: ba[0], b: ba[1], c: ba[2], d: ba[3] }
      })()
    : null
  const gridMatrix = composed ?? t
  ctx.save()
  for (let g = -GRID_N; g <= GRID_N; g++) {
    ctx.strokeStyle = g === 0 ? COLORS.tGridAxis : COLORS.tGrid
    ctx.lineWidth = g === 0 ? 1.6 : 1
    drawImageSegment(ctx, linalg.transformPoint, gridMatrix, vp, g, -GRID_N, g, GRID_N)
    drawImageSegment(ctx, linalg.transformPoint, gridMatrix, vp, -GRID_N, g, GRID_N, g)
  }
  ctx.restore()

  // 路徑三站:x → T(x) → U(T(x)),每一站都是 core 的 transformPoint 算的。
  const mid = linalg.transformPoint(t.a, t.b, t.c, t.d, x.x, x.y)
  const xTip = S(x.x, x.y)
  const midTip = S(mid[0], mid[1])

  // 步驟虛線(路徑觀):x 尖端 → T(x) 尖端,標「① T」;有第二步再接「② U」。
  const drawStep = (from: [number, number], to: [number, number], color: string, name: string) => {
    ctx.save()
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.setLineDash([5, 4])
    ctx.beginPath()
    ctx.moveTo(from[0], from[1])
    ctx.lineTo(to[0], to[1])
    ctx.stroke()
    ctx.restore()
    label(ctx, name, [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2], color)
  }
  drawStep(xTip, midTip, COLORS.mid, '① T')

  // 向量箭頭(向量觀):三站各一支,從原點出發。
  drawArrow(ctx, origin, xTip, COLORS.x, 2.5)
  drawArrow(ctx, origin, midTip, COLORS.mid, 2.5)
  dot(ctx, midTip, COLORS.mid)
  label(ctx, 'T(x)', midTip, COLORS.mid)

  if (u && composed) {
    // 第二步:T(x) 再被 U 送走 —— 終點 U(T(x))。
    const end = linalg.transformPoint(u.a, u.b, u.c, u.d, mid[0], mid[1])
    const endTip = S(end[0], end[1])
    drawStep(midTip, endTip, COLORS.end, `② ${outerLabel}`)
    drawArrow(ctx, origin, endTip, COLORS.end, 2.5)
    dot(ctx, endTip, COLORS.end)
    label(ctx, `${outerLabel}(T(x))`, endTip, COLORS.end)

    // 一步直達:x 左乘合成矩陣 BA(core 的 compose 已算好)——
    // 白圓環必套住兩步走出來的終點:T_B ∘ T_A = T_BA 每幀上演。
    const oneStep = linalg.transformPoint(
      composed.a,
      composed.b,
      composed.c,
      composed.d,
      x.x,
      x.y,
    )
    drawRing(ctx, S(oneStep[0], oneStep[1]), COLORS.ring)
  }

  // x 畫最後(handle 永遠在最上層,塌縮時也抓得到)。
  dot(ctx, xTip, COLORS.x)
  label(ctx, 'x', xTip, COLORS.x)
}

export function CompositionCanvas({
  linalg,
  t,
  u,
  outerLabel,
  x,
  onChangeX,
}: CompositionCanvasProps) {
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
    drawCompositionScene(ctx, vp, linalg, t, u, outerLabel, x)
  }, [linalg, t, u, outerLabel, x, size])

  // 唯一的 handle:出發點 x(矩陣用輸入框 / preset 調 —— 這頁的主角是路徑)。
  const buildHandles = (vp: Viewport): Handle<HandleId>[] => {
    const [sx, sy] = vp.toScreen(x.x, x.y)
    return [{ id: 'x', sx, sy, priority: 0 }]
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
    const [wx, wy] = vp.toWorld(px, py)
    onChangeX({ x: wx, y: wy })
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
