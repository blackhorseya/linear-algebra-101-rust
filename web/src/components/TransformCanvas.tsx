import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Linalg } from '../lib/linalg'

export interface Matrix2x2 {
  a: number
  b: number
  c: number
  d: number
}
export interface Vec2 {
  x: number
  y: number
}

interface TransformCanvasProps {
  linalg: Linalg
  m: Matrix2x2
  v: Vec2
  onChangeMatrix: (m: Matrix2x2) => void
  onChangeV: (v: Vec2) => void
}

// ── 視口 / 常數 ──────────────────────────────────────────────
const WORLD_HALF = 8 // 視口世界半徑(±8 單位都看得到)
const GRID_N = 6 // 變換後網格與線段的範圍(保持畫面可讀)
const HIT_PX = 12 // 拖曳 handle 的命中半徑(CSS px)
const V_EPS = 0.04 // v 視為「在原點」的門檻(世界單位)

// canvas 不能吃 Tailwind class,顏色集中對齊主題色票
const COLORS = {
  bg: '#020617', // slate-950
  refGrid: '#1e293b', // slate-800
  axis: '#475569', // slate-600
  tGrid: '#4c4368', // 變換後網格(violet 偏暗)
  tGridAxis: '#7c6db0', // 變換後座標軸的像(亮一點)
  iHat: '#34d399', // emerald-400
  jHat: '#f87171', // red-400
  v: '#a78bfa', // violet-400
  av: '#fbbf24', // amber-400
  eigen: '#34d399', // emerald 虛線(特徵方向)
  label: '#cbd5e1', // slate-300
  square: 'rgba(167,139,250,0.10)', // violet 單位方格填色
} as const

type Pt = [number, number]

interface Viewport {
  toScreen: (wx: number, wy: number) => Pt
  toWorld: (px: number, py: number) => Pt
}

function makeViewport(size: number): Viewport {
  const scale = size / 2 / WORLD_HALF
  const cx = size / 2
  const cy = size / 2
  return {
    // 世界座標 → 螢幕 px;螢幕 y 向下,故翻轉。
    toScreen: (wx, wy) => [cx + wx * scale, cy - wy * scale],
    toWorld: (px, py) => [(px - cx) / scale, (cy - py) / scale],
  }
}

// ── 純繪圖 helper(JS 只畫圖,任何「變換點」都來自 WASM)─────────
function stroke(ctx: CanvasRenderingContext2D, p0: Pt, p1: Pt) {
  ctx.beginPath()
  ctx.moveTo(p0[0], p0[1])
  ctx.lineTo(p1[0], p1[1])
  ctx.stroke()
}

function dot(ctx: CanvasRenderingContext2D, p: Pt, color: string) {
  ctx.save()
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.arc(p[0], p[1], 4, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

function label(ctx: CanvasRenderingContext2D, text: string, p: Pt, color: string) {
  ctx.save()
  ctx.fillStyle = color
  ctx.font = '12px ui-monospace, monospace'
  ctx.fillText(text, p[0] + 8, p[1] - 6)
  ctx.restore()
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  from: Pt,
  to: Pt,
  color: string,
  lineWidth: number,
) {
  const dx = to[0] - from[0]
  const dy = to[1] - from[1]
  const len = Math.hypot(dx, dy)
  ctx.save()
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = lineWidth
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  stroke(ctx, from, to)
  if (len > 6) {
    const ang = Math.atan2(dy, dx)
    const head = 10
    const spread = 0.45
    ctx.beginPath()
    ctx.moveTo(to[0], to[1])
    ctx.lineTo(to[0] - head * Math.cos(ang - spread), to[1] - head * Math.sin(ang - spread))
    ctx.lineTo(to[0] - head * Math.cos(ang + spread), to[1] - head * Math.sin(ang + spread))
    ctx.closePath()
    ctx.fill()
  }
  ctx.restore()
}

// 畫世界線段在 A 之下的像:A 為線性,只需 transformPoint 兩端點再連直線。
function drawImageSegment(
  ctx: CanvasRenderingContext2D,
  linalg: Linalg,
  m: Matrix2x2,
  vp: Viewport,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
) {
  const p = linalg.transformPoint(m.a, m.b, m.c, m.d, x0, y0)
  const q = linalg.transformPoint(m.a, m.b, m.c, m.d, x1, y1)
  if (![p[0], p[1], q[0], q[1]].every(Number.isFinite)) return
  stroke(ctx, vp.toScreen(p[0], p[1]), vp.toScreen(q[0], q[1]))
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  m: Matrix2x2,
  v: Vec2,
  size: number,
  dpr: number,
) {
  // resize 會重置 context transform,故每次重設 dpr 縮放(之後一律用 CSS px 作畫)。
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, size, size)
  ctx.fillStyle = COLORS.bg
  ctx.fillRect(0, 0, size, size)

  const S = vp.toScreen
  const origin = S(0, 0)

  // 1. 參考網格 + 座標軸(純 JS 螢幕運算)
  ctx.save()
  ctx.lineWidth = 1
  ctx.strokeStyle = COLORS.refGrid
  for (let k = -WORLD_HALF; k <= WORLD_HALF; k++) {
    stroke(ctx, S(k, -WORLD_HALF), S(k, WORLD_HALF))
    stroke(ctx, S(-WORLD_HALF, k), S(WORLD_HALF, k))
  }
  ctx.strokeStyle = COLORS.axis
  ctx.lineWidth = 1.5
  stroke(ctx, S(-WORLD_HALF, 0), S(WORLD_HALF, 0))
  stroke(ctx, S(0, -WORLD_HALF), S(0, WORLD_HALF))
  ctx.restore()

  // 2. 變換後網格(每條線的像都靠 WASM transformPoint 算端點)
  ctx.save()
  for (let k = -GRID_N; k <= GRID_N; k++) {
    ctx.strokeStyle = k === 0 ? COLORS.tGridAxis : COLORS.tGrid
    ctx.lineWidth = k === 0 ? 1.8 : 1
    drawImageSegment(ctx, linalg, m, vp, k, -GRID_N, k, GRID_N) // x=k 的像
    drawImageSegment(ctx, linalg, m, vp, -GRID_N, k, GRID_N, k) // y=k 的像
  }
  ctx.restore()

  // 3. 單位方格的像(平行四邊形)。遠角 î'+ĵ' 由 WASM 取得,JS 不做加法。
  const iTip = S(m.a, m.c)
  const jTip = S(m.b, m.d)
  const farW = linalg.transformPoint(m.a, m.b, m.c, m.d, 1, 1)
  if ([m.a, m.c, m.b, m.d, farW[0], farW[1]].every(Number.isFinite)) {
    const far = S(farW[0], farW[1])
    ctx.save()
    ctx.fillStyle = COLORS.square
    ctx.beginPath()
    ctx.moveTo(origin[0], origin[1])
    ctx.lineTo(iTip[0], iTip[1])
    ctx.lineTo(far[0], far[1])
    ctx.lineTo(jTip[0], jTip[1])
    ctx.closePath()
    ctx.fill()
    ctx.restore()
  }

  // 4. 基底箭頭 î'、ĵ'(端點即拖曳 handle,畫小圓點示意)
  drawArrow(ctx, origin, iTip, COLORS.iHat, 2.5)
  drawArrow(ctx, origin, jTip, COLORS.jHat, 2.5)
  dot(ctx, iTip, COLORS.iHat)
  dot(ctx, jTip, COLORS.jHat)
  label(ctx, "î'", iTip, COLORS.iHat)
  label(ctx, "ĵ'", jTip, COLORS.jHat)

  // 5. v 與 A·v(A·v 由 WASM 算;平行則高亮特徵方向)
  const vLen = Math.hypot(v.x, v.y)
  const vTip = S(v.x, v.y)
  const av = linalg.transformPoint(m.a, m.b, m.c, m.d, v.x, v.y)
  const avTip = S(av[0], av[1])
  const parallel = vLen >= V_EPS && linalg.areParallel(v.x, v.y, av[0], av[1])

  if (parallel) {
    ctx.save()
    ctx.strokeStyle = COLORS.eigen
    ctx.globalAlpha = 0.4
    ctx.setLineDash([6, 5])
    const ux = v.x / vLen
    const uy = v.y / vLen
    stroke(ctx, S(-WORLD_HALF * ux, -WORLD_HALF * uy), S(WORLD_HALF * ux, WORLD_HALF * uy))
    ctx.restore()
  }

  if (vLen >= V_EPS) {
    drawArrow(ctx, origin, avTip, COLORS.av, parallel ? 4 : 2.5)
    drawArrow(ctx, origin, vTip, COLORS.v, parallel ? 4 : 2.5)
    label(ctx, 'A·v', avTip, COLORS.av)
    label(ctx, 'v', vTip, COLORS.v)
  }
  dot(ctx, vTip, COLORS.v) // 近原點時只剩這顆淡點,仍可被抓回來
}

// ── 拖曳 hit-test ────────────────────────────────────────────
type HandleId = 'iHat' | 'jHat' | 'v'
interface Handle {
  id: HandleId
  sx: number
  sy: number
  priority: number // 小者優先(平手時 v > iHat > jHat)
}

function hitTest(px: number, py: number, handles: Handle[], threshold: number): HandleId | null {
  let best: Handle | null = null
  let bestDist = Infinity
  for (const h of handles) {
    const d = Math.hypot(px - h.sx, py - h.sy)
    if (d > threshold) continue
    if (d < bestDist || (d === bestDist && (best === null || h.priority < best.priority))) {
      best = h
      bestDist = d
    }
  }
  return best ? best.id : null
}

export function TransformCanvas({ linalg, m, v, onChangeMatrix, onChangeV }: TransformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef<HandleId | null>(null)
  const [size, setSize] = useState(0)

  // 容器寬度驅動正方形 canvas
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0
      setSize(Math.max(1, Math.floor(w)))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // 重畫:m / v / size 任一變動就重繪(useLayoutEffect 避免 resize 一幀閃爍)
  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size <= 0) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(size * dpr)
    canvas.height = Math.round(size * dpr)
    canvas.style.width = `${size}px`
    canvas.style.height = `${size}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    drawScene(ctx, makeViewport(size), linalg, m, v, size, dpr)
  }, [linalg, m, v, size])

  // handlers 由 React 每次 render 重建,必看到最新 m/v(無 stale closure)
  const buildHandles = (vp: Viewport): Handle[] => {
    const [vx, vy] = vp.toScreen(v.x, v.y)
    const [ix, iy] = vp.toScreen(m.a, m.c)
    const [jx, jy] = vp.toScreen(m.b, m.d)
    return [
      { id: 'v', sx: vx, sy: vy, priority: 0 },
      { id: 'iHat', sx: ix, sy: iy, priority: 1 },
      { id: 'jHat', sx: jx, sy: jy, priority: 2 },
    ]
  }

  const pointerPos = (e: React.PointerEvent<HTMLCanvasElement>): Pt => {
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
      // hover affordance
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
