import { useEffect, useState, type RefObject } from 'react'

// 跨視覺化共用的 Canvas 2D 基本元件:座標映射、HiDPI、繪圖原語、拖曳 hit-test。
// 純 Canvas,無繪圖庫(延續 repo 的最小依賴精神)。

export type Pt = [number, number]
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

export const WORLD_HALF = 8 // 視口世界半徑(±8 單位都看得到)
export const GRID_N = 6 // 網格線 / 線段範圍(保持畫面可讀)
export const HIT_PX = 12 // 拖曳 handle 命中半徑(CSS px)
export const ORIGIN_EPS = 0.04 // 向量視為「在原點」的門檻(世界單位)

// canvas 不能吃 Tailwind class,結構色集中在此對齊主題色票。
export const BASE_COLORS = {
  bg: '#020617', // slate-950
  refGrid: '#1e293b', // slate-800
  axis: '#475569', // slate-600
  label: '#cbd5e1', // slate-300
} as const

export interface Viewport {
  toScreen: (wx: number, wy: number) => Pt
  toWorld: (px: number, py: number) => Pt
}

export function makeViewport(size: number): Viewport {
  const scale = size / 2 / WORLD_HALF
  const cx = size / 2
  const cy = size / 2
  return {
    // 世界座標 → 螢幕 px;螢幕 y 向下,故翻轉。
    toScreen: (wx, wy) => [cx + wx * scale, cy - wy * scale],
    toWorld: (px, py) => [(px - cx) / scale, (cy - py) / scale],
  }
}

// 量好 HiDPI、清屏、填背景,回傳一個可直接用 CSS px 作畫的 context。
export function beginFrame(canvas: HTMLCanvasElement, size: number): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.round(size * dpr)
  canvas.height = Math.round(size * dpr)
  canvas.style.width = `${size}px`
  canvas.style.height = `${size}px`
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0) // resize 會重置 transform,每幀重設
  ctx.clearRect(0, 0, size, size)
  ctx.fillStyle = BASE_COLORS.bg
  ctx.fillRect(0, 0, size, size)
  return ctx
}

export function strokeLine(ctx: CanvasRenderingContext2D, p0: Pt, p1: Pt) {
  ctx.beginPath()
  ctx.moveTo(p0[0], p0[1])
  ctx.lineTo(p1[0], p1[1])
  ctx.stroke()
}

export function dot(ctx: CanvasRenderingContext2D, p: Pt, color: string) {
  ctx.save()
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.arc(p[0], p[1], 4, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

/** 空心圓環:標記「另一條計算路徑」的落點 —— 與箭頭尖端重合即定理成立
 *(linearity 的守恆律、standard-matrix 的規則 vs 矩陣兩用)。 */
export function drawRing(ctx: CanvasRenderingContext2D, p: Pt, color: string) {
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.arc(p[0], p[1], 7, 0, Math.PI * 2)
  ctx.stroke()
  ctx.restore()
}

export function label(ctx: CanvasRenderingContext2D, text: string, p: Pt, color: string) {
  ctx.save()
  ctx.fillStyle = color
  ctx.font = '12px ui-monospace, monospace'
  ctx.fillText(text, p[0] + 8, p[1] - 6)
  ctx.restore()
}

export function drawArrow(
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
  strokeLine(ctx, from, to)
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

// 參考網格 + 座標軸(純 JS 螢幕運算,兩個視覺化共用)。
export function drawReferenceGrid(ctx: CanvasRenderingContext2D, vp: Viewport) {
  const S = vp.toScreen
  ctx.save()
  ctx.lineWidth = 1
  ctx.strokeStyle = BASE_COLORS.refGrid
  for (let k = -WORLD_HALF; k <= WORLD_HALF; k++) {
    strokeLine(ctx, S(k, -WORLD_HALF), S(k, WORLD_HALF))
    strokeLine(ctx, S(-WORLD_HALF, k), S(WORLD_HALF, k))
  }
  ctx.strokeStyle = BASE_COLORS.axis
  ctx.lineWidth = 1.5
  strokeLine(ctx, S(-WORLD_HALF, 0), S(WORLD_HALF, 0))
  strokeLine(ctx, S(0, -WORLD_HALF), S(0, WORLD_HALF))
  ctx.restore()
}

// 一個 2×2 變換(由 transformPoint 提供)把世界線段 (x0,y0)–(x1,y1) 送到的像。
// A 為線性 → 只需轉兩端點再連直線。端點非有限(det≈0 collapse)則略過。
export function drawImageSegment(
  ctx: CanvasRenderingContext2D,
  transformPoint: (a: number, b: number, c: number, d: number, x: number, y: number) => Float64Array,
  m: Matrix2x2,
  vp: Viewport,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
) {
  const p = transformPoint(m.a, m.b, m.c, m.d, x0, y0)
  const q = transformPoint(m.a, m.b, m.c, m.d, x1, y1)
  if (![p[0], p[1], q[0], q[1]].every(Number.isFinite)) return
  strokeLine(ctx, vp.toScreen(p[0], p[1]), vp.toScreen(q[0], q[1]))
}

export interface Handle<T extends string = string> {
  id: T
  sx: number
  sy: number
  priority: number // 小者優先(平手時用來決定誰被抓到)
}

export function hitTest<T extends string>(
  px: number,
  py: number,
  handles: Handle<T>[],
  threshold: number,
): T | null {
  let best: Handle<T> | null = null
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

// 容器寬度驅動的正方形邊長(CSS px)。供 canvas 自適應縮放。
export function useSquareSize(ref: RefObject<HTMLElement | null>): number {
  const [size, setSize] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0
      setSize(Math.max(1, Math.floor(w)))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return size
}
