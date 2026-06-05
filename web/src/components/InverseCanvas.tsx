import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Linalg } from '../lib/linalg'
import {
  beginFrame,
  dot,
  drawArrow,
  drawImageSegment,
  drawReferenceGrid,
  GRID_N,
  label,
  makeViewport,
  useSquareSize,
  type Matrix2x2,
  type Viewport,
} from '../lib/canvas'

interface InverseCanvasProps {
  linalg: Linalg
  /**
   * 目標矩陣 = 當前步驟的 working(Eₖ⋯E₁·A)。步驟切換時 Canvas 自己從
   * 「畫面當下的矩陣」tween 過去,呈現「基本矩陣逐步把 A 變回 I」的漸變。
   */
  target: Matrix2x2
}

// invertibility 專屬色票(結構色在 canvas.ts 的 BASE_COLORS);沿 transform 頁的語意:
// 網格 violet、基底 emerald/red、單位方格的像依 det 正負換色(det<0 = 翻面)。
const COLORS = {
  tGrid: '#4c4368', // 變換後網格(violet 偏暗)
  tGridAxis: '#7c6db0', // 變換後座標軸的像(亮一點)
  iHat: '#34d399', // emerald-400
  jHat: '#f87171', // red-400
  squarePos: 'rgba(167,139,250,0.12)', // det>0:violet(定向不變)
  squareNeg: 'rgba(251,191,36,0.12)', // det<0:amber(平面翻面)
} as const

/** 步間漸變時長。短於自動播放間隔(700ms),tween 不會堆積。 */
const TWEEN_MS = 400

/** ease-in-out(二次):起終平滑,中段等速。 */
function ease(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - (2 - 2 * t) ** 2 / 2
}

/**
 * 畫出 working 矩陣作為 2D 變換的像:變換後網格 + 單位方格的像 + 基底箭頭。
 * 端點一律由 WASM `transformPoint` 計算(JS 不重寫線代);det≈0 塌陷時
 * `drawImageSegment` 對非有限端點自動略過。
 */
function drawWorking(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  linalg: Linalg,
  m: Matrix2x2,
) {
  const S = vp.toScreen
  const origin = S(0, 0)

  // 變換後網格(每條線的像都由 WASM transformPoint 算端點)
  ctx.save()
  for (let k = -GRID_N; k <= GRID_N; k++) {
    ctx.strokeStyle = k === 0 ? COLORS.tGridAxis : COLORS.tGrid
    ctx.lineWidth = k === 0 ? 1.8 : 1
    drawImageSegment(ctx, linalg.transformPoint, m, vp, k, -GRID_N, k, GRID_N) // x=k 的像
    drawImageSegment(ctx, linalg.transformPoint, m, vp, -GRID_N, k, GRID_N, k) // y=k 的像
  }
  ctx.restore()

  // 單位方格的像(平行四邊形):面積 = |det|,det<0 翻面換色(det 由 WASM 算)。
  // swap 步的 lerp 中途 det 會過 0(平面瞬間塌平再翻面)—— 這是鏡射如實的幾何。
  const iTip = S(m.a, m.c)
  const jTip = S(m.b, m.d)
  const farW = linalg.transformPoint(m.a, m.b, m.c, m.d, 1, 1)
  if ([m.a, m.c, m.b, m.d, farW[0], farW[1]].every(Number.isFinite)) {
    const det = linalg.determinant(m.a, m.b, m.c, m.d)
    const far = S(farW[0], farW[1])
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

  // 基底箭頭 î'、ĵ'(working 的行向量;化到 Iₙ 時回到標準基底)
  drawArrow(ctx, origin, iTip, COLORS.iHat, 2.5)
  drawArrow(ctx, origin, jTip, COLORS.jHat, 2.5)
  dot(ctx, iTip, COLORS.iHat)
  dot(ctx, jTip, COLORS.jHat)
  label(ctx, "î'", iTip, COLORS.iHat)
  label(ctx, "ĵ'", jTip, COLORS.jHat)
}

/**
 * 「基本矩陣逐步把 A 變回 I」的幾何視圖(n = 2 專用,controlled、不可拖曳 ——
 * 矩陣來自頁面的輸入網格與步驟狀態)。
 *
 * tween 不持有步驟狀態:`target` 變動時以 rAF 在 TWEEN_MS 內把「畫面當下的矩陣」
 * 線性插值到目標,slider 快速拖曳 / 自動播放都自然銜接(每次從當前畫面出發)。
 */
export function InverseCanvas({ linalg, target }: InverseCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const size = useSquareSize(containerRef)

  // displayed 是唯一驅動重畫的 state;動畫起點走 ref(讀「畫面當下」不觸發 render)。
  // ref 的同步寫入放在下方的重畫 useLayoutEffect 裡(render 期間不可寫 ref),
  // 它先於 useEffect 執行 —— tween effect 讀到的必是最新畫面。
  const [displayed, setDisplayed] = useState<Matrix2x2>(target)
  const displayedRef = useRef(displayed)

  useEffect(() => {
    const from = displayedRef.current
    const dest = target
    if (
      from.a === dest.a &&
      from.b === dest.b &&
      from.c === dest.c &&
      from.d === dest.d
    )
      return // 已在目標上(如初次 mount):不起動畫
    let raf = 0
    const t0 = performance.now()
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / TWEEN_MS)
      const k = ease(t)
      setDisplayed({
        a: from.a + (dest.a - from.a) * k,
        b: from.b + (dest.b - from.b) * k,
        c: from.c + (dest.c - from.c) * k,
        d: from.d + (dest.d - from.d) * k,
      })
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf) // 目標再變 / 卸載:停掉舊動畫
  }, [target])

  useLayoutEffect(() => {
    displayedRef.current = displayed // 同步「畫面當下」給 tween effect 當起點
    const canvas = canvasRef.current
    if (!canvas || size <= 0) return
    const ctx = beginFrame(canvas, size)
    if (!ctx) return
    const vp = makeViewport(size)
    drawReferenceGrid(ctx, vp)
    drawWorking(ctx, vp, linalg, displayed)
  }, [linalg, displayed, size])

  return (
    <div ref={containerRef} className="aspect-square w-full max-w-xl">
      <canvas ref={canvasRef} className="rounded-lg border border-slate-800" />
    </div>
  )
}
