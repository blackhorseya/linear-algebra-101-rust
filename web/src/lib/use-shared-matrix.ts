import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useRouter } from '@tanstack/react-router'
import { decodeMatrix, encodeMatrix } from './shared-matrix'

// 跨頁共享矩陣的讀寫 hook。
//
// 核心模型:「mount 快照 + 單向 debounce 寫出」。頁面互動的真相永遠是
// 本地 useState;掛載那一刻讀一次 URL,之後不再訂閱 search —— 資料流
// 嚴格單向(URL → 初始值、本地變更 → URL),結構上不可能迴圈,不需要
// 「判斷是不是自己寫的」那套髒 flag。快照為什麼夠:站內導航/上下頁/
// 手改網址列都會讓頁面元件重新 mount(頁內編輯用 replace 不產生
// history entry),掛載期間唯一會改 m 的就是自己的 debounce 寫出。

/** 低於 Safari history.replaceState 節流門檻(約 100 次 / 30 秒)。 */
const DEBOUNCE_MS = 250

export function useSharedMatrix(): {
  /** mount 快照:URL 的共享方陣;沒有或格式壞掉為 null(頁面回自己的教學預設)。 */
  initial: number[][] | null
  /** debounce 寫回 URL(replace,不塞 history);拖曳每 frame 呼叫也只是重設計時器。 */
  publish: (g: number[][]) => void
} {
  const router = useRouter()
  const navigate = useNavigate()

  // useState 惰性初始化 = 「只在掛載那一刻讀一次 URL」的最自然寫法。
  const [initial] = useState<number[][] | null>(() => {
    const { m } = router.state.location.search as { m?: unknown }
    return typeof m === 'string' ? decodeMatrix(m) : null
  })

  const pending = useRef<{ id: number; g: number[][] } | null>(null)

  const write = useCallback(
    (g: number[][]) => {
      void navigate({
        to: '.', // 停在當前 route,只改 search
        replace: true, // 頁內編輯不塞 history(back 才不會踩過 200 步矩陣微調)
        resetScroll: false,
        search: (prev) => ({ ...prev, m: encodeMatrix(g) }),
      })
    },
    [navigate],
  )

  const publish = useCallback(
    (g: number[][]) => {
      if (pending.current) clearTimeout(pending.current.id)
      pending.current = {
        g,
        id: window.setTimeout(() => {
          pending.current = null
          write(g)
        }, DEBOUNCE_MS),
      }
    },
    [write],
  )

  // unmount 時 flush 最後一筆:拖完矩陣 250ms 內就點連結離開也不能掉狀態。
  // setTimeout(0) 避開在 React commit phase 中導航;此時新頁面已載入且
  // retainSearchParams 帶著「舊的 m」過去,functional search update 恰好
  // 把它修正成最新值。
  useEffect(
    () => () => {
      if (!pending.current) return
      clearTimeout(pending.current.id)
      const { g } = pending.current
      pending.current = null
      window.setTimeout(() => write(g), 0)
    },
    [write],
  )

  return { initial, publish }
}
