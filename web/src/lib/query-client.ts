import { QueryClient } from '@tanstack/react-query'

/**
 * 建立整個 app 共用的 QueryClient。
 *
 * 這個前端沒有遠端 API,所有「資料」都是本地、確定性的線性代數計算
 * (同樣的輸入恆得同樣的輸出)。Query 在這裡的角色不是同步 server-state,
 * 而是「記憶化昂貴的本地計算」。快取策略因此與打 API 的情境相反 ——
 * 算過一次就鎖住,永不在背景重抓:
 */
export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // 結果是確定性的,永遠不會「過期」→ 不必背景重新計算
        staleTime: Infinity,
        // 失敗代表程式邏輯錯(非暫時性網路問題),重試只會重現同樣的錯
        retry: false,
        // 沒有遠端資料會悄悄變動,切回分頁 / 重連都不需要重抓
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
        // gcTime 維持預設(5 分鐘):沒人訂閱的快取仍會回收,避免記憶體無限長大。
        // 重算成本很低,寧可釋放也不囤積。
      },
    },
  })
}
