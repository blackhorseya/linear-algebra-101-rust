import {
  createRootRouteWithContext,
  Link,
  Outlet,
  retainSearchParams,
} from '@tanstack/react-router'
import { TanStackRouterDevtools } from '@tanstack/react-router-devtools'
import type { QueryClient } from '@tanstack/react-query'
import { decodeMatrix } from '../lib/shared-matrix'

// 宣告 router context 的型別:每個 route 的 loader / beforeLoad 都能型別安全地
// 拿到這個 queryClient 去 prefetch 資料。實際的值在 src/main.tsx 注入。
interface RouterContext {
  queryClient: QueryClient
}

// retainSearchParams 複製的是解析後的「原始」search(不是 validateSearch
// 過濾後的結果),手改網址塞進來的壞 m 會一直黏著跨頁旅行 —— 包一層在每次
// 導航時把無效的 m 整個移除。兩個細節:(1) 必須排在 retainSearchParams
// **之前**,才會作用在它回填後的最終結果上;(2) 移除要用解構拿掉 key,
// 寫 `m: undefined` 會留下 own property,retain 看到 key 存在就不回填了。
function dropInvalidM({
  search,
  next,
}: {
  search: { m?: string }
  next: (search: { m?: string }) => { m?: string }
}): { m?: string } {
  const out = next(search)
  if (typeof out.m === 'string' && decodeMatrix(out.m) !== null) return out
  const rest = { ...out }
  delete rest.m
  return rest
}

export const Route = createRootRouteWithContext<RouterContext>()({
  // 跨頁共享矩陣 ?m= 掛在 root:schema 由所有子路由繼承;retainSearchParams
  // 讓既有 <Link>(不帶 search)導航時自動保留 m —— 頁面連結零改動。
  // 手寫 validate(最小依賴,不引 zod):壞值直接洗掉,頁面回各自教學預設。
  validateSearch: (search: Record<string, unknown>): { m?: string } =>
    typeof search.m === 'string' && decodeMatrix(search.m) !== null
      ? { m: search.m }
      : {},
  search: { middlewares: [dropInvalidM, retainSearchParams(['m'])] },
  component: RootLayout,
})

function RootLayout() {
  return (
    <div className="min-h-svh bg-slate-950 text-slate-200">
      <header className="border-b border-slate-800">
        <nav className="mx-auto flex max-w-4xl items-center px-6 py-4">
          {/* 功能入口集中在首頁(routes/index.tsx 的 FEATURES),
              header 只留品牌名作為回首頁的 Link。 */}
          <Link
            to="/"
            className="font-semibold text-slate-50 transition hover:text-violet-300"
          >
            線性代數 101
          </Link>
        </nav>
      </header>

      <main className="mx-auto max-w-4xl px-6 py-10">
        <Outlet />
      </main>

      {/* 只在 dev 顯示;production build 會被 tree-shake 掉 */}
      <TanStackRouterDevtools position="bottom-right" />
    </div>
  )
}
