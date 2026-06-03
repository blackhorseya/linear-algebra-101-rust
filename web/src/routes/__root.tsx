import {
  createRootRouteWithContext,
  Link,
  Outlet,
} from '@tanstack/react-router'
import { TanStackRouterDevtools } from '@tanstack/react-router-devtools'
import type { QueryClient } from '@tanstack/react-query'

// 宣告 router context 的型別:每個 route 的 loader / beforeLoad 都能型別安全地
// 拿到這個 queryClient 去 prefetch 資料。實際的值在 src/main.tsx 注入。
interface RouterContext {
  queryClient: QueryClient
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
})

function RootLayout() {
  return (
    <div className="min-h-svh bg-slate-950 text-slate-200">
      <header className="border-b border-slate-800">
        <nav className="mx-auto flex max-w-4xl items-center gap-6 px-6 py-4">
          <span className="font-semibold text-slate-50">線性代數 101</span>
          <div className="flex gap-4 text-sm">
            {/* TanStack Router 會在符合的 Link 上自動掛 `active` class,
                這裡用 Tailwind 的 `[&.active]:` 變體上色。 */}
            <Link
              to="/"
              activeOptions={{ exact: true }}
              className="text-slate-400 transition hover:text-slate-50 [&.active]:text-violet-400"
            >
              首頁
            </Link>
            <Link
              to="/transform"
              className="text-slate-400 transition hover:text-slate-50 [&.active]:text-violet-400"
            >
              2D 變換
            </Link>
          </div>
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
