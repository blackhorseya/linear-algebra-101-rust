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
