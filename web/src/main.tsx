import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider, createRouter } from '@tanstack/react-router'
import { QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { routeTree } from './routeTree.gen'
import { createQueryClient } from './lib/query-client'
import './index.css'

const queryClient = createQueryClient()

const router = createRouter({
  routeTree,
  // 把 queryClient 注入 router context(型別在 __root.tsx 宣告),
  // 讓 route loader 能直接用它做資料 prefetch。
  context: { queryClient },
  // 不讓 Router 自己快取 preload 結果,快取一律交給 TanStack Query 管。
  defaultPreloadStaleTime: 0,
  // Wrap 把整棵 router 樹包進 QueryClientProvider,讓元件能用 useQuery。
  Wrap: ({ children }) => (
    <QueryClientProvider client={queryClient}>
      {children}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  ),
})

// 讓 Link / useNavigate 等 API 取得全域型別推導(型別安全路由的關鍵)。
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
