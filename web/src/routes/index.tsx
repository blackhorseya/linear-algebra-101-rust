import { createFileRoute } from '@tanstack/react-router'

// 檔名 index.tsx → 對應路徑 '/'。新增 about.tsx 就會自動產生 '/about'。
export const Route = createFileRoute('/')({
  component: Home,
})

const STACK = [
  ['React 19', 'UI 函式庫'],
  ['Vite', '開發伺服器與打包'],
  ['TypeScript', '型別安全'],
  ['Tailwind CSS v4', 'utility-first 樣式'],
  ['TanStack Router', '型別安全的 file-based 路由'],
  ['TanStack Query', 'server-state 與資料快取'],
] as const

function Home() {
  return (
    <section className="space-y-8">
      <div className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight text-slate-50">
          線性代數 101 · Web
        </h1>
        <p className="text-slate-400">
          這是 Rust 線性代數 library 的前端 playground，腳手架已就緒。
        </p>
      </div>

      <ul className="grid gap-3 sm:grid-cols-2">
        {STACK.map(([name, desc]) => (
          <li
            key={name}
            className="rounded-lg border border-slate-800 bg-slate-900/50 px-4 py-3"
          >
            <p className="font-medium text-slate-100">{name}</p>
            <p className="text-sm text-slate-400">{desc}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}
