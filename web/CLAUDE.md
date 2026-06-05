# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本檔只講 `web/` 內部的架構與慣例。跨 repo 的鐵律(core 零改動、計算單一真相在 Rust、最小依賴、pnpm、Vercel 部署坑)在**根目錄的 `CLAUDE.md`**,先讀那份。

## 常用指令

| 指令 | 作用 |
|------|------|
| `pnpm dev` | Vite dev server(HMR) |
| `pnpm build` | `tsc -b` 型別檢查 + Vite 打包 |
| `pnpm lint` | ESLint(flat config) |
| `pnpm exec tsc -b` | 只跑型別檢查,不打包 |

沒有 test runner:數學正確性由 Rust core 的測試保證(`task test`),前端只是繪圖與互動層,目前不另寫 JS 測試。

**先決條件**:`src/lib/wasm/` 是 wasm-pack 產物且整個被 gitignore —— fresh clone 直接 `pnpm dev` 會掛。先在 **repo 根目錄**跑 `task wasm:build` 產出它;改了 `src/wasm.rs` 之後也要重跑。

## 架構:資料怎麼流

```
src/routes/*.tsx(頁面,持有狀態)
  → useQuery(['linalg'], loadLinalg)     # 唯一非同步點:WASM 初始化
  → Linalg 介面(src/lib/linalg.ts)       # 同步呼叫,計算全在 Rust
  → src/lib/wasm/*(wasm-bindgen glue,生成物)
```

### WASM 邊界:全部鎖在 `src/lib/linalg.ts`

- **只有 `linalg.ts` 可以 import `src/lib/wasm/`**。其他程式碼一律透過它 export 的 `Linalg` 介面取用運算,看不到 wasm-bindgen 的型別。
- `loadLinalg()` 用模組層級變數 memoize init Promise(配 Query 的 `staleTime: Infinity` 雙重保險),整個 app 只初始化一次。`.wasm` 用 Vite 的 `?url` import 餵給 `init()`。
- **複雜回傳值用 SoA + `free()` 模式**(見 `runEliminate`):WASM 端以 Structure-of-Arrays 平行陣列輸出(每個欄位一條 typed array,少跨界),`linalg.ts` 縫回 plain-JS 物件後立刻 `free()` —— 呼叫端拿到的物件不持有任何 WASM 指標,不必管生命週期。WASM 端的 u8 編碼(phase、solution kind)→ 字串的對照表順序必須與 `src/wasm.rs` 一致。
- **新增一個 binding 的流程**:`src/wasm.rs` 加 export(根目錄那邊的規矩見根 CLAUDE.md)→ `task wasm:build` → 在 `linalg.ts` 的 `Linalg` 介面補型別與綁定 → 頁面經 `linalg.xxx()` 使用。

### Query 的角色:memoize 本地計算,不是同步 server state

`src/lib/query-client.ts` 的設定與打 API 的情境**相反**(`staleTime: Infinity`、`retry: false`、不 refetch),因為所有「資料」都是確定性的本地線代計算。慣例:

- WASM **載入**交給 Query 管 loading / error,所有頁面共用同一個 `queryKey: ['linalg']`。
- 載入完成後的**運算呼叫是同步的**,直接在 render 裡呼叫(2×2 乘法很便宜),不要再各包一層 Query;較重的計算(如消去 trace)用 `useMemo` 就好。

### Routing:TanStack Router file-based

- 新頁面 = `src/routes/<name>.tsx` 一支(`createFileRoute('/<name>')`)+ 在 `__root.tsx` 的 nav 加 `Link`。
- `routeTree.gen.ts` 由 vite plugin 自動產生 —— **有進版控但永遠不要手改**,ESLint 也忽略它。vite.config.ts 中 `tanstackRouter()` 必須排在 `react()` 之前。
- Router context 型別在 `__root.tsx` 宣告(`{ queryClient }`),值在 `main.tsx` 注入;route 檔依慣例 export `Route` 物件,該目錄已關閉 `react-refresh/only-export-components`。

### Canvas 視覺化慣例

- **共用繪圖原語在 `src/lib/canvas.ts`**:世界↔螢幕座標映射(`makeViewport`)、HiDPI 設定(`beginFrame`)、箭頭/網格/拖曳 hit-test、容器自適應(`useSquareSize`)。新視覺化先看這裡有沒有現成的。
- **Canvas 元件全 controlled**(`src/components/*Canvas.tsx`):狀態(矩陣、向量)住在 route 頁面,Canvas 收 `linalg` + 狀態 + `onChange` callbacks,在 `useLayoutEffect` 裡整幀重畫。數字輸入框與 Canvas 共用同一份 state —— 單一真相。
- 拖曳用 pointer events + `setPointerCapture`,handle 命中靠 `hitTest`(近者優先、平手看 priority)。
- **顏色用 hex token,不用 Tailwind class**(canvas 吃不到):結構色(背景、網格、軸)集中在 `canvas.ts` 的 `BASE_COLORS`,各視覺化專屬色票在自己元件內的 `COLORS`,數值對齊 Tailwind 色票(slate/violet 暗色主題)並註記對應名稱。
- 畫「變換後的像」時,端點一律由 WASM `transformPoint` 算(JS 不做向量加法/乘法),非有限值(det≈0 塌陷)就略過該線段。

### 共用 UI

小元件(`Status` / `Row` / `NumberField`)在 `src/components/ui.tsx`;數字格式化用 `src/lib/format.ts` 的 `fmt`。頁面文案是教學導向的繁體中文,新頁面延續同樣的解說口吻(把公式攤開、帶入實際數字)。
