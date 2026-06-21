import { createFileRoute, Link } from '@tanstack/react-router'

// 檔名 index.tsx → 對應路徑 '/'。新增 about.tsx 就會自動產生 '/about'。
export const Route = createFileRoute('/')({
  component: Home,
})

// 功能入口(原本掛在 header nav,改為首頁卡片)。
// `as const` 讓 `to` 保持字面量型別,Link 的路徑打錯會直接編譯失敗。
// 新增視覺化頁面時在這裡補一筆即可。
const FEATURES = [
  {
    to: '/transform',
    title: '2D 線性變換',
    desc: '拖動向量、調整 2×2 矩陣,看整個平面如何被變形;行列式的幾何意義 = 面積縮放與翻面。',
  },
  {
    to: '/linearity',
    title: '線性轉換與守恆律',
    desc: '拖動 u、v 看 shear / 投影下的影像:T(u+v) 與 T(u)+T(v) 永遠在同一點會合 —— Theorem 2.7「矩陣誘導必線性」看得見。',
  },
  {
    to: '/standard-matrix',
    title: '標準矩陣取樣',
    desc: '選幾何規則(旋轉、反射、剪切⋯),看 e₁、e₂ 的影像被取樣、直放成矩陣的行 —— Theorem 2.9「線性轉換必由唯一矩陣誘導」現場上演。',
  },
  {
    to: '/range',
    title: '值域與映成',
    desc: '拖動矩陣的行向量,看 Range(T) = Col(A) 從整個平面塌成直線;拖 w 問「到得了嗎?」—— 可達性與映成判定由 core 即時計算。',
  },
  {
    to: '/nullspace',
    title: '零空間與 rank-nullity',
    desc: '/range 的對偶:拖輸入 v 看它的像 Av,落到核線上時被壓到原點;nullity 與 rank 各自由 core 算,相加 = 2 當場驗證 rank-nullity 定理。',
  },
  {
    to: '/rank',
    title: '行秩 = 列秩',
    desc: 'Row A(domain)與 Col A(codomain)是不同空間的不同子空間,維度卻永遠相等。拖列向量或行向量,看兩邊的維度同進同退 —— rank(A) = rank(Aᵀ) 看得見。',
  },
  {
    to: '/composition',
    title: '合成與可逆性',
    desc: '拖 x 看「先 T 再 U」兩步路徑與「一步 BA」直達永遠會合(T_B ∘ T_A = T_BA);切到逆轉換模式看「變形 → 復原」,Summary Table 三燈由 core 點亮。',
  },
  {
    to: '/span',
    title: '張成 Span',
    desc: '兩個向量能「張成」多大的空間?拖到共線的瞬間,平面塌縮成一條線。',
  },
  {
    to: '/multiply',
    title: '矩陣乘法',
    desc: '任意尺寸 (m×n)·(n×p) 的 row × col 互動展開:點 C 的任一格看它由哪一列點積哪一欄,並親手碰到「維度不合不能乘」。',
  },
  {
    to: '/elimination',
    title: '高斯消去',
    desc: '逐步播放 forward / backward 消去,看矩陣化成 RREF,並判讀唯一解、無限多解或無解。',
  },
  {
    to: '/invertibility',
    title: '可逆矩陣',
    desc: 'Gauss-Jordan 累乘基本矩陣求 A⁻¹,並以 2D 變換看每個列運算的幾何意義(鏡射、伸縮、剪切)。',
  },
  {
    to: '/determinant',
    title: '行列式',
    desc: '拖 î′、ĵ′ 看單位正方形的像:|det| = 面積縮放、det < 0 = 翻面;切到 3×3 / 4×4 看 det 推廣為(超)體積,det 路與 rank 路對帳 Theorem 3.4(a)。',
  },
] as const

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
    <section className="space-y-10">
      <div className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight text-slate-50">
          線性代數 101 · Web
        </h1>
        <p className="text-slate-400">
          Rust 手刻線性代數 library 的互動圖解 —— 所有計算由 Rust(WASM)完成,JS 只負責畫圖與互動。
        </p>
      </div>

      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-100">互動圖解</h2>
        <ul className="grid gap-3 sm:grid-cols-2">
          {FEATURES.map(({ to, title, desc }) => (
            <li key={to}>
              <Link
                to={to}
                className="block h-full rounded-lg border border-slate-800 bg-slate-900/50 px-4 py-3 transition hover:border-violet-500/60 hover:bg-slate-900"
              >
                <p className="font-medium text-violet-300">{title} →</p>
                <p className="mt-1 text-sm text-slate-400">{desc}</p>
              </Link>
            </li>
          ))}
        </ul>
      </div>

      <div className="space-y-4">
        <h2 className="text-sm font-semibold text-slate-500">技術棧</h2>
        <ul className="grid gap-2 sm:grid-cols-3">
          {STACK.map(([name, desc]) => (
            <li
              key={name}
              className="rounded-md border border-slate-800/60 px-3 py-2"
            >
              <p className="text-sm text-slate-300">{name}</p>
              <p className="text-xs text-slate-500">{desc}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
