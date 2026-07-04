// 固定 2×2 的頁面在共享矩陣為 n×n(n>2)時的提示:本頁只讀寫左上子塊,
// 避免「URL 是 3×3 但頁面顯示 2×2」的困惑。n 不存在或 ≤2 時不渲染。
export function SharedMatrixNote({ n }: { n?: number }) {
  if (!n || n <= 2) return null
  return (
    <p className="text-xs text-slate-500">
      共享矩陣目前是 {n}×{n}:本頁讀寫它的左上 2×2 子塊,完整矩陣可到
      「可逆性」或「行列式」頁檢視。
    </p>
  )
}
