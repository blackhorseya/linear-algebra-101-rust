// 視覺化頁面共用的小 UI 元件。

export function Status({ children }: { children: React.ReactNode }) {
  return <p className="text-slate-400">{children}</p>
}

export function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="font-mono text-slate-100">{children}</span>
    </div>
  )
}

export function NumberField({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (value: number) => void
}) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-slate-400">{label}</span>
      <input
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-24 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100 focus:border-violet-500 focus:outline-none"
      />
    </label>
  )
}
