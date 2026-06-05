import type { ReactNode } from 'react'

/**
 * 逐步播放控制:進度條 + 首尾 / 單步跳轉,可選自動播放(▶ / ⏸)。
 *
 * 從 elimination 頁抽出共用。`isPlaying` / `onTogglePlay` 都有給才會顯示
 * 播放鈕 —— 不需要自動播放的頁面照舊只傳前三個 props,行為與抽出前相同。
 */
export function PlaybackControls({
  step,
  count,
  onChange,
  isPlaying,
  onTogglePlay,
}: {
  step: number
  count: number
  onChange: (step: number) => void
  isPlaying?: boolean
  onTogglePlay?: () => void
}) {
  return (
    <div className="space-y-3 border-t border-slate-800 pt-4">
      <input
        type="range"
        min={0}
        max={Math.max(0, count - 1)}
        value={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-violet-500"
      />
      <div className="flex items-center justify-center gap-2">
        {onTogglePlay && (
          <NavButton onClick={onTogglePlay} disabled={count <= 1}>
            {isPlaying ? '⏸ 暫停' : '▶ 播放'}
          </NavButton>
        )}
        <NavButton onClick={() => onChange(0)} disabled={step === 0}>
          ⏮ 最前
        </NavButton>
        <NavButton
          onClick={() => onChange(Math.max(0, step - 1))}
          disabled={step === 0}
        >
          ← 上一步
        </NavButton>
        <NavButton
          onClick={() => onChange(Math.min(count - 1, step + 1))}
          disabled={step >= count - 1}
        >
          下一步 →
        </NavButton>
        <NavButton
          onClick={() => onChange(count - 1)}
          disabled={step >= count - 1}
        >
          最後 ⏭
        </NavButton>
      </div>
    </div>
  )
}

function NavButton({
  onClick,
  disabled,
  children,
}: {
  onClick: () => void
  disabled?: boolean
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded border border-slate-700 px-3 py-1.5 text-sm text-slate-200 transition hover:border-violet-500 hover:text-violet-300 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {children}
    </button>
  )
}
