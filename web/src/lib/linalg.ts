import init, {
  transform_point,
  are_parallel,
} from "./wasm/linear_algebra_101.js";
// `--target web` 的 glue 不會自動 import .wasm,要把它的 URL 交給 init()。
// `?url` 讓 Vite 把這顆 wasm 當資產處理並回傳可 fetch 的網址(dev / build 皆然)。
import wasmUrl from "./wasm/linear_algebra_101_bg.wasm?url";

/** 初始化後可用的線代運算(全部在 Rust 算,JS 只是轉呼叫)。 */
export interface Linalg {
  /** 2×2 矩陣 `[[a,b],[c,d]]` 作用在點 `(x,y)`,回傳變換後的 `[x', y']`。 */
  transformPoint: (
    a: number,
    b: number,
    c: number,
    d: number,
    x: number,
    y: number,
  ) => Float64Array;
  /** 兩個 2D 向量是否平行(共線 / 線性相依)。 */
  areParallel: (ux: number, uy: number, wx: number, wy: number) => boolean;
}

// 模組層級 memoize:init 是非同步且只該跑一次。即使多個元件同時呼叫,
// 也共用同一個 Promise(配合 Query 的 staleTime: Infinity 是雙重保險)。
let instance: Promise<Linalg> | null = null;

/** 載入並初始化 WASM 模組,回傳綁好的運算 API。重複呼叫共用同一次初始化。 */
export function loadLinalg(): Promise<Linalg> {
  // 已初始化就早退(讓 TS 在後續流程把 instance 收窄為非 null)。
  if (instance) return instance;
  instance = init({ module_or_path: wasmUrl }).then(() => ({
    transformPoint: transform_point,
    areParallel: are_parallel,
  }));
  return instance;
}
