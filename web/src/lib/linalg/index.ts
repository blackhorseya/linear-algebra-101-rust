// WASM 邊界的唯一入口 —— 其他程式碼一律 import '../lib/linalg',看不到
// wasm-bindgen 的型別。結構鏡像 src/wasm/:一個視覺化章一支子模組,本檔只負責
// 組裝與 re-export(對應 mod.rs 的角色);跨章共用的縫合工具在私有的 helpers.ts。
import init from "../wasm/linear_algebra_101.js";
// `--target web` 的 glue 不會自動 import .wasm,要把它的 URL 交給 init()。
// `?url` 讓 Vite 把這顆 wasm 當資產處理並回傳可 fetch 的網址(dev / build 皆然)。
import wasmUrl from "../wasm/linear_algebra_101_bg.wasm?url";

import { compositionOps, type CompositionOps } from "./composition";
import { coordinatesOps, type CoordinatesOps } from "./coordinates";
import { determinantOps, type DeterminantOps } from "./determinant";
import { eliminationOps, type EliminationOps } from "./elimination";
import { inverseOps, type InverseOps } from "./inverse";
import { linearityOps, type LinearityOps } from "./linearity";
import { multiplyOps, type MultiplyOps } from "./multiply";
import { operatorOps, type OperatorOps } from "./operator";
import { rangeOps, type RangeOps } from "./range";
import { standardMatrixOps, type StandardMatrixOps } from "./standard-matrix";
import { subspaceOps, type SubspaceOps } from "./subspace";
import { transformOps, type TransformOps } from "./transform";

// 各章的公開型別一律從本檔 re-export,呼叫端只認 '../lib/linalg' 一個路徑。
export type { TransformationReportJS } from "./composition";
export type {
  ElimPhase,
  EliminationStepJS,
  EliminationTraceJS,
  SolutionKind,
} from "./elimination";
export type { EroKind, InverseStepJS, InverseTraceJS } from "./inverse";
export type { MultiplyExpansionJS } from "./multiply";
export type { SolveResult } from "./range";
export type { RuleKind } from "./standard-matrix";

/**
 * 初始化後可用的線代運算(全部在 Rust 算,JS 只是轉呼叫)。
 *
 * 各章的運算介面攤平合併在同一層 —— 鏡像 wasm 端「`#[wasm_bindgen]` 匯出攤平
 * 在套件根層、與模組巢狀無關」的慣例,所以拆章對呼叫端 API 零影響。
 */
export interface Linalg
  extends TransformOps,
    DeterminantOps,
    MultiplyOps,
    EliminationOps,
    InverseOps,
    LinearityOps,
    StandardMatrixOps,
    RangeOps,
    CompositionOps,
    SubspaceOps,
    CoordinatesOps,
    OperatorOps {}

// 模組層級 memoize:init 是非同步且只該跑一次。即使多個元件同時呼叫,
// 也共用同一個 Promise(配合 Query 的 staleTime: Infinity 是雙重保險)。
let instance: Promise<Linalg> | null = null;

/** 載入並初始化 WASM 模組,回傳綁好的運算 API。重複呼叫共用同一次初始化。 */
export function loadLinalg(): Promise<Linalg> {
  // 已初始化就早退(讓 TS 在後續流程把 instance 收窄為非 null)。
  if (instance) return instance;
  instance = init({ module_or_path: wasmUrl }).then(() => ({
    ...transformOps,
    ...determinantOps,
    ...multiplyOps,
    ...eliminationOps,
    ...inverseOps,
    ...linearityOps,
    ...standardMatrixOps,
    ...rangeOps,
    ...compositionOps,
    ...subspaceOps,
    ...coordinatesOps,
    ...operatorOps,
  }));
  return instance;
}
