# linear-algebra-101-rust

用 **Rust** 從零實作 linear algebra,邊寫邊學。

這個 repo 不依賴 [nalgebra](https://nalgebra.org/) 或 [ndarray](https://docs.rs/ndarray) 之類的數值函式庫,而是親手把 vector、matrix 以及它們的運算刻出來。目的不是做出最快的實作,而是透過程式碼建立對線性代數的直覺 — 當你能用 Rust 的型別與函式表達一個數學概念,代表你真的懂了它。

> 這是 [linear-algebra-101](https://github.com/blackhorseya/linear-algebra-101)(Go 版)的 Rust 改寫。

## 為什麼用 Rust 學線性代數?

- **型別逼你想清楚結構** — `Vector` 與 `Matrix` 是不同型別,維度不合的運算應該在邊界就被擋下來,而不是算到一半才爆。
- **函式即定義** — `linear_combination(scalars, vectors)` 的實作就是「Σ 純量ᵢ · 向量ᵢ」的數學定義,寫出來就懂了。
- **錯誤是值,不是例外** — 維度不合回傳 `Result<_, _>` 而非 panic,呼叫端被型別系統逼著面對「這個運算可能無效」這件事 — 這正是線性代數裡維度條件的精神。
- **測試即證明** — 每個運算都能用已知結果(例如 identity matrix 乘任何 matrix 等於自己)寫成 test,把數學性質變成可驗證的程式;搭配 [`proptest`](https://docs.rs/proptest) 還能把定律變成隨機驗證。

## 學習路徑

`[x]` 已實作(含測試),`[ ]` 尚未動工。本專案嚴格依 [Go 版](https://github.com/blackhorseya/linear-algebra-101) git log 正序逐 commit 移植,進度貼合教材(Lay/Strang 風格,程式碼註解標 Theorem 1.x)—— 所以「向量空間 / 線性獨立」會排在「內積 / 範數」前面,而非直覺順序。

### 1. Vector(向量)
- [x] 加法 `add`、純量乘法 `scale`
- [x] 線性組合 `linear_combination`、標準基底 eᵢ `standard`
- [x] 相等 / 近似相等 `equals` / `approx_equals`、零向量 `is_zero`、平行(共線)`is_parallel`
- [ ] 減法 `sub`、內積 `dot`、長度 / 範數 `norm`、單位向量 `normalize`、夾角 `angle`

### 2. Matrix(矩陣)
- [x] 建構子 `new` / `from_rows`、加法 / 純量乘法、轉置 `transpose`、單位矩陣 `identity`
- [x] 矩陣–向量乘積 `multiply_vector`(A·v,column view 的核心)、column / row 抽取、stochastic 判定 `is_stochastic`
- [x] 基本列運算 EROs(`swap_rows` / `scale_row` / `add_scaled_row`)、列階梯形判定 `is_row_echelon_form` / `is_reduced_row_echelon_form`
- [x] 矩陣乘法 `multiply`(matrix × matrix,線性映射的合成)、維度相容述詞 `can_multiply`、方陣冪 `power`(Aᵏ,A⁰ = I)
- [x] 對角矩陣 `DiagonalMatrix`(parse-don't-validate 的 newtype,O(n) 乘法,只存對角線)

### 3. 向量空間:span、線性獨立、basis、座標
- [x] span:`Span`、`spans_all`、`on_line` / `on_plane` / `affine_span`
- [x] 線性獨立:`is_linearly_independent` / `is_linearly_dependent`、冗餘數 `redundancy_count`、可移除行 `removable_columns`、首個相依索引 `first_dependent_index`
- [x] basis `is_basis` / `is_standard_basis`、基底座標 `coordinates` / `from_coordinates`

### 4. 線性方程組與分解
- [x] 線性方程組 Ax=b:`System`、`solve`、一致性 `is_consistent`、解的分類 `Solution` / `RowKind`
- [x] Gaussian elimination:`row_echelon_form` / `reduced_row_echelon_form`、秩 `rank`、零化度 `nullity`、pivot / free 行
- [x] 基本矩陣 elementary matrices(`elementary_swap` / `elementary_scale` / `elementary_add_scaled`:Iₙ + 一次 ERO;左乘 E = 施作該列運算)
- [x] 可逆判定 `is_invertible`(可逆矩陣定理 IMT:RREF = Iₙ ⟺ rank = n ⟺ 行向量獨立 ⟺ 唯一解⋯⋯等價條件以 laws 互驗)
- [x] 反矩陣 `inverse`(Gauss-Jordan 累乘基本矩陣 P = Eₖ⋯E₁,Theorem 2.3 直接寫成演算法;Theorem 2.2 代數性質以 laws 驗證)
- [ ] 行列式 `determinant`

### 5. 線性轉換(Linear Transformation)
- [x] 矩陣作為函數:`Transformation`(A 誘導 T_A: ℝⁿ → ℝᵐ)、定義域 / 對應域維度 `domain_dim` / `codomain_dim` / `dimensions`
- [x] 映射 `apply`(T_A(x) = Ax,委派 `multiply_vector`,維度檢查隨之繼承)
- [x] 線性性質驗證 `verify_linearity`(T(u+v) = T(u)+T(v)、T(cu) = c·T(u);泛型 `Fn(&Vector) -> Vector` 收任意映射,可識破仿射轉換)
- [x] 單位 / 零轉換 `identity` / `zero`(I(x) = x、T₀(x) = 0;零轉換不必方陣,0 ∈ codomain ℝᵐ)
- [x] Theorem 2.7:矩陣誘導的轉換必為線性(laws,proptest 隨機掃 —— 整數策略配精確 equals、浮點策略配 1e-9 容差)
- [x] 標準矩陣 `standard_matrix`(Theorem 2.9:對任意映射做標準基底取樣,T(eⱼ) 直放為 A 的第 j 行;codomain 維度從輸出導出,非線性映射取樣出的矩陣重現不了原函數)
- [x] 幾何轉換的標準矩陣:x 軸反射(example test 示範「幾何規則 → `standard_matrix` → 矩陣數值」的工作流 —— 寫規則,讓構造器去發現 [[1, 0], [0, −1]])
- [x] Theorem 2.9 laws:標準矩陣**存在且唯一**(round-trip 重建 == 誘導矩陣(整數精確)、T(v) = Av 對任意 v(浮點 1e-9);維度也隨機 —— `prop_flat_map` 先抽形狀再抽內容,涵蓋 ℝⁿ → ℝᵐ 各種組合)
- [x] 單位 / 零轉換的標準矩陣對帳(identity_matrix / zero_matrix 不另刻 —— 就是第二單元的 `Matrix::identity` / `Matrix::new`;構造器對「行為」取樣重新發現同一個矩陣,零映射的 m 與 n 解耦)
- [x] 標準矩陣取樣互動圖解:選幾何規則(旋轉 / 反射 / 剪切⋯)看 T(e₁)、T(e₂) 直放成 A 的行,可在 [web 視覺化](#視覺化)操作(矩陣由 core 的 `standard_matrix` 當場取樣,「規則路徑 vs 矩陣路徑」兩路對帳)
- [x] 守恆律互動圖解:拖動 u、v 看 shear / 投影下的影像,T(u+v) 與 T(u)+T(v) 永遠重合,可在 [web 視覺化](#視覺化)操作(✓ 由 core 的 `verify_linearity` 經 WASM 當場驗證)

### 6. 進階主題
- [x] 線性變換與幾何意義(2D):矩陣作為 2D 變換 + 線性相依,可在 [web 視覺化](#視覺化)互動操作(透過 WASM 呼叫 core 的 `multiply_vector` / `is_parallel`)
- [ ] LU 分解、特徵值 / 特徵向量 eigenvalue / eigenvector

> 每個主題對應一支 `src/*.rs` 與其 inline `#[cfg(test)]` 測試模組。

## 開始使用

```bash
# 取得程式碼
git clone git@github.com:blackhorseya/linear-algebra-101-rust.git
cd linear-algebra-101-rust

# 跑測試(學習過程主要透過測試驗證)
cargo test

# 看測試覆蓋率(需先安裝 cargo-llvm-cov)
cargo install cargo-llvm-cov
cargo llvm-cov
```

> 需要 Rust 1.85 以上(2024 edition,見 `Cargo.toml`)。
>
> 常用指令由 [Taskfile](https://taskfile.dev) 包裝:`task test`、`task check`(fmt + lint + test 的 pre-commit gate)、`task cover` 等,`task` 列出全部。

## 視覺化

`web/` 是一個 React + Vite + TanStack 前端,把 core 的運算透過 **WASM** 接到 Canvas,做「矩陣作為 2D 線性變換」、「線性相依 / 平行」、「矩陣乘法 row × col 展開」(任意尺寸,點 C 的任一格攤開 dot product,維度相容性由 core 的 `can_multiply` 判定)、「高斯消去逐步播放」、「可逆矩陣 / 基本矩陣」(逐步左乘 Eₖ 累積 P = A⁻¹,配 IMT 等價條件面板)、「線性轉換守恆律」與「標準矩陣取樣」(幾何規則經 core 的 `standard_matrix` 當場取樣出矩陣)的互動視覺化。**計算只在 Rust 一份** — JS 只負責繪圖與互動,每個變換後的點都是 core 算的。

WASM binding 鎖在 `#[cfg(feature = "wasm")]`(`src/wasm.rs`)後面:沒開 `wasm` feature 時等於不存在,`cargo test` / `task check` 完全不受影響。

```bash
# 1. 建 WASM 套件 → web/src/lib/wasm(需 wasm-pack 0.15+ 與 wasm32 target)
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
task wasm:build

# 2. 跑前端(web/ 一律用 pnpm)
cd web && pnpm install && pnpm dev
```

## 專案結構

純 library crate(無 `main.rs` / bin)。一個概念一個模組:

```
.
├── Cargo.toml          # wasm 為 optional feature,預設不啟用
├── Taskfile.yml        # cargo 指令包裝(task test / check / wasm:build …)
├── src/
│   ├── lib.rs          # crate root,re-export 各模組公開 API
│   ├── error.rs        # 共用錯誤型別 LinAlgError
│   ├── vector.rs       # Vector:加法、純量乘法、線性組合、標準基底、平行
│   ├── matrix.rs       # Matrix:加法 / 純量乘、轉置、identity、A·v、乘法 / 冪、EROs、echelon 判定
│   ├── diagonal.rs     # DiagonalMatrix:對角陣 newtype,O(n) 乘法
│   ├── span.rs         # span 與線性組合見證
│   ├── independence.rs # 線性獨立 / 冗餘 / 可移除行
│   ├── basis.rs        # basis 判定
│   ├── coordinates.rs  # 基底下的座標表示與還原
│   ├── system.rs       # 線性方程組 Ax=b 與解的分類
│   ├── elimination.rs  # Gaussian elimination、rank、nullity
│   ├── inverse.rs      # 可逆矩陣:基本矩陣 elementary_*、可逆判定 is_invertible、反矩陣 inverse
│   ├── predicate_set.rs# 以述詞表示的集合 PredicateSet
│   └── wasm.rs         # #[cfg(feature = "wasm")] WASM binding(2D 變換視覺化)
└── web/                # React + Vite 前端(透過 WASM 呼叫 core)
```

crate 名稱為 `linear_algebra_101`,測試以 inline `#[cfg(test)] mod tests` 與實作同檔(white-box,可存取 private 欄位)。

## License

[Apache License 2.0](./LICENSE)
