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
- [x] 子空間三公理 `contains_zero` / `closed_at`(掛在 `PredicateSet<Vector>`:0 ∈ W、加法 / 純量乘法封閉的**單點見證**檢查 —— 蘊涵語意,前提不成立空虛真;隨機抽樣是全稱命題的驗證,留給 proptest laws,抽樣只能反證不能證明(第一象限對 c < 0 不封閉,laws 整族全滅))
- [x] 零空間成員判定 `null_space_contains`(Theorem 4.2:Null A = { v : Av = 0 } 是 ℝⁿ 的子空間 —— 成員判定只要「代入驗算」O(mn),與 `range_contains` 要「解方程」O(n³) 不對稱;laws:核植入法依建構產 Av = 0 的成員、Null A 包成 `PredicateSet` 過三公理機器,題目 1 與 2 合龍)
- [x] 列空間生成集 `row_space_generators`(Row A = Col Aᵀ 是換句話說不是定理:Aᵀ 的行就是 A 的列 —— 列空間零成本繼承行空間機器;Col A 生成集與 Range = Col A 則**零新碼**:5-3 的 `range_generating_set` 與 law `image_is_always_reachable` 已收)
- [x] 縮減定理 `reduce_to_basis`(Theorem 4.3:生成集 → 丟冗餘 → 基底,保留 pivot 行的**原始**向量(非 RREF 行,同 `range_basis` 陷阱);引擎 Span→`pivot_columns`,與既有 `range_basis` 是同一操作的一般版 —— 行空間基底就是「對 A 的行做縮減」的特例;維度權威沿用既有 `Span::dimension()` = rank,不另立型別;laws:Theorem 4.3 縮減後獨立 + span 不變 + 為子集、size = rank(Theorem 4.5 維度良定,且加冗餘生成元素不改維度=任兩基底等勢)、不丟 ⟺ 本來就獨立(題 1 基底⟺獨立)、Col A 基底三路對帳 `reduce_to_basis(行)` == `range_basis` ⊆ 原始行(題 2))

- [x] 零空間互動圖解:拖輸入 v 看它的像 Av 被壓向哪裡,落到核線上時 Av 塌進原點、v 變綠,可在 [web 視覺化](#視覺化)操作(`/range` 的**對偶** —— 輸入端壓扁 vs 輸出端覆蓋;v 是否在核裡由 core 的 `null_space_contains` 即時判定,nullity 與 rank 各自由 core 算、相加 = 2 當場驗證 rank-nullity 定理;核線方向由 transformPoint 掃描出「被壓最扁」的方向)

- [x] 擴展定理 `extend_to_basis`(Theorem 4.4:獨立集 li 補成基底 —— li 放最前接上 full_basis 後委派 `reduce_to_basis`,pivot 最左優先貪婪選 ⟹ li 獨立故全成 pivot 全保留、full_basis 只補新方向;與 `reduce_to_basis` 同引擎「往內縮 vs 往外補」;laws:Theorem 4.4(li 前綴保留 + 結果是 ℝⁿ 基底)、Theorem 4.6 鴿籠(>dim 必相依,題 2)、Theorem 4.7 捷徑(|S|=dim 時 LI⟺generating⟺rank,題 3)、子空間包含(V⊆W ⟹ dimV≤dimW、等維⟹V=W,題 4))

### 4. 線性方程組與分解
- [x] 線性方程組 Ax=b:`System`、`solve`、一致性 `is_consistent`、解的分類 `Solution` / `RowKind`
- [x] Gaussian elimination:`row_echelon_form` / `reduced_row_echelon_form`、秩 `rank`、零化度 `nullity`、pivot / free 行
- [x] 基本矩陣 elementary matrices(`elementary_swap` / `elementary_scale` / `elementary_add_scaled`:Iₙ + 一次 ERO;左乘 E = 施作該列運算)
- [x] 可逆判定 `is_invertible`(可逆矩陣定理 IMT:RREF = Iₙ ⟺ rank = n ⟺ 行向量獨立 ⟺ 唯一解⋯⋯等價條件以 laws 互驗)
- [x] 反矩陣 `inverse`(Gauss-Jordan 累乘基本矩陣 P = Eₖ⋯E₁,Theorem 2.3 直接寫成演算法;Theorem 2.2 代數性質以 laws 驗證)
- [x] 子矩陣 `submatrix`(A₍ᵢⱼ₎:刪第 i 列第 j 行 —— 餘因子展開的原料;1×1 → 0×0 邊界全定義、錯誤面只剩索引越界;laws:形狀各減一、內容索引映射對帳、轉置對偶 (Aᵀ)₍ⱼᵢ₎ = (A₍ᵢⱼ₎)ᵀ)
- [x] 行列式(定義版)`determinant_recursive`(遞迴餘因子展開 det A = Σⱼ (−1)^(1+j) a₁ⱼ det A₁ⱼ,O(n!) 教學版;base 0×0 = 1 空積讓 1×1 自然落入展開、非方陣 NotSquare;laws:det Iₙ = 1 + **ERO 效果三部曲**(swap 變號 / scale 倍乘 / add 不變)—— 練 4 Gaussian 版的理論根據先存證)
- [x] 三角矩陣快速路徑 `is_upper_triangular` / `is_lower_triangular` / `determinant_triangular`(Theorem 3.2:三角方陣 det = 對角線乘積 O(n);兩述詞獨立實作、非方陣恆 false,fast path 不適用回 None 而非錯誤;laws:上下三角各與定義版對帳、轉置對偶 lower(A) ⟺ upper(Aᵀ))
- [x] 行列式(實用版,得正名)`determinant`(Theorem 3.3:Gaussian forward 消去 O(n³),只准 swap(記翻號 (−1)^r)與 add(det 不變)、絕不 scale;奇異時 early return 精確 0.0、自換不翻號;laws:與定義版及三角 fast path 對帳 —— 三路對帳網閉合,12×12 釘 O(n³) vs O(n!) 的結構性差距)
- [x] 行列式三大代數性質 laws(Theorem 3.4:(a) 可逆 ⟺ det ≠ 0(與可逆矩陣章會師)、(b) det(AB) = det A · det B(乘法積性,量級放大需**相對容差**)、(c) det Aᵀ = det A(行列對稱)—— 本章 laws 收官,把 det 與 IMT、乘法、轉置縫起來)
- [x] 基本矩陣的行列式 laws(Theorem 3.3(d):det E 規律 swap → −1、scale(c) → c、add → 1(E 從 I 經一次 ERO 套三部曲)+ det(EA) = det E · det A —— elimination 的 ERO、inverse 的基本矩陣、determinant 三章會師;單元 6-1 其餘題目皆 5-5 已收,零新 API 純 reuse)
- [x] 行列式互動圖解:拖 î′、ĵ′ 看單位正方形的像 —— |det| = 面積縮放、det < 0 翻面、det = 0 塌縮成線;3×3 / 4×4 推廣為(超)體積,可在 [web 視覺化](#視覺化)操作(det 路(Gaussian 消去)與 rank 路(`is_invertible`)兩條獨立計算當場對帳 Theorem 3.4(a))

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
- [x] 值域的生成集合 `range_generating_set`(Range(T) = Col(A):每個輸出 T(x) = x₁a₁ + ⋯ + xₙaₙ 都是行向量的線性組合,行向量整組就是值域的生成集合,各自住在 codomain ℝᵐ)
- [x] 值域成員判定 `range_contains`(w ∈ Range(T) ⟺ Ax = w 相容 —— 成員判定就是一致性判定,委派 `System::is_consistent`;w 不在 codomain 回 false)
- [x] 映成判定 `is_onto`(Theorem 2.10:onto ⟺ rank(A) = m —— 全稱命題壓縮成一個數字比較;laws:onto ⟺ 全標準基底可達、高瘦必不映成、方陣 onto ⟺ 可逆(IMT 第十條等價條件))
- [x] 不可達向量 `unreachable_vector`(不映成的具體見證:掃描 e₁…e_m 回第一支不在值域的 —— proper subspace 裝不下整組 spanning set,故見證必存在;映成時掃描空手而回,Option 與映成性嚴絲合縫)
- [x] 值域的基底 `range_basis`(行對應定理:列運算保持行**之間**的線性關係,pivot 落在哪幾行、原矩陣的那幾支行就是獨立的 —— 索引問 RREF、內容問原始 A;laws 三條合起來即「是基底」的完整證明:獨立 + 住在值域 + 大小 = rank)
- [x] 值域覆蓋互動圖解:拖動行向量看 Range(T) = Col(A) 從平面塌成直線,拖 w 看可達性即時判定,可在 [web 視覺化](#視覺化)操作(可達性 / 基底 / 映成 / 不可達見證全由 core 的 `range` 模組當場計算)
- [x] 標準矩陣取樣互動圖解:選幾何規則(旋轉 / 反射 / 剪切⋯)看 T(e₁)、T(e₂) 直放成 A 的行,可在 [web 視覺化](#視覺化)操作(矩陣由 core 的 `standard_matrix` 當場取樣,「規則路徑 vs 矩陣路徑」兩路對帳)
- [x] 守恆律互動圖解:拖動 u、v 看 shear / 投影下的影像,T(u+v) 與 T(u)+T(v) 永遠重合,可在 [web 視覺化](#視覺化)操作(✓ 由 core 的 `verify_linearity` 經 WASM 當場驗證)
- [x] 一對一判定 `is_one_to_one`(Theorem 2.11:1-1 ⟺ rank(A) = n ⟺ nullity = 0 —— 與 `is_onto` 完美對偶,同一個 rank 兩端各問一次;laws:nullity 交叉驗證、寬矮必非 1-1(鴿籠)、轉置對偶(T_A 1-1 ⟺ T_{Aᵀ} onto)、方陣 1-1 ⟺ 可逆(IMT))
- [x] 合成 `compose`(T_B ∘ T_A = T_BA:合成的標準矩陣 = 乘積 B·A —— `u.compose(&t)` 讀作 U ∘ T,「合成就是乘法」從定理升格為 API;維度檢查由 `multiply` 傳播(中間空間接得上 ⟺ can_multiply),不收 epsilon(乘法精確);laws:(U∘T)(x) = U(T(x))、結合律、identity 中立 —— Transformation 在 ∘ 下構成 monoid)
- [x] 逆轉換 `inverse`(Theorem 2.13:T 可逆 ⟺ A 可逆,T⁻¹ = T_{A⁻¹} —— 委派 Gauss-Jordan 的 `Matrix::inverse`,失敗分層原樣傳播(NotSquare / NotInvertible);laws:T⁻¹(T(x)) = x 雙向、T⁻¹ ∘ T = I(compose 與 inverse 會師)、襪子鞋子 (U∘T)⁻¹ = T⁻¹∘U⁻¹、對合 (T⁻¹)⁻¹ = T、Theorem 2.12 存證(可逆 ⟺ 1-1 且 onto))
- [x] 可逆性綜合判定表 `report`(講義 2.8 Summary Table:一次回答 1-1 / onto / 可逆三問 —— `TransformationReport` 純輸出值用 pub 欄位;is_invertible 走函數視角(Theorem 2.12 的雙射定義 1-1 && onto),Theorem 2.13(⟺ A 可逆)當 law 對帳;laws:三欄與獨立述詞逐欄一致、方陣三位一體(IMT:全亮或全滅)、非方陣恆不可逆)
- [x] 合成與可逆性互動圖解:拖 x 看「先 T 再 U」兩步路徑與「一步 BA」直達會合(T_B ∘ T_A = T_BA),逆轉換模式看「變形 → 復原」,可在 [web 視覺化](#視覺化)操作(合成 / 求逆 / Summary Table 三燈全由 core 的 `composition` 模組當場計算)

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

`web/` 是一個 React + Vite + TanStack 前端,把 core 的運算透過 **WASM** 接到 Canvas,做「矩陣作為 2D 線性變換」、「線性相依 / 平行」、「矩陣乘法 row × col 展開」(任意尺寸,點 C 的任一格攤開 dot product,維度相容性由 core 的 `can_multiply` 判定)、「高斯消去逐步播放」、「可逆矩陣 / 基本矩陣」(逐步左乘 Eₖ 累積 P = A⁻¹,配 IMT 等價條件面板)、「線性轉換守恆律」、「標準矩陣取樣」(幾何規則經 core 的 `standard_matrix` 當場取樣出矩陣)、「值域與映成」(拖行向量看 Range = Col(A) 塌縮、拖 w 由 core 的 `range_contains` 即時判定可達性)、「零空間與 rank-nullity」(`/range` 的對偶:拖輸入 v 看像 Av 被壓到核線、塌進原點,nullity + rank = 2 由 core 兩次獨立計算當場對帳)、「合成與可逆性」(兩步路徑與一步 BA 會合、逆轉換的「變形 → 復原」,合成 / 求逆 / Summary Table 全由 core 的 `composition` 模組計算)與「行列式」(拖 î′ / ĵ′ 看單位正方形的像的有號面積,3×3 / 4×4 推廣為(超)體積,det 路與 rank 路由 core 獨立計算、當場對帳 Theorem 3.4(a))的互動視覺化。**計算只在 Rust 一份** — JS 只負責繪圖與互動,每個變換後的點都是 core 算的。

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
