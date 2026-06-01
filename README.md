# linear-algebra-101-rust

用 **Rust** 從零實作 linear algebra,邊寫邊學。

這個 repo 不依賴 [nalgebra](https://nalgebra.org/) 或 [ndarray](https://docs.rs/ndarray) 之類的數值函式庫,而是親手把 vector、matrix 以及它們的運算刻出來。目的不是做出最快的實作,而是透過程式碼建立對線性代數的直覺 — 當你能用 Rust 的型別與函式表達一個數學概念,代表你真的懂了它。

> 這是 [linear-algebra-101](https://github.com/blackhorseya/linear-algebra-101)(Go 版)的 Rust 改寫。

## 為什麼用 Rust 學線性代數?

- **型別逼你想清楚結構** — `Vector` 與 `Matrix` 是不同型別,維度不合的運算應該在邊界就被擋下來,而不是算到一半才爆。
- **函式即定義** — `dot(a, b)` 的實作就是 dot product 的數學定義,寫出來就懂了。
- **錯誤是值,不是例外** — 維度不合回傳 `Result<_, _>` 而非 panic,呼叫端被型別系統逼著面對「這個運算可能無效」這件事 — 這正是線性代數裡維度條件的精神。
- **測試即證明** — 每個運算都能用已知結果(例如 identity matrix 乘任何 matrix 等於自己)寫成 test,把數學性質變成可驗證的程式;搭配 [`proptest`](https://docs.rs/proptest) 還能把定律變成隨機驗證。

## 學習路徑

由淺入深,後面的概念建立在前面的之上:

### 1. Vector(向量)
- [ ] 加法 / 減法 `add`, `sub`
- [ ] 純量乘法 `scale`
- [ ] 內積 dot product `dot`
- [ ] 長度 / 範數 `norm`(L2)
- [ ] 單位向量 `normalize`
- [ ] 向量夾角 `angle`(由 dot product 推導)

### 2. Matrix(矩陣)
- [ ] 表示法與建構子(維度驗證)
- [ ] 加法 / 純量乘法
- [ ] 轉置 `transpose`
- [ ] 矩陣乘法 `mul`(row × column 的 dot product)
- [ ] 單位矩陣 `identity`

### 3. 線性方程組與分解
- [ ] Gaussian elimination(高斯消去法)
- [ ] 行列式 `determinant`
- [ ] 反矩陣 `inverse`
- [ ] 秩 `rank`

### 4. 進階主題
- [ ] LU 分解
- [ ] 特徵值 / 特徵向量 eigenvalue / eigenvector
- [ ] 線性變換與幾何意義(旋轉、投影、縮放)

> 進度會隨著實作逐步勾選。每個主題對應一支 `src/*.rs` 與其 `#[cfg(test)]` 測試模組。

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

## 專案結構

純 library crate 風格,概念對應檔案:

```
.
├── Cargo.toml
└── src/
    ├── lib.rs        # crate root,re-export 各模組公開 API
    ├── vector.rs     # Vector 型別與運算
    ├── matrix.rs     # Matrix 型別與運算
    └── ...           # 隨學習路徑擴充
```

crate 名稱為 `linear_algebra_101`,測試以 inline `#[cfg(test)] mod tests` 形式與實作同檔。

## License

[Apache License 2.0](./LICENSE)
