//! 座標系統(單元 7-2)—— 給前端「斜格點陣」圖解用:拖 b₁、b₂ 換基底、拖 x 移點,
//! 看同一個點在不同基底下的座標 [x]_B = (c₁, c₂),與 x = c₁b₁ + c₂b₂ 的平行四邊形分解。
//!
//! 與 core 的關係:直接委派 `coordinates`(解 [x]_B)與 `from_coordinates`(由座標重建 x)——
//! binding 只做 2D 形狀的攤平與「基底退化 → 空陣列」的邊界編碼,零演算法(沿 range 章
//! 「只有積木接線」的精神)。epsilon 寫死 TRACE_EPSILON(沿 range / eliminate 慣例:
//! 拖曳座標數量級穩定)。

use super::helpers::TRACE_EPSILON;
use crate::{Vector, coordinates, from_coordinates};
use wasm_bindgen::prelude::*;

/// x 在有序基底 B = {b₁, b₂} 下的座標 [x]_B = (c₁, c₂)(core 的 `coordinates`):
/// 回 `[c₁, c₂]`;當 b₁ ∥ b₂(退化、不是 ℝ² 的基底)時回 `[]`(邊界編碼:空 = 座標未定義)。
/// 前端據此切換「畫斜格 / 顯示退化警告」,綠 / 紅都由 core 判,不在 JS 寫死。
#[wasm_bindgen]
pub fn coordinates_2d(b1x: f64, b1y: f64, b2x: f64, b2y: f64, px: f64, py: f64) -> Vec<f64> {
    let basis = [
        Vector::from_vec(vec![b1x, b1y]),
        Vector::from_vec(vec![b2x, b2y]),
    ];
    let x = Vector::from_vec(vec![px, py]);
    coordinates(TRACE_EPSILON, &x, &basis)
        .map(|c| c.entries().to_vec())
        .unwrap_or_default()
}

/// 由座標重建向量:x = c₁·b₁ + c₂·b₂(core 的 `from_coordinates`),回 `[x, y]`。
/// 前端用它把「core 重建的落點」畫成圓環套住拖曳中的 x —— 雙射兩路會合的見證。
#[wasm_bindgen]
pub fn from_coordinates_2d(b1x: f64, b1y: f64, b2x: f64, b2y: f64, c1: f64, c2: f64) -> Vec<f64> {
    let basis = [
        Vector::from_vec(vec![b1x, b1y]),
        Vector::from_vec(vec![b2x, b2y]),
    ];
    from_coordinates(&Vector::from_vec(vec![c1, c2]), &basis)
        .expect("座標數 = 基底向量數 = 2,linear_combination 不該失敗")
        .entries()
        .to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 傾斜基底 b₁=(2,1)、b₂=(−1,1),x=(3,3):解 2c₁−c₂=3、c₁+c₂=3 ⇒ (2,1)。
    #[test]
    fn coordinates_2d_reads_weights_in_tilted_basis() {
        let c = coordinates_2d(2.0, 1.0, -1.0, 1.0, 3.0, 3.0);
        assert_eq!(c.len(), 2);
        assert!(
            (c[0] - 2.0).abs() < 1e-9 && (c[1] - 1.0).abs() < 1e-9,
            "c={c:?}"
        );
    }

    /// 標準基底:座標映射是 identity —— [x]_E = x。
    #[test]
    fn coordinates_2d_standard_basis_is_identity() {
        assert_eq!(coordinates_2d(1.0, 0.0, 0.0, 1.0, 3.0, 5.0), vec![3.0, 5.0]);
    }

    /// b₁ ∥ b₂(退化)→ 不是基底 → 空陣列(座標未定義的邊界編碼)。
    #[test]
    fn coordinates_2d_degenerate_basis_is_empty() {
        assert!(coordinates_2d(1.0, 1.0, 2.0, 2.0, 3.0, 3.0).is_empty());
    }

    /// round-trip:from_coordinates_2d(coordinates_2d(x)) 重建回 x(雙射 —— 與 core 的
    /// `coordinates_round_trip_is_identity` law 同一件事,在邊界層再驗一次 marshalling)。
    #[test]
    fn from_coordinates_2d_round_trips() {
        let (b1x, b1y, b2x, b2y) = (2.0, 1.0, -1.0, 1.0);
        let c = coordinates_2d(b1x, b1y, b2x, b2y, 3.0, 3.0);
        let back = from_coordinates_2d(b1x, b1y, b2x, b2y, c[0], c[1]);
        assert!(
            (back[0] - 3.0).abs() < 1e-9 && (back[1] - 3.0).abs() < 1e-9,
            "back={back:?}"
        );
    }
}
