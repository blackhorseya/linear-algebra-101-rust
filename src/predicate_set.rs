//! `PredicateSet` —— 用「成員規則」(membership rule)而非列舉元素定義的集合。
//!
//! 對應原始 Go 專案 commit `e01285e`
//! (`feat: add PredicateSet for sets defined by membership rules`)。
//!
//! 一個集合有兩種看法:
//! - **外延(extensional)**:把元素一個個列出來 —— `{2, 4, 6, …}`。需要能逐一
//!   比較元素(`==`),而且只裝得下有限多個。
//! - **內涵(intensional)**:給一條「誰算成員」的規則 —— `{ x | x 是偶數 }`。
//!   `PredicateSet` 就是這條規則本身(集合的**特徵函數**):`contains(x)` 為真
//!   ⟺ x 是成員。集合建構式 `{ y | P(y) }` 不過就是給述詞 P 取個名字當集合讀:
//!   `x ∈ {y | P(y)} ⟺ P(x)`。
//!
//! 為什麼這個專案需要它?因為「向量的集合」(span、解集合)在 Rust **沒辦法**用
//! `HashSet<Vector>` 表示:`Vector` 內含 `Vec<f64>`,而 `f64` 因為 NaN 既不是
//! `Eq` 也不是 `Hash`,`Vector` 因此進不了 `HashSet`;就算硬塞 `HashSet<*const _>`
//! 之類,比的也是指標身分而非向量的值 —— 錯的相等概念。再說 span 與解集合往往是
//! **無限**的,根本列不完。改用述詞兩個問題一起解決:成員資格用「算」的、不用
//! 「比」的,而且一條規則就描述了無限集合。
//!
//! 代價:你能問「x 在不在裡面?」(`contains`),但**不能**列舉它、數它的大小,
//! 或靠逐一列舉判定子集合 / 相等 —— 可能有無限多個要檢查。那些問題之後會變成數學
//! 論證(例如算 rank),而不是迴圈。

use std::rc::Rc;

/// 由成員規則定義的集合:`PredicateSet::new(p)` 代表 `{ x | p(x) }`。
///
/// 為何內部是 `Rc<dyn Fn(&T) -> bool>` 而不是 Go 那種裸函式?
/// - `dyn Fn`:union / intersection 等運算要把既有集合「包」進一個新的 closure,
///   每個 closure capture 的環境不同、型別也各異,只能用 trait object 統一持有。
/// - `Rc`:組合運算(像 De Morgan)會把**同一個**集合用在多處,需要共享所有權。
///   `Rc::clone` 只複製指標、不複製規則,讓 `PredicateSet` 能像數學物件一樣被
///   重複使用而不被 move 走 —— 也因此這些方法收 `&self` 而非吃掉 `self`。
/// - `&T`(借用而非取值):讓 `contains` 不必搬走或複製像 `Vector` 這種帶 heap
///   資料的元素。
pub struct PredicateSet<T> {
    predicate: Rc<dyn Fn(&T) -> bool>,
}

impl<T> Clone for PredicateSet<T> {
    fn clone(&self) -> Self {
        // 只複製 Rc 指標(refcount +1),不複製底層規則。`#[derive(Clone)]` 在這裡
        // 行不通:它會要求 `T: Clone`,但被複製的根本不是 T,而是那條規則的指標。
        PredicateSet {
            predicate: Rc::clone(&self.predicate),
        }
    }
}

impl<T: 'static> PredicateSet<T> {
    /// 從一條成員規則建立集合:`new(p)` 即 `{ x | p(x) }`。
    pub fn new(predicate: impl Fn(&T) -> bool + 'static) -> PredicateSet<T> {
        PredicateSet {
            predicate: Rc::new(predicate),
        }
    }

    /// 空集合 ∅ = `{ x | false }`:沒有任何元素,`contains` 永遠回傳 `false`。
    /// 規則是常數 false —— 集合代數的最小元(⊥)。不一致線性系統的解集合就等於它。
    pub fn empty() -> PredicateSet<T> {
        PredicateSet::new(|_| false)
    }

    /// 全集 U = `{ x | true }`:型別 T 的每個值都是成員,`contains` 永遠 `true`。
    /// 規則是常數 true —— 代數的最大元(⊤),也是 `complement` 隱含的「宇集」。
    pub fn universal() -> PredicateSet<T> {
        PredicateSet::new(|_| true)
    }

    /// x ∈ self 嗎?就是套用成員規則 —— 對 `{ x | P(x) }` 而言即 `P(x)`。
    pub fn contains(&self, x: &T) -> bool {
        (self.predicate)(x)
    }

    /// 聯集 self ∪ other = `{ x | x ∈ self 或 x ∈ other }` —— 把布林 OR 抬升到集合。
    ///
    /// 這支當作「在 Rust 裡組合 `PredicateSet`」的範例:先 `Rc::clone` 兩邊的規則
    /// (共享、不複製),再 `move` 進一個新的 closure,讓新集合擁有那兩份指標。回傳
    /// 的是新規則、不列舉任何元素 —— 兩邊都可能是無限的。
    pub fn union(&self, other: &PredicateSet<T>) -> PredicateSet<T> {
        let p = Rc::clone(&self.predicate);
        let q = Rc::clone(&other.predicate);
        PredicateSet::new(move |x| p(x) || q(x))
    }

    /// 交集 self ∩ other = `{ x | x ∈ self 且 x ∈ other }` —— 布林 AND 抬升到集合。
    pub fn intersection(&self, other: &PredicateSet<T>) -> PredicateSet<T> {
        // 換你寫:照上面 `union` 的同一套路 —— 把兩邊的規則各 `Rc::clone` 一份、
        // `move` 進一個新 closure,只是把 `||` 換成 `&&`。
        let p = Rc::clone(&self.predicate);
        let q = Rc::clone(&other.predicate);
        PredicateSet::new(move |x| p(x) && q(x))
    }

    /// 補集 selfᶜ = `{ x | x ∉ self }` —— 布林 NOT 抬升到集合。
    ///
    /// 注意:**不需要**傳入宇集參數。對述詞集合而言,把規則取反就已經描述了「所有
    /// 不在 self 裡的東西」,宇集 U 是隱含的(就是整個型別 T)。
    pub fn complement(&self) -> PredicateSet<T> {
        // 換你寫:只 `Rc::clone` self 一份規則、`move` 進新 closure,回傳 `!p(x)`。
        // 想想為何這裡不需要(也用不到)宇集參數 —— 對照 doc 註解。
        let p = Rc::clone(&self.predicate);
        PredicateSet::new(move |x| !p(x))
    }

    /// 差集 self \ other = `{ x | x ∈ self 且 x ∉ other }`。
    ///
    /// 刻意**用前三個連接詞組合**而成:self \ other 就是 self ∩ otherᶜ。藉此說明
    /// union / intersection / complement 已生成整個集合代數 —— 有了它們,這支自動可用。
    pub fn difference(&self, other: &PredicateSet<T>) -> PredicateSet<T> {
        self.intersection(&other.complement())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;

    // 述詞集合怎麼測 —— 用「探點(probe)」而非列舉。
    //
    // 一個 `PredicateSet` 可能是無限的,沒辦法逐一列舉成員來比較兩個集合。改用固定
    // 一串樣本點(probes),斷言兩集合對每個探點給出相同的成員判定。這是性質檢查、
    // 不是證明:某個探點不一致就是確鑿的反例,但所有探點都一致只是強烈證據、不是
    // 鐵證。這道縫隙正是為何之後要證明兩個 span 相等得靠數學論證(算 rank),而不是
    // 迴圈 —— 在無限定義域上沒辦法用探點探出一條定理。

    /// 第一個讓 a、b 成員判定不一致的探點;全部一致回 `None`。
    fn first_disagreement<'a, T: 'static>(
        a: &PredicateSet<T>,
        b: &PredicateSet<T>,
        probes: &'a [T],
    ) -> Option<&'a T> {
        probes.iter().find(|x| a.contains(x) != b.contains(x))
    }

    /// 涵蓋負數、零、奇偶、大小 —— 足以區分下面用到的述詞(偶數、正數…)。
    const PROBES: [i32; 13] = [-101, -3, -2, -1, 0, 1, 2, 3, 7, 10, 11, 100, 101];

    /// 其他測試都建立在這兩個集合之上。
    /// `even` = `{ n | n 是偶數 }`(無限、兩種正負號);`positive` = `{ n | n > 0 }`(無限)。
    fn even_set() -> PredicateSet<i32> {
        PredicateSet::new(|n| n % 2 == 0)
    }
    fn positive_set() -> PredicateSet<i32> {
        PredicateSet::new(|n| *n > 0)
    }

    #[test]
    fn empty_and_universal_are_the_two_extremes() {
        // 釘住代數的兩個極端:∅ 拒絕一切(規則是常數 false),U 接受一切(常數 true)。
        let empty = PredicateSet::<i32>::empty();
        let universal = PredicateSet::<i32>::universal();
        for x in PROBES {
            assert!(!empty.contains(&x), "∅ 不該包含 {x}");
            assert!(universal.contains(&x), "U 應包含每個元素,卻漏了 {x}");
        }
    }

    #[test]
    fn union_is_boolean_or() {
        // x ∈ A∪B ⟺ x∈A 或 x∈B,對照直接用 || 算出的參考述詞。
        let got = even_set().union(&positive_set());
        let want = PredicateSet::new(|n: &i32| n % 2 == 0 || *n > 0);
        assert!(
            first_disagreement(&got, &want, &PROBES).is_none(),
            "union 與 (even OR positive) 不一致"
        );
    }

    #[test]
    fn intersection_is_boolean_and() {
        let got = even_set().intersection(&positive_set());
        let want = PredicateSet::new(|n: &i32| n % 2 == 0 && *n > 0);
        assert!(
            first_disagreement(&got, &want, &PROBES).is_none(),
            "intersection 與 (even AND positive) 不一致"
        );
    }

    #[test]
    fn complement_is_boolean_not() {
        // x ∈ Aᶜ ⟺ x∉A;不需宇集參數,宇集隱含為整個型別 T。
        let got = even_set().complement();
        let want = PredicateSet::new(|n: &i32| n % 2 != 0);
        assert!(
            first_disagreement(&got, &want, &PROBES).is_none(),
            "complement 與 (not even) 不一致"
        );
    }

    #[test]
    fn excluded_middle_and_contradiction() {
        // 把兩條邏輯定律變成集合恆等式,在樣本上探測:
        //   A ∪ Aᶜ = U  (排中律:p ∨ ¬p = true)
        //   A ∩ Aᶜ = ∅  (矛盾律:p ∧ ¬p = false)
        let a = even_set();
        assert!(
            first_disagreement(
                &a.union(&a.complement()),
                &PredicateSet::universal(),
                &PROBES
            )
            .is_none(),
            "排中律失敗:A ∪ Aᶜ 不是全集"
        );
        assert!(
            first_disagreement(
                &a.intersection(&a.complement()),
                &PredicateSet::empty(),
                &PROBES
            )
            .is_none(),
            "矛盾律失敗:A ∩ Aᶜ 不是空集"
        );
    }

    #[test]
    fn de_morgan_laws() {
        // (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ  以及  (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ。
        // 注意 a、b 各被用了好幾次 —— 方法收 &self、靠 Rc 共享,集合不會用一次就沒。
        let a = even_set();
        let b = positive_set();

        let left1 = a.union(&b).complement();
        let right1 = a.complement().intersection(&b.complement());
        assert!(
            first_disagreement(&left1, &right1, &PROBES).is_none(),
            "De Morgan 失敗:(A∪B)ᶜ ≠ Aᶜ∩Bᶜ"
        );

        let left2 = a.intersection(&b).complement();
        let right2 = a.complement().union(&b.complement());
        assert!(
            first_disagreement(&left2, &right2, &PROBES).is_none(),
            "De Morgan 失敗:(A∩B)ᶜ ≠ Aᶜ∪Bᶜ"
        );
    }

    #[test]
    fn difference_is_intersection_with_complement() {
        // positive \ even = 正奇數:接受 1,3,7,11,101,拒絕偶數與非正數。
        let got = positive_set().difference(&even_set());
        let want = PredicateSet::new(|n: &i32| *n > 0 && n % 2 != 0);
        assert!(
            first_disagreement(&got, &want, &PROBES).is_none(),
            "difference 與 (positive AND odd) 不一致"
        );
    }

    #[test]
    fn holds_genuinely_infinite_set() {
        // 述詞表示法的回報:全體偶數是真正無限的(沒有任何有限容器裝得下),
        // 但成員判定仍是一行、瞬間可決的規則。
        let even = even_set();
        for x in [-1000, -2, 0, 2, 4, 1000, 1_000_000] {
            assert!(even.contains(&x), "even 應包含 {x}");
        }
        for x in [-999, -1, 1, 3, 999_999] {
            assert!(!even.contains(&x), "even 不應包含 {x}");
        }
    }

    #[test]
    fn holds_vectors_by_value_not_identity() {
        // `PredicateSet<T>` 吃得下 `HashSet<T>` 吃不下的 T:`Vector` 帶 `Vec<f64>` 欄位,
        // 進不了 `HashSet`;就算用指標當 key,比的也只是指標身分。這裡成員資格從向量的
        // **值**算出來(is_zero),才是對的相等概念 —— 也是日後把 span、解集合表示成
        // 向量集合的基石。
        let zeros = PredicateSet::new(|v: &Vector| v.is_zero());

        assert!(zeros.contains(&Vector::new(3)), "剛建好的零向量應是成員");
        assert!(
            !zeros.contains(&Vector::from_vec(vec![0.0, 1.0, 0.0])),
            "非零向量不該是成員"
        );
        // 另一個「不同的」全零向量仍必須是成員:這正是指標身分比較會弄錯、
        // 而述詞成員判定能答對的情況。
        assert!(
            zeros.contains(&Vector::from_vec(vec![0.0, 0.0, 0.0])),
            "成員資格看值:另一個全零向量也得算在內"
        );
    }
}
