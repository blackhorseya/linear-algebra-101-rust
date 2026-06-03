/** 把浮點數收成最多 4 位、去掉尾隨 0(-0 也歸 0)。 */
export function fmt(n: number): string {
  return Number(n.toFixed(4)).toString()
}
