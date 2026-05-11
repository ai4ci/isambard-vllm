/** Returns true if semver string `a` is strictly less than `b`. */
export function semverLt(a: string, b: string): boolean {
  const parse = (v: string) => v.split(".").map((n) => parseInt(n, 10) || 0);
  const [a1, a2, a3] = parse(a);
  const [b1, b2, b3] = parse(b);
  if (a1 !== b1) return a1 < b1;
  if (a2 !== b2) return a2 < b2;
  return a3 < b3;
}

/** Returns true if semver string `a` is greater than or equal to `b`. */
export function semverGte(a: string, b: string): boolean {
  return !semverLt(a, b);
}

/** Returns a new array sorted descending (highest version first). */
export function semverSort(versions: string[]): string[] {
  return [...versions].sort((a, b) => semverLt(a, b) ? 1 : semverLt(b, a) ? -1 : 0);
}
