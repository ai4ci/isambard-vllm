import { describe, it, expect } from "bun:test";
import { semverLt, semverGte, semverSort } from "../src/semver.ts";

describe("semverLt", () => {
  it("returns false when versions are equal", () => {
    expect(semverLt("0.9.1", "0.9.1")).toBe(false);
  });

  it("returns true when major is less", () => {
    expect(semverLt("0.9.1", "1.0.0")).toBe(true);
  });

  it("returns false when major is greater", () => {
    expect(semverLt("1.0.0", "0.9.1")).toBe(false);
  });

  it("returns true when minor is less", () => {
    expect(semverLt("0.8.0", "0.9.0")).toBe(true);
  });

  it("returns false when minor is greater", () => {
    expect(semverLt("0.10.0", "0.9.0")).toBe(false);
  });

  it("returns true when patch is less", () => {
    expect(semverLt("0.9.0", "0.9.1")).toBe(true);
  });

  it("returns false when patch is greater", () => {
    expect(semverLt("0.9.1", "0.9.0")).toBe(false);
  });

  it("handles two-part version strings", () => {
    expect(semverLt("0.9", "0.10")).toBe(true);
  });
});

describe("semverGte", () => {
  it("returns true when versions are equal", () => {
    expect(semverGte("0.19.1", "0.19.1")).toBe(true);
  });

  it("returns true when version is greater", () => {
    expect(semverGte("0.20.0", "0.19.1")).toBe(true);
  });

  it("returns false when version is less", () => {
    expect(semverGte("0.18.0", "0.19.1")).toBe(false);
  });
});

describe("semverSort", () => {
  it("sorts versions in descending order (highest first)", () => {
    expect(semverSort(["0.18.0", "0.20.0", "0.19.1"])).toEqual(["0.20.0", "0.19.1", "0.18.0"]);
  });

  it("handles a single version", () => {
    expect(semverSort(["0.19.1"])).toEqual(["0.19.1"]);
  });

  it("handles an empty array", () => {
    expect(semverSort([])).toEqual([]);
  });
});


describe("semverLt", () => {
  it("returns false when versions are equal", () => {
    expect(semverLt("0.9.1", "0.9.1")).toBe(false);
  });

  it("returns true when major is less", () => {
    expect(semverLt("0.9.1", "1.0.0")).toBe(true);
  });

  it("returns false when major is greater", () => {
    expect(semverLt("1.0.0", "0.9.1")).toBe(false);
  });

  it("returns true when minor is less", () => {
    expect(semverLt("0.8.0", "0.9.0")).toBe(true);
  });

  it("returns false when minor is greater", () => {
    expect(semverLt("0.10.0", "0.9.0")).toBe(false);
  });

  it("returns true when patch is less", () => {
    expect(semverLt("0.9.0", "0.9.1")).toBe(true);
  });

  it("returns false when patch is greater", () => {
    expect(semverLt("0.9.1", "0.9.0")).toBe(false);
  });

  it("handles two-part version strings", () => {
    expect(semverLt("0.9", "0.10")).toBe(true);
  });
});
