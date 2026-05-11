import { describe, it, expect } from "bun:test";
import { selectBestVersion } from "../src/commands/start.ts";

describe("selectBestVersion", () => {
  it("returns the highest installed version >= min", () => {
    expect(selectBestVersion(["0.18.0", "0.19.1", "0.20.0"], "0.19.1")).toBe("0.20.0");
  });

  it("returns exact match when only that version is installed", () => {
    expect(selectBestVersion(["0.19.1"], "0.19.1")).toBe("0.19.1");
  });

  it("returns null when no installed version satisfies the minimum", () => {
    expect(selectBestVersion(["0.18.0", "0.18.5"], "0.19.1")).toBeNull();
  });

  it("returns null for empty installed list", () => {
    expect(selectBestVersion([], "0.19.1")).toBeNull();
  });

  it("returns the sole installed version when it exceeds minimum", () => {
    expect(selectBestVersion(["0.20.0"], "0.19.1")).toBe("0.20.0");
  });
});
