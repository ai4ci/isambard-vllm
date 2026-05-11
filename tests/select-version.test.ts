import { describe, it, expect } from "bun:test";
import { selectBestVersion } from "../src/commands/start.ts";
import { semverSort } from "../src/semver.ts";

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

describe("version discovery without minVllmVersion", () => {
  it("semverSort returns highest version first (used when no minVllmVersion set)", () => {
    // Regression: venv path must use discovered vllmVersion, not config.vllmVersion (which is undefined)
    const installed = ["0.18.0", "0.19.1", "0.20.0"];
    const selected = semverSort(installed)[0];
    expect(selected).toBe("0.20.0");
    expect(selected).not.toBeUndefined();
  });

  it("single installed version is selected correctly", () => {
    const installed = ["0.19.1"];
    const selected = semverSort(installed)[0];
    expect(selected).toBe("0.19.1");
    expect(selected).not.toBeUndefined();
  });
});
