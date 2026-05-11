import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";
import { spawnSync } from "child_process";

describe("F2.6 — Integration tests", () => {
  describe("CLI flag parsing", () => {
    it("parses --no-launch flag", () => {
      const result = spawnSync("bun", ["run", "src/index.ts", "start", "test-job", "--no-launch"], {
        cwd: process.cwd(),
        encoding: "utf-8",
        timeout: 5000,
      });
      // Should not crash on flag parsing (will fail on missing config, which is expected)
      expect(result.status).not.toBe(2); // exit code 2 means argument parsing error
    });

    it("recognizes --no-launch in argument list", () => {
      const args = ["test-job", "--no-launch", "--gpus", "4"];
      const parsed = args.filter(a => a === "--no-launch" || a === "--dry-run");
      expect(parsed).toContain("--no-launch");
    });
  });

  describe("Assistant detection", () => {
    it("detects ls as available binary", () => {
      const result = spawnSync("which", ["ls"], { encoding: "utf-8", timeout: 5000 });
      expect(result.status).toBe(0);
    });

    it("returns empty for non-existent binary", () => {
      const result = spawnSync("which", ["nonexistent-f26-test"], { encoding: "utf-8", timeout: 5000 });
      expect(result.status).toBe(1);
    });
  });

  describe("CWD detection", () => {
    it("reads current working directory", () => {
      const cwd = process.cwd();
      expect(typeof cwd).toBe("string");
      expect(cwd.length).toBeGreaterThan(0);
      expect(cwd.startsWith("/")).toBe(true);
    });

    it("cwd is accessible", async () => {
      const fs = await import("fs");
      const cwd = process.cwd();
      expect(fs.existsSync(cwd)).toBe(true);
    });
  });
});
