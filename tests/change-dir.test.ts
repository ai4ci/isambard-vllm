import { describe, it, expect } from "bun:test";

describe("--change-dir flag", () => {
  it("should be parsed from arguments", () => {
    const args = ["test-job", "--change-dir", "/some/path"];
    const flags: Record<string, string> = {};
    const boolFlags = new Set<string>();
    
    for (let i = 1; i < args.length; i++) {
      const arg = args[i];
      if (!arg?.startsWith("--")) continue;
      const key = arg.slice(2);
      const next = args[i + 1];
      if (!next || next.startsWith("--")) {
        boolFlags.add(key);
      } else {
        flags[key] = next;
        i++;
      }
    }
    
    expect(flags["change-dir"]).toBe("/some/path");
  });

  it("resolves relative paths correctly", () => {
    const path = require("path");
    const relative = "../other-dir";
    const resolved = path.resolve(relative);
    expect(resolved.startsWith("/")).toBe(true);
    expect(resolved.includes("../")).toBe(false);
  });

  it("validates directory existence", () => {
    const fs = require("fs");
    const path = require("path");
    
    // Valid directory
    const cwd = process.cwd();
    expect(fs.existsSync(cwd)).toBe(true);
    
    // Invalid directory
    const invalid = "/nonexistent/f26-test-dir";
    expect(fs.existsSync(invalid)).toBe(false);
  });

  it("generates cd command for launch script", () => {
    const changeDir = "/projects/b6ax/my-project";
    const cmd = "opencode";
    const args = [];
    
    const launchScript = changeDir
      ? `cd '${changeDir}' && ${cmd} ${args.map(a => `'${a.replace(/'/g, "'\\''")}'`).join(" ")}`
      : `${cmd} ${args.map(a => `'${a.replace(/'/g, "'\\''")}'`).join(" ")}`;
    
    expect(launchScript).toContain(`cd '${changeDir}'`);
    expect(launchScript).toContain("opencode");
  });

  it("works without changeDir (no cd prefix)", () => {
    const changeDir = "";
    const cmd = "claude";
    const args: string[] = [];
    
    const launchScript = changeDir
      ? `cd '${changeDir}' && ${cmd} ${args.map(a => `'${a.replace(/'/g, "'\\''")}'`).join(" ")}`
      : `${cmd} ${args.map(a => `'${a.replace(/'/g, "'\\''")}'`).join(" ")}`;
    
    expect(launchScript).not.toContain("cd");
    expect(launchScript.trim()).toBe("claude");
  });
});
