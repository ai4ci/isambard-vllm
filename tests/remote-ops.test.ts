import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join, basename } from "path";
import { makeRemoteOps } from "../src/remote-ops.ts";
import type { Config } from "../src/config.ts";

const mockConfig: Config = {
  loginHost: "login.example.com",
  username: "testuser",
  projectDir: "/projects/myproject",
  defaultLocalPort: 11434,
  vllmVersion: "0.9.1",
};

describe("makeRemoteOps — dry-run mode", () => {
  let dryRunDir: string;

  beforeEach(() => {
    dryRunDir = mkdtempSync(join(tmpdir(), "ivllm-ops-test-"));
  });

  afterEach(() => {
    rmSync(dryRunDir, { recursive: true, force: true });
  });

  it("runRemote returns exitCode 0 without executing SSH", async () => {
    const ops = makeRemoteOps(mockConfig, true, dryRunDir);
    const result = await ops.runRemote("echo hello");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe("");
  });

  it("runRemote returns empty stdout without executing SSH", async () => {
    const ops = makeRemoteOps(mockConfig, true, dryRunDir);
    const { stdout } = await ops.runRemote("cat /etc/hostname", { silent: true });
    expect(stdout).toBe("");
  });

  it("copyFile writes source content to dryRunDir using remote basename", async () => {
    const srcPath = join(dryRunDir, "source.sh");
    writeFileSync(srcPath, "#!/bin/bash\necho hello");
    const destDir = mkdtempSync(join(tmpdir(), "ivllm-dest-test-"));
    try {
      const ops = makeRemoteOps(mockConfig, true, destDir);
      await ops.copyFile(srcPath, "/remote/workdir/source.sh");
      const destPath = join(destDir, "source.sh");
      expect(existsSync(destPath)).toBe(true);
      expect(readFileSync(destPath, "utf-8")).toBe("#!/bin/bash\necho hello");
    } finally {
      rmSync(destDir, { recursive: true, force: true });
    }
  });

  it("copyFile uses the basename of the remote path as the destination filename", async () => {
    const srcPath = join(dryRunDir, "local-name.sh");
    writeFileSync(srcPath, "content");
    const destDir = mkdtempSync(join(tmpdir(), "ivllm-dest-test-"));
    try {
      const ops = makeRemoteOps(mockConfig, true, destDir);
      await ops.copyFile(srcPath, "/remote/work/remote-name.slurm.sh");
      expect(existsSync(join(destDir, "remote-name.slurm.sh"))).toBe(true);
      expect(existsSync(join(destDir, "local-name.sh"))).toBe(false);
    } finally {
      rmSync(destDir, { recursive: true, force: true });
    }
  });

  it("runRemote with silent: true still returns success", async () => {
    const ops = makeRemoteOps(mockConfig, true, dryRunDir);
    const result = await ops.runRemote("test -f /remote/file", { silent: true });
    expect(result.exitCode).toBe(0);
  });
});

describe("makeRemoteOps — real mode", () => {
  it("returns an object with runRemote and copyFile methods", () => {
    const ops = makeRemoteOps(mockConfig, false);
    expect(typeof ops.runRemote).toBe("function");
    expect(typeof ops.copyFile).toBe("function");
  });
});
