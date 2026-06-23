import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import {
  mkdtempSync,
  writeFileSync,
  readFileSync,
  existsSync,
  rmSync,
} from 'fs';
import { tmpdir } from 'os';
import { join, basename } from 'path';
import { makeRemoteOps } from '../src/remote-ops.ts';
import type { Config } from '../src/config.ts';

const mockConfig: Config = {
  loginHost: 'login.example.com',
  username: 'testuser',
  projectDir: '/projects/myproject',
  defaultLocalPort: 11434,
};

describe('makeRemoteOps — dry-run mode', () => {
  let dryRunDir: string;

  beforeEach(() => {
    dryRunDir = mkdtempSync(join(tmpdir(), 'ivllm-ops-test-'));
  });

  afterEach(() => {
    rmSync(dryRunDir, { recursive: true, force: true });
  });

  it('runRemote returns exitCode 0 without executing SSH', async () => {
    const ops = makeRemoteOps(mockConfig, true);
    const result = await ops.runRemote('echo hello');
    expect(result.exitCode).toBe(0);
  });

  it('runRemote with silent: true still returns success', async () => {
    const ops = makeRemoteOps(mockConfig, true, dryRunDir);
    const result = await ops.runRemote('test -f /remote/file', {
      silent: true,
      env: [],
    });
    expect(result.exitCode).toBe(0);
  });
});

describe('makeRemoteOps — real mode', () => {
  it('returns an object with runRemote and copyFile methods', () => {
    const ops = makeRemoteOps(mockConfig, false);
    expect(typeof ops.runRemote).toBe('function');
    expect(typeof ops.copyFile).toBe('function');
  });
});
