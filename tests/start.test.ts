import { describe, it, expect } from 'bun:test';
import { parseJobDetails, hfCachePath, parseStartArgs } from '../src/job.ts';
import type { Credentials } from '../src/types.ts';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { writeFileSync } from 'node:fs';

const creds: Credentials = {
  loginHost: 'test.example.com',
  username: 'test-user',
  projectDir: '/projects/p',
  defaultLocalPort: 11434,
  hfToken: 'HFTOKEN',
};

function writeTmp(content: string): string {
  const path = join(tmpdir(), `ivllm-test-${Date.now()}.yaml`);
  writeFileSync(path, content, 'utf-8');
  return path;
}

describe('parseJobDetails', () => {
  it('parses a complete running job', () => {
    const json = JSON.stringify({
      status: 'running',
      job_name: 'my-job',
      slurm_job_id: '12345',
      compute_hostname: 'compute01',
      model: 'Qwen/Qwen2.5-0.5B-Instruct',
      server_port: 8000,
    });
    const result = parseJobDetails(json);
    expect(result?.status).toBe('running');
    expect(result?.slurm_job_id).toBe('12345');
    expect(result?.compute_hostname).toBe('compute01');
    expect(result?.server_port).toBe(8000);
  });

  it('parses a pending job with only required fields', () => {
    const json = JSON.stringify({ status: 'pending', job_name: 'my-job' });
    const result = parseJobDetails(json);
    expect(result?.status).toBe('pending');
    expect(result?.job_name).toBe('my-job');
  });

  it('parses a failed job with error field', () => {
    const json = JSON.stringify({
      status: 'failed',
      job_name: 'my-job',
      error: 'vLLM process died during startup',
    });
    const result = parseJobDetails(json);
    expect(result?.status).toBe('failed');
    expect(result?.error).toBe('vLLM process died during startup');
  });

  it('returns null for empty string', () => {
    expect(parseJobDetails('')).toBeNull();
  });

  it('returns null for invalid JSON', () => {
    expect(parseJobDetails('not json')).toBeNull();
  });

  it('returns null for JSON missing status field', () => {
    expect(parseJobDetails(JSON.stringify({ job_name: 'x' }))).toBeNull();
  });
});

describe('hfCachePath', () => {
  it('builds path for org/model format', () => {
    expect(hfCachePath('/projects/p/hf', 'Qwen/Qwen2.5-0.5B-Instruct')).toBe(
      '/projects/p/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct',
    );
  });

  it('builds path for org with hyphens', () => {
    expect(hfCachePath('/projects/p/hf', 'meta-llama/Llama-3-8b')).toBe(
      '/projects/p/hf/hub/models--meta-llama--Llama-3-8b',
    );
  });

  it('builds path for model with no org', () => {
    expect(hfCachePath('/projects/p/hf', 'gpt2')).toBe(
      '/projects/p/hf/hub/models--gpt2',
    );
  });
});

const path = writeTmp(
  'model: Qwen/Qwen2.5-0.5B-Instruct\nmax-model-len: 8192\n',
);

describe('parseStartArgs', () => {
  it('parses required args (non-mock: --config only, model comes from YAML)', async () => {
    const result = await parseStartArgs(['my-job', '--config', path], creds);
    expect(result.jobName).toBe('my-job');
    expect(result.configFile).toBe(path);
  });

  it('applies defaults for optional args', async () => {
    const result = await parseStartArgs(['my-job', '--config', path], creds);
    expect(result.gpuCount).toBe(1);
    expect(result.timeLimit).toBe('8:00:00');
    expect(result.serverPort).toBe(8000);
  });

  it('parses optional --local-port', async () => {
    const result = await parseStartArgs(
      ['my-job', '--config', path, '--local-port', '11435'],
      creds,
    );
    expect(result.localPort).toBe(11435);
  });

  it('parses optional --gpus', async () => {
    const result = await parseStartArgs(
      ['my-job', '--config', path, '--gpus', '8'],
      creds,
    );
    expect(result.gpuCount).toBe(8);
  });

  it('parses optional --time', async () => {
    const result = await parseStartArgs(
      ['my-job', '--config', path, '--time', '8:00:00'],
      creds,
    );
    expect(result.timeLimit).toBe('8:00:00');
  });

  it('throws when job name is missing', () => {
    expect(() => parseStartArgs(['--config', path], creds)).toThrow(
      /job name/i,
    );
  });

  it('--config is still accepted when provided', async () => {
    const result = await parseStartArgs(['my-job', '--config', path], creds);
    expect(result.configFile).toBe(path);
  });

  it('--dry-run flag sets dryRun: true', async () => {
    const result = await parseStartArgs(
      ['my-job', '--config', path, '--dry-run'],
      creds,
    );
    expect(result.dryRun).toBe(true);
  });

  it('dryRun defaults to false when flag absent', async () => {
    const result = await parseStartArgs(['my-job', '--config', path], creds);
    expect(result.dryRun).toBe(false);
  });

  it('--mock flag sets mock: true', async () => {
    const result = await parseStartArgs(
      ['my-job', '--model', 'm', '--config', path, '--mock'],
      creds,
    );
    expect(result.mock).toBe(true);
  });

  it('mock defaults to false when flag absent', async () => {
    const result = await parseStartArgs(['my-job', '--config', path], creds);
    expect(result.mock).toBe(false);
  });
});
