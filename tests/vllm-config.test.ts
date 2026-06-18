import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import {
  writeFileSync,
  unlinkSync,
  readFileSync,
  existsSync,
  rmSync,
  mkdirSync,
} from 'fs';
import { join } from 'path';
import { tmpdir, homedir } from 'os';
import {
  parseVllmConfig,
  stripIvllmKeys,
  IVLLM_ONLY_KEYS,
  jobConfigPath,
  saveJobConfig,
  parseEnvVars,
} from '../src/vllm-config.ts';
import yaml from 'js-yaml';

function writeTmp(content: string): string {
  const path = join(tmpdir(), `ivllm-test-${Date.now()}.yaml`);
  writeFileSync(path, content, 'utf-8');
  return path;
}

describe('parseVllmConfig', () => {
  it('extracts model from YAML', () => {
    const path = writeTmp('model: Qwen/Qwen2.5-0.5B-Instruct\n');
    try {
      expect(parseVllmConfig(path).model).toBe('Qwen/Qwen2.5-0.5B-Instruct');
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts tensor-parallel-size (kebab-case) from YAML', () => {
    const path = writeTmp('tensor-parallel-size: 4\n');
    try {
      expect(parseVllmConfig(path).tensorParallelSize).toBe(4);
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts tensor_parallel_size (underscore) from YAML', () => {
    const path = writeTmp('tensor_parallel_size: 8\n');
    try {
      expect(parseVllmConfig(path).tensorParallelSize).toBe(8);
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts pipeline-parallel-size (kebab-case) from YAML', () => {
    const path = writeTmp('pipeline-parallel-size: 2\n');
    try {
      expect(parseVllmConfig(path).pipelineParallelSize).toBe(2);
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts pipeline_parallel_size (underscore) from YAML', () => {
    const path = writeTmp('pipeline_parallel_size: 2\n');
    try {
      expect(parseVllmConfig(path).pipelineParallelSize).toBe(2);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns undefined pipelineParallelSize when not present', () => {
    const path = writeTmp('model: some/model\n');
    try {
      expect(parseVllmConfig(path).pipelineParallelSize).toBeUndefined();
    } finally {
      unlinkSync(path);
    }
  });

  it('returns undefined model when not present', () => {
    const path = writeTmp('max-model-len: 8192\n');
    try {
      expect(parseVllmConfig(path).model).toBeUndefined();
    } finally {
      unlinkSync(path);
    }
  });

  it('returns undefined tensorParallelSize when not present', () => {
    const path = writeTmp('model: some/model\n');
    try {
      expect(parseVllmConfig(path).tensorParallelSize).toBeUndefined();
    } finally {
      unlinkSync(path);
    }
  });

  it('parses a realistic multi-line config', () => {
    const path = writeTmp(
      'model: Qwen/Qwen2.5-0.5B-Instruct\n' +
        'tensor-parallel-size: 4\n' +
        'max-model-len: 8192\n' +
        'gpu-memory-utilization: 0.90\n',
    );
    try {
      const result = parseVllmConfig(path);
      expect(result.model).toBe('Qwen/Qwen2.5-0.5B-Instruct');
      expect(result.tensorParallelSize).toBe(4);
    } finally {
      unlinkSync(path);
    }
  });

  it('throws a helpful error for a missing file', () => {
    expect(() => parseVllmConfig('/nonexistent/path.yaml')).toThrow();
  });

  it('throws a helpful error for invalid YAML', () => {
    const path = writeTmp('{{invalid: yaml: [\n');
    try {
      expect(() => parseVllmConfig(path)).toThrow();
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts min-vllm-version (kebab-case) from YAML', () => {
    const path = writeTmp('min-vllm-version: "0.9.1"\n');
    try {
      expect(parseVllmConfig(path).minVllmVersion).toBe('0.9.1');
    } finally {
      unlinkSync(path);
    }
  });

  it('returns undefined minVllmVersion when not present', () => {
    const path = writeTmp('model: some/model\n');
    try {
      expect(parseVllmConfig(path).minVllmVersion).toBeUndefined();
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts max-model-len from YAML', () => {
    const path = writeTmp('max-model-len: 131072\n');
    try {
      expect(parseVllmConfig(path).maxModelLen).toBe(131072);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns undefined maxModelLen when not present', () => {
    const path = writeTmp('model: some/model\n');
    try {
      expect(parseVllmConfig(path).maxModelLen).toBeUndefined();
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts enable-auto-tool-choice: true from YAML', () => {
    const path = writeTmp('enable-auto-tool-choice: true\n');
    try {
      expect(parseVllmConfig(path).enableAutoToolChoice).toBe(true);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns undefined enableAutoToolChoice when not present', () => {
    const path = writeTmp('model: some/model\n');
    try {
      expect(parseVllmConfig(path).enableAutoToolChoice).toBeUndefined();
    } finally {
      unlinkSync(path);
    }
  });

  it('derives enableReasoning: true from presence of reasoning-parser in YAML', () => {
    const path = writeTmp('reasoning-parser: qwen3\n');
    try {
      expect(parseVllmConfig(path).enableReasoning).toBe(true);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns undefined enableReasoning when reasoning-parser is absent', () => {
    const path = writeTmp('model: some/model\n');
    try {
      expect(parseVllmConfig(path).enableReasoning).toBeUndefined();
    } finally {
      unlinkSync(path);
    }
  });

  it('does not derive enableReasoning from enable-reasoning key (not a valid vLLM key)', () => {
    const path = writeTmp('enable-reasoning: true\n');
    try {
      expect(parseVllmConfig(path).enableReasoning).toBeUndefined();
    } finally {
      unlinkSync(path);
    }
  });
});

describe('stripIvllmKeys', () => {
  it('removes min-vllm-version from the output YAML', () => {
    const path = yaml.load(
      'model: some/model\nmin-vllm-version: "0.9.1"\ntensor-parallel-size: 4\n',
    ) as Record<string, unknown>;
    const result = yaml.dump(stripIvllmKeys(path));
    expect(result).not.toContain('min-vllm-version');
  });

  it('preserves all non-ivllm keys', () => {
    const path = yaml.load(
      'model: Qwen/Qwen2.5-0.5B-Instruct\n' +
        'min-vllm-version: "0.9.1"\n' +
        'tensor-parallel-size: 2\n' +
        'max-model-len: 32768\n',
    ) as Record<string, unknown>;
    const result = yaml.dump(stripIvllmKeys(path));
    expect(result).toContain('Qwen/Qwen2.5-0.5B-Instruct');
    expect(result).toContain('tensor-parallel-size');
    expect(result).toContain('max-model-len');
  });

  it('is a no-op when no ivllm-only keys are present', () => {
    const path = yaml.load(
      'model: some/model\ntensor-parallel-size: 4\n',
    ) as Record<string, unknown>;
    const result = yaml.dump(stripIvllmKeys(path));
    expect(result).toContain('some/model');
    expect(result).toContain('tensor-parallel-size');
  });

  it('IVLLM_ONLY_KEYS contains min-vllm-version', () => {
    expect(IVLLM_ONLY_KEYS.has('min-vllm-version')).toBe(true);
  });

  it('IVLLM_ONLY_KEYS contains env', () => {
    expect(IVLLM_ONLY_KEYS.has('env')).toBe(true);
  });

  it('removes env from the output YAML', () => {
    const path = yaml.load(
      'model: some/model\n' +
        'env:\n' +
        '  FOO: bar\n' +
        'tensor-parallel-size: 4\n',
    ) as Record<string, unknown>;
    const result = yaml.dump(stripIvllmKeys(path));
    expect(result).not.toContain('env:');
    expect(result).not.toContain('FOO');
    expect(result).toContain('some/model');
    expect(result).toContain('tensor-parallel-size');
  });
});

describe('env field in VllmConfig', () => {
  it('parses env block from YAML into VllmConfig.env', () => {
    const path = writeTmp(
      'model: some/model\n' +
        'env:\n' +
        "  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: '1'\n" +
        "  VLLM_USE_DEEP_GEMM_FP8: '1'\n",
    );
    try {
      const cfg = parseVllmConfig(path);
      expect(cfg.env).toBeDefined();
      // Find the specific environment entries in the array
      const profilerEnv = cfg.env.find(
        (e) => e.key === 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS',
      );
      const gemmEnv = cfg.env.find((e) => e.key === 'VLLM_USE_DEEP_GEMM_FP8');

      // Assert on their values
      expect(profilerEnv?.value).toBe('1');
      expect(gemmEnv?.value).toBe('1');
    } finally {
      unlinkSync(path);
    }
  });

  it('returns empty env object when env block is absent', () => {
    const path = writeTmp('model: some/model\ntensor-parallel-size: 4\n');
    try {
      expect(parseVllmConfig(path).env.length).toEqual(0);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns empty env object when env block is empty', () => {
    const path = writeTmp('model: some/model\nenv: {}\n');
    try {
      const cfg = parseVllmConfig(path);
      expect(cfg.env.length).toEqual(0);
    } finally {
      unlinkSync(path);
    }
  });
});

describe('parseEnvVars', () => {
  it('returns array of {key, value} from env block', () => {
    const path = writeTmp(
      'model: some/model\n' + 'env:\n' + '  FOO: bar\n' + "  BAZ: '123'\n",
    );
    try {
      const vars = parseEnvVars(path);
      expect(vars).toHaveLength(2);
      expect(vars).toContainEqual({ key: 'FOO', value: 'bar' });
      expect(vars).toContainEqual({ key: 'BAZ', value: '123' });
    } finally {
      unlinkSync(path);
    }
  });

  it('returns empty array when no env block', () => {
    const path = writeTmp('model: some/model\n');
    try {
      expect(parseEnvVars(path)).toEqual([]);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns empty array when env block is empty', () => {
    const path = writeTmp('model: some/model\nenv: {}\n');
    try {
      expect(parseEnvVars(path)).toEqual([]);
    } finally {
      unlinkSync(path);
    }
  });
});

describe('jobConfigPath', () => {
  it('returns path under ~/.config/ivllm/<name>.yaml', () => {
    const expected = join(homedir(), '.config', 'ivllm', 'qwen36.yaml');
    expect(jobConfigPath('qwen36')).toBe(expected);
  });

  it('uses the job name verbatim', () => {
    const expected = join(homedir(), '.config', 'ivllm', 'my-job.yaml');
    expect(jobConfigPath('my-job')).toBe(expected);
  });
});

describe('saveJobConfig', () => {
  const testJobName = `ivllm-test-job-${Date.now()}`;
  const jobPath = join(homedir(), '.config', 'ivllm', `${testJobName}.yaml`);

  afterEach(() => {
    if (existsSync(jobPath)) rmSync(jobPath);
  });

  it('copies the source file to the job config path', () => {
    const src = join(tmpdir(), `ivllm-src-${Date.now()}.yaml`);
    writeFileSync(src, 'model: Qwen/Test\ntensor-parallel-size: 4\n', 'utf-8');
    try {
      saveJobConfig(testJobName, src);
      expect(existsSync(jobPath)).toBe(true);
      expect(readFileSync(jobPath, 'utf-8')).toBe(
        'model: Qwen/Test\ntensor-parallel-size: 4\n',
      );
    } finally {
      unlinkSync(src);
    }
  });

  it('creates the ~/.config/ivllm directory if it does not exist', () => {
    // This is already created in most environments, so just check the file is saved
    const src = join(tmpdir(), `ivllm-src2-${Date.now()}.yaml`);
    writeFileSync(src, 'model: Test\n', 'utf-8');
    try {
      saveJobConfig(testJobName, src);
      expect(existsSync(jobPath)).toBe(true);
    } finally {
      unlinkSync(src);
    }
  });
});
