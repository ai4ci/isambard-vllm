import { ModelRegistry } from './model-registry.js';
import { RemoteExecutor } from '../types.js';
import { renderInferenceScript } from '../templates/inference.js';
import { parseVllmConfig } from '../vllm-config.js';

interface JobDetails {
  status: string;
  slurm_job_id?: number;
  server_port?: number;
  node_hostname?: string;
  error?: string;
}

/**
 * Manages SLURM job lifecycle for models
 */
export class JobManager {
  private executor: RemoteExecutor;
  private registry: ModelRegistry;
  private loginHost: string;

  constructor(executor: RemoteExecutor, registry: ModelRegistry, loginHost: string) {
    this.executor = executor;
    this.registry = registry;
    this.loginHost = loginHost;
  }

  /**
   * Submit a SLURM job for a model
   */
  async submitJob(modelName: string, configPath: string): Promise<number> {
    const model = this.registry.getModel(modelName);
    if (!model) throw new Error(`Model ${modelName} not found`);

    // Parse vLLM config to get model details
    const vllmConfig = await parseVllmConfig(configPath);
    const port = this.registry.acquirePort();

    // Render SLURM script
    const slurmScript = renderInferenceScript({
      jobName: `router-${modelName}`,
      configPath,
      model: vllmConfig.model,
      port,
      nodeCount: 1, // Router uses single-node for now
    });

    // Write script to temp location
    const fs = await import('fs');
    const path = await import('path');
    const tempDir = path.join('/tmp', `router-${Date.now()}`);
    const scriptPath = path.join(tempDir, 'submit.sls');
    fs.mkdirSync(tempDir, { recursive: true });
    fs.writeFileSync(scriptPath, slurmScript);

    try {
      // Copy script to LOGIN
      const remoteScriptPath = `/tmp/router-${modelName}.sls`;
      await this.executor.copyFile(scriptPath, remoteScriptPath);

      // Submit job
      const jobIdStr = await this.executor.runCommand(`sbatch ${remoteScriptPath}`);
      const jobId = parseInt(jobIdStr.match(/Submitted batch (\d+)/)?.[1] || '0');

      if (!jobId) throw new Error('Failed to parse SLURM job ID');

      // Update registry state
      this.registry.updateState(modelName, {
        status: 'starting',
        slurmJobId: jobId,
        port,
      });

      return jobId;
    } finally {
      // Cleanup temp files
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  }

  /**
   * Poll job status via job_details.json
   */
  async pollJobStatus(modelName: string): Promise<JobDetails> {
    const state = this.registry.getState(modelName);
    if (!state?.slurmJobId) throw new Error(`No job found for model ${modelName}`);

    try {
      // Read job_details.json from remote
      const jobDetailsPath = `/tmp/router-${modelName}/job_details.json`;
      const content = await this.executor.readFile(jobDetailsPath);
      const details = JSON.parse(content) as JobDetails;

      // Update registry state
      const updates: Partial<any> = {};
      if (details.status === 'running') {
        updates.status = 'running';
        updates.startedAt = new Date();
        updates.lastActivityAt = new Date();
        updates.nodeHostname = details.node_hostname;
        updates.server_port = details.server_port;
      } else if (details.status === 'failed' || details.status === 'timeout') {
        updates.status = 'failed';
        updates.error = details.error;
      }

      this.registry.updateState(modelName, updates);
      return details;
    } catch (error) {
      // File might not exist yet or job failed
      return {
        status: 'starting',
        slurm_job_id: state.slurmJobId,
      };
    }
  }

  /**
   * Cancel a SLURM job
   */
  async cancelJob(modelName: string): Promise<void> {
    const state = this.registry.getState(modelName);
    if (!state?.slurmJobId) return;

    try {
      await this.executor.runCommand(`scancel ${state.slurmJobId}`);
    } catch (error) {
      // Job might already be cancelled
    }

    // Release port
    if (state.port) {
      this.registry.releasePort(state.port);
    }

    // Update state
    this.registry.updateState(modelName, {
      status: 'stopped',
      slurmJobId: undefined,
      port: undefined,
      nodeHostname: undefined,
    });
  }

  /**
   * Get logs from a running job
   */
  async getLogs(modelName: string): Promise<string> {
    const state = this.registry.getState(modelName);
    if (!state) throw new Error(`Model ${modelName} not found`);

    const logPath = `/tmp/router-${modelName}/vllm.log`;
    try {
      return await this.executor.readFile(logPath);
    } catch {
      return 'No logs available';
    }
  }
}
