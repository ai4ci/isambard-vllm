import { writeFileSync, unlinkSync, existsSync } from 'fs';
import { join, basename } from 'path';
import { tmpdir } from 'os';
import { loadConfig, assertConfigured } from '../config.ts';
import { makeRemoteOps } from '../remote-ops.ts';
import { renderInferenceScript } from '../templates/inference.ts';
import {
  parseVllmConfig,
  resolveGpuCount,
  writeStrippedConfig,
  parseEnvVars,
} from '../vllm-config.ts';
import {
  selectBestVersion,
  listInstalledVersions,
  isLocalPortInUse,
  sleep,
  timestamp,
} from '../session-helper.ts';
import { semverSort } from '../semver.ts';
import { spawn } from 'child_process';
import { parseStartArgs } from '../job.ts';

export async function cmdInteractive(args: string[]): Promise<void> {
  const config = loadConfig();
  assertConfigured(config);

  const startArgs = parseStartArgs(args);
  const { jobName, timeLimit, serverPort } = startArgs;
  let configFile = startArgs.configFile;
  const localPort = startArgs.localPort ?? config.defaultLocalPort;
  const hfHome = `${config.projectDir}/hf`;
  const remoteWorkDir = `$HOME/${jobName}`;
  const remoteWorkDirScp = `~/${jobName}`;
  const remoteJobDetails = `${remoteWorkDir}/job_details.json`;

  const ops = makeRemoteOps(config, startArgs.dryRun, undefined);

  // 1. Pre-flight checks
  const portInUse = await isLocalPortInUse(localPort);
  if (portInUse) {
    console.error(`Error: Port ${localPort} in use by PID ${portInUse.pid}.`);
    process.exit(1);
  }

  // 2. Resolve Config/Model/Version
  const yamlConfig = parseVllmConfig(configFile!);
  const model = yamlConfig.model!;
  const { gpuCount, nodeCount } = resolveGpuCount(
    startArgs.gpuCount,
    yamlConfig,
  );
  const installed = await listInstalledVersions(config, ops);
  const vllmVersion = selectBestVersion(
    installed,
    yamlConfig.minVllmVersion || '0.19.1',
  )!;

  // 3. Prepare script
  const envVars = parseEnvVars(configFile!);
  const script = renderInferenceScript({
    jobName,
    model,
    vllmVersion,
    hfHome,
    configFileName: basename(configFile!),
    workDir: remoteWorkDir,
    serverPort,
    gpuCount,
    nodeCount,
    timeLimit,
    isInteractive: true,
    envVars,
  });

  // 4. Submit via srun (synchronous, blocking)
  console.log(`Launching interactive session '${jobName}'...`);
  const srunArgs = [
    config.loginHost,
    `mkdir -p ${remoteWorkDir} && cat > ${remoteJobDetails} <<EOF
{"status": "initialising", "job_name": "${jobName}"}
EOF
`,
    `srun --nodes=${nodeCount} --gpus-per-node=${Math.floor(gpuCount / nodeCount)} --mem=0 --time=${timeLimit} bash -s < -`,
  ];

  // Actually, this is too complex for a single srun call.
  // Better pattern: copy script, then run srun script.sh
  const localScriptTmp = join(tmpdir(), `ivllm-interactive-${jobName}.sh`);
  writeFileSync(localScriptTmp, script, 'utf-8');
  await ops.copyFile(
    localScriptTmp,
    `${remoteWorkDirScp}/${jobName}.interactive.sh`,
  );

  console.log('Allocation requested. Waiting for resources...');

  const srunProcess = spawn(
    'ssh',
    [config.loginHost, `bash ${remoteWorkDir}/${jobName}.interactive.sh`],
    { stdio: 'inherit' },
  );

  // 5. Poll for "running" to start tunnel
  console.log('Waiting for vLLM to be ready...');
  while (true) {
    const { stdout } = await ops.runRemote(`cat ${remoteJobDetails}`, {
      silent: true,
    });
    if (stdout.includes('"status":"running"')) break;
    await sleep(2000);
  }

  console.log('Tunneling...');
  const tunnel = spawn('ssh', [
    '-N',
    '-L',
    `${localPort}:localhost:${serverPort}`,
    config.loginHost,
  ]);

  srunProcess.on('exit', () => {
    tunnel.kill();
    process.exit(0);
  });
}
