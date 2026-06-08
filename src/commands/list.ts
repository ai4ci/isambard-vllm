import { listJobConfigs } from '../vllm-config.ts';

export async function cmdList(args: string[]): Promise<void> {
  if (args.includes('--help') || args.includes('-h')) {
    console.log(
      `
Usage: ivllm list

List stored vLLM job configs in ~/.config/ivllm/.
Each job has a stored YAML config containing the model and parallelism settings.

Examples:
  ivllm list
`.trim(),
    );
    return;
  }

  const entries = listJobConfigs();

  if (entries.length === 0) {
    console.log('No stored vLLM configs found in ~/.config/ivllm/.');
    console.log(
      '  First run: ivllm start <job-name> --config <path-to-vllm.yaml>',
    );
    return;
  }

  console.log(`Stored vLLM configs (${entries.length}):`);
  console.log('');

  // Compute column widths
  const nameWidth = Math.max(
    'JOB'.length,
    ...entries.map((e) => e.jobName.length),
  );
  const modelWidth = Math.max(
    'MODEL'.length,
    ...entries.map((e) => (e.model || '').length),
  );

  const header = `${'JOB'.padEnd(nameWidth)}  ${'MODEL'.padEnd(modelWidth)}  TP  PP`;
  const separator = `${'─'.repeat(nameWidth)}  ${'─'.repeat(modelWidth)}  ──  ──`;

  console.log(header);
  console.log(separator);

  for (const entry of entries) {
    const model = entry.model || '(unknown)';
    const tp = entry.tensorParallelSize?.toString() ?? '—';
    const pp = entry.pipelineParallelSize?.toString() ?? '—';
    console.log(
      `${entry.jobName.padEnd(nameWidth)}  ${model.padEnd(modelWidth)}  ${tp.padStart(2)}  ${pp.padStart(2)}`,
    );
  }
}
