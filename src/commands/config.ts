import { loadCredentials, saveConfig } from '../config.ts';

/**
 *
 * @param args
 */
export async function cmdConfig(args: string[]): Promise<void> {
    // Handle help flag
    if (args.includes('--help') || args.includes('-h')) {
        console.log(`
        Usage: ivllm config [options]

        Options:
        --login-host <host>     SSH login node (e.g. XXXX.aip2.isambard)
        --username <user>       HPC username (e.g. YYYY.XXXX)
        --project-dir <path>    HPC project dir (e.g. /projects/XXXX)
        --default-local-port <port>  Local port for API (default: 11434)
        --hf-token <token>      HuggingFace token for gated models
        --help, -h              Show this help message

        Examples:
        ivllm config --login-host XXXX.aip2.isambard --username YYYY.XXXX --project-dir /projects/XXXX
        ivllm config --hf-token hf_...
        ivllm config  # Show current configuration
        `);
        return;
    }

    const config = loadCredentials();
    const flags: Record<string, string> = {};
    for (let i = 0; i < args.length - 1; i++) {
        if (args[i]?.startsWith('--')) flags[args[i]!.slice(2)] = args[i + 1] ?? '';
    }
    if (Object.keys(flags).length === 0) {
        console.log(JSON.stringify(config, null, 2));
        return;
    }
    if (flags['login-host']) config.loginHost = flags['login-host']!;
    if (flags['username']) config.username = flags['username']!;
    if (flags['project-dir']) config.projectDir = flags['project-dir']!;
    if (flags['default-local-port'])
        config.defaultLocalPort = parseInt(flags['default-local-port']!, 10);
    if (flags['hf-token']) config.hfToken = flags['hf-token']!;
    saveConfig(config);
    console.log('Configuration saved.');
}
