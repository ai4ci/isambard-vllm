import { PortPool } from './types.js';

/**
 * Default port range for vLLM backends
 */
const DEFAULT_PORT_START = 11435;
const DEFAULT_PORT_END = 11534;

/**
 * Manages a pool of available ports for vLLM backends
 */
export class PortPoolManager implements PortPool {
  private readonly startPort: number;
  private readonly endPort: number;
  private readonly acquiredPorts: Set<number> = new Set();

  constructor(startPort: number = DEFAULT_PORT_START, endPort: number = DEFAULT_PORT_END) {
    this.startPort = startPort;
    this.endPort = endPort;
  }

  /**
   * Acquire an available port from the pool
   * @throws Error if no ports available
   */
  acquire(): number {
    for (let port = this.startPort; port <= this.endPort; port++) {
      if (!this.acquiredPorts.has(port)) {
        this.acquiredPorts.add(port);
        return port;
      }
    }
    throw new Error(`No available ports in range ${this.startPort}-${this.endPort}`);
  }

  /**
   * Release a port back to the pool
   */
  release(port: number): void {
    if (port < this.startPort || port > this.endPort) {
      throw new Error(`Port ${port} is outside managed range ${this.startPort}-${this.endPort}`);
    }
    this.acquiredPorts.delete(port);
  }

  /**
   * Check if a port is available
   */
  isAvailable(port: number): boolean {
    if (port < this.startPort || port > this.endPort) {
      return false;
    }
    return !this.acquiredPorts.has(port);
  }

  /**
   * Get count of currently acquired ports (for testing)
   */
  get acquiredCount(): number {
    return this.acquiredPorts.size;
  }
}
