import { describe, it, expect } from 'bun:test';
import { PortPoolManager } from '../../src/router/port-pool.js';

describe('Port Pool Manager', () => {
  describe('acquire', () => {
    it('acquires ports sequentially from start', () => {
      const pool = new PortPoolManager(11435, 11444); // Small range for testing
      
      const port1 = pool.acquire();
      const port2 = pool.acquire();
      const port3 = pool.acquire();
      
      expect(port1).toBe(11435);
      expect(port2).toBe(11436);
      expect(port3).toBe(11437);
    });

    it('throws when pool exhausted', () => {
      const pool = new PortPoolManager(11435, 11435); // Only 1 port
      
      pool.acquire();
      
      expect(() => pool.acquire()).toThrow('No available ports in range 11435-11435');
    });

    it('skips already acquired ports', () => {
      const pool = new PortPoolManager(11435, 11439);
      
      pool.acquire(); // 11435
      pool.release(11435);
      pool.acquire(); // Should get 11435 again
      
      expect(pool.acquire()).toBe(11436);
    });
  });

  describe('release', () => {
    it('releases port back to pool', () => {
      const pool = new PortPoolManager(11435, 11439);
      
      const port = pool.acquire();
      pool.release(port);
      
      expect(pool.isAvailable(port)).toBe(true);
      expect(pool.acquire()).toBe(port); // Can acquire again
    });

    it('throws on port outside range', () => {
      const pool = new PortPoolManager(11435, 11439);
      
      expect(() => pool.release(11434)).toThrow('outside managed range');
      expect(() => pool.release(11440)).toThrow('outside managed range');
    });
  });

  describe('isAvailable', () => {
    it('returns false for acquired ports', () => {
      const pool = new PortPoolManager(11435, 11439);
      
      const port = pool.acquire();
      expect(pool.isAvailable(port)).toBe(false);
    });

    it('returns true for unacquired ports in range', () => {
      const pool = new PortPoolManager(11435, 11439);
      
      expect(pool.isAvailable(11435)).toBe(true);
      expect(pool.isAvailable(11438)).toBe(true);
    });

    it('returns false for ports outside range', () => {
      const pool = new PortPoolManager(11435, 11439);
      
      expect(pool.isAvailable(11434)).toBe(false);
      expect(pool.isAvailable(11440)).toBe(false);
    });
  });

  describe('default range', () => {
    it('uses default range 11435-11534', () => {
      const pool = new PortPoolManager();
      
      expect(pool.isAvailable(11435)).toBe(true);
      expect(pool.isAvailable(11534)).toBe(true);
      expect(pool.isAvailable(11434)).toBe(false);
      expect(pool.isAvailable(11535)).toBe(false);
    });
  });
});
