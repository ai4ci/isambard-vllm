import { describe, it, expect } from "bun:test";
import { parseJobId, parseJobState, parseSlurmQueueState } from "../src/slurm.ts";

describe("parseJobId", () => {
  it("extracts job ID from standard sbatch output", () => {
    expect(parseJobId("Submitted batch job 12345")).toBe("12345");
  });

  it("returns null for unrecognised output", () => {
    expect(parseJobId("error: something went wrong")).toBeNull();
  });

  it("returns null for empty string", () => {
    expect(parseJobId("")).toBeNull();
  });
});

describe("parseJobState", () => {
  it("maps COMPLETED to 'completed'", () => {
    expect(parseJobState("COMPLETED")).toBe("completed");
  });

  it("maps RUNNING to 'running'", () => {
    expect(parseJobState("RUNNING")).toBe("running");
  });

  it("maps PENDING to 'running' (job exists, not yet started)", () => {
    expect(parseJobState("PENDING")).toBe("running");
  });

  it("maps FAILED to 'failed'", () => {
    expect(parseJobState("FAILED")).toBe("failed");
  });

  it("maps CANCELLED to 'failed'", () => {
    expect(parseJobState("CANCELLED")).toBe("failed");
  });

  it("maps TIMEOUT to 'failed'", () => {
    expect(parseJobState("TIMEOUT")).toBe("failed");
  });

  it("returns null for empty string", () => {
    expect(parseJobState("")).toBeNull();
  });
});

describe("parseSlurmQueueState", () => {
  it("parses PENDING state with reason", () => {
    const result = parseSlurmQueueState("PENDING Priority");
    expect(result).toEqual({ state: "PENDING", reason: "Priority" });
  });

  it("parses RUNNING state with reason None", () => {
    const result = parseSlurmQueueState("RUNNING None");
    expect(result).toEqual({ state: "RUNNING", reason: "None" });
  });

  it("parses state with multi-word reason", () => {
    const result = parseSlurmQueueState("PENDING Resources");
    expect(result).toEqual({ state: "PENDING", reason: "Resources" });
  });

  it("returns null for empty output (job not in queue)", () => {
    expect(parseSlurmQueueState("")).toBeNull();
  });

  it("returns null for whitespace-only output", () => {
    expect(parseSlurmQueueState("   \n  ")).toBeNull();
  });

  it("handles leading/trailing whitespace", () => {
    const result = parseSlurmQueueState("  PENDING Priority  ");
    expect(result).toEqual({ state: "PENDING", reason: "Priority" });
  });
});
