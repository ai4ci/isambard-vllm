import { describe, test, expect } from "bun:test";
import { parseStopArgs } from "../src/commands/stop.ts";

describe("parseStopArgs", () => {
  test("first positional arg is job name", () => {
    expect(parseStopArgs(["myjob"])).toEqual({ jobName: "myjob" });
  });

  test("no args throws error", () => {
    expect(() => parseStopArgs([])).toThrow(/job name/i);
  });

  test("flag as first arg throws error", () => {
    expect(() => parseStopArgs(["--force"])).toThrow(/job name/i);
  });

  test("ignores extra positional args after job name", () => {
    expect(parseStopArgs(["myjob", "extra"])).toEqual({ jobName: "myjob" });
  });
});
