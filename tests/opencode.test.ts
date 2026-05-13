import { describe, it, expect } from "bun:test";
import { formatOpencodeSnippet } from "../src/opencode.ts";

describe("formatOpencodeSnippet", () => {
  const base = { model: "google/gemma-4-31B-it", localPort: 11434, maxModelLen: 262144 };

  it("includes $schema for opencode", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out["$schema"]).toBe("https://opencode.ai/config.json");
  });

  it("sets the default model to isambard-vllm/<model>", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.model).toBe("isambard-vllm/google/gemma-4-31B-it");
  });

  it("includes provider key isambard-vllm", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"]).toBeDefined();
  });

  it("sets the npm package", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].npm).toBe("@ai-sdk/openai-compatible");
  });

  it("sets the provider name", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].name).toBe("Isambard vLLM Server");
  });

  it("sets baseURL using localPort", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].options.baseURL).toBe("http://localhost:11434/v1");
  });

  it("uses a different localPort in baseURL", () => {
    const out = JSON.parse(formatOpencodeSnippet({ ...base, localPort: 9999 }));
    expect(out.provider["isambard-vllm"].options.baseURL).toBe("http://localhost:9999/v1");
  });

  it("sets apiKey to EMPTY", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].options.apiKey).toBe("EMPTY");
  });

  it("includes the model in the models map", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"]).toBeDefined();
  });

  it("sets the model display name with (Isambard) suffix", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"].name)
      .toBe("google/gemma-4-31B-it (Isambard)");
  });

  it("sets context limit from maxModelLen", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"].limit.context).toBe(262144);
  });

  it("sets output limit equal to context limit", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"].limit.output).toBe(262144);
  });

  it("defaults context/output to 4096 when maxModelLen is not provided", () => {
    const out = JSON.parse(formatOpencodeSnippet({ model: "some/model", localPort: 11434 }));
    const m = out.provider["isambard-vllm"].models["some/model"];
    expect(m.limit.context).toBe(4096);
    expect(m.limit.output).toBe(4096);
  });

  it("includes tool_call: true when toolCall is true", () => {
    const out = JSON.parse(formatOpencodeSnippet({ ...base, toolCall: true }));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"].tool_call).toBe(true);
  });

  it("omits tool_call when toolCall is false", () => {
    const out = JSON.parse(formatOpencodeSnippet({ ...base, toolCall: false }));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"].tool_call).toBeUndefined();
  });

  it("omits tool_call when toolCall is not provided", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"].tool_call).toBeUndefined();
  });

  it("includes reasoning: true when reasoning is true", () => {
    const out = JSON.parse(formatOpencodeSnippet({ ...base, reasoning: true }));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"].reasoning).toBe(true);
  });

  it("omits reasoning when reasoning is false", () => {
    const out = JSON.parse(formatOpencodeSnippet({ ...base, reasoning: false }));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"].reasoning).toBeUndefined();
  });

  it("omits reasoning when not provided", () => {
    const out = JSON.parse(formatOpencodeSnippet(base));
    expect(out.provider["isambard-vllm"].models["google/gemma-4-31B-it"].reasoning).toBeUndefined();
  });

  it("produces valid JSON", () => {
    expect(() => JSON.parse(formatOpencodeSnippet({ ...base, toolCall: true, reasoning: true }))).not.toThrow();
  });
});
