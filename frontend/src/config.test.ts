import { describe, expect, it } from "vitest";

/** Frontend config defaults (real backend unless mock env is set). */
describe("frontend config", () => {
  it("vitest runs in the Vite project", () => {
    expect(import.meta.env.MODE).toBeDefined();
  });

  it("defaults to real WebSocket unless VITE_MOCK_WS is true", () => {
    const mockFlag = import.meta.env.VITE_MOCK_WS;
    const mockEnabled = mockFlag === "true" || mockFlag === "1";
    expect(mockEnabled).toBe(false);
  });
});
