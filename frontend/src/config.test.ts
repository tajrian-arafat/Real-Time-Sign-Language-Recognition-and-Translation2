import { describe, expect, it } from "vitest";

/** Scaffold: frontend test runner wired (expand when UI modules land). */
describe("frontend scaffold", () => {
  it("vitest runs in the Vite project", () => {
    expect(import.meta.env.MODE).toBeDefined();
  });
});
