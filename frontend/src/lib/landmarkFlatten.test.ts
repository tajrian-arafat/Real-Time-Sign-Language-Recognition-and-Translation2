import { describe, expect, it } from "vitest";
import { flattenLandmarkFrame, LANDMARK_FEATURE_DIM } from "./landmarkFlatten";

describe("flattenLandmarkFrame", () => {
  it("matches backend feature dimension (392)", () => {
    const vec = flattenLandmarkFrame({
      hands: [
        Array.from({ length: 21 }, (_, i) => ({ x: i * 0.01, y: 0.1, z: 0 })),
      ],
      pose: Array.from({ length: 33 }, (_, i) => ({
        x: 0.5,
        y: i * 0.01,
        z: 0,
      })),
      face: Array.from({ length: 40 }, () => ({ x: 0.2, y: 0.3, z: 0.01 })),
    });
    expect(vec).toHaveLength(LANDMARK_FEATURE_DIM);
    expect(vec[0]).toBeCloseTo(0);
    expect(vec[1]).toBeCloseTo(0.1);
  });

  it("zero-pads when detections are sparse", () => {
    const vec = flattenLandmarkFrame({ hands: [], pose: undefined, face: undefined });
    expect(vec).toHaveLength(LANDMARK_FEATURE_DIM);
    expect(vec.every((v) => v === 0)).toBe(true);
  });
});
