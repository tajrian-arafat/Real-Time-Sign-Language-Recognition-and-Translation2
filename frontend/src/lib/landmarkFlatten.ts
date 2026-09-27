/** Flatten MediaPipe Tasks detections to match backend/landmark_server.py (392 dims). */

export const LANDMARK_FEATURE_DIM = 392;

export type FlattenInput = {
  hands: { x: number; y: number; z?: number }[][];
  pose: { x: number; y: number; z?: number }[] | undefined;
  face: { x: number; y: number; z?: number }[] | undefined;
};

function appendPoints(
  coords: number[],
  points: { x: number; y: number; z?: number }[] | undefined,
  maxPoints: number,
): void {
  if (!points?.length) {
    return;
  }
  for (const pt of points.slice(0, maxPoints)) {
    coords.push(pt.x, pt.y, pt.z ?? 0);
  }
}

export function flattenLandmarkFrame(input: FlattenInput): number[] {
  const coords: number[] = [];
  for (const hand of input.hands.slice(0, 2)) {
    appendPoints(coords, hand, 21);
  }
  appendPoints(coords, input.pose, 33);
  appendPoints(coords, input.face, 40);

  if (coords.length < LANDMARK_FEATURE_DIM) {
    coords.push(...Array(LANDMARK_FEATURE_DIM - coords.length).fill(0));
  }
  return coords.slice(0, LANDMARK_FEATURE_DIM);
}
