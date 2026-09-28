/** Flatten MediaPipe Tasks detections to Holistic layout (543×3) for backend packing. */

export const HOLISTIC_LANDMARK_COUNT = 543;
export const HOLISTIC_FLAT_DIM = HOLISTIC_LANDMARK_COUNT * 3;

/** @deprecated Use HOLISTIC_FLAT_DIM; backend accepts legacy 392-byte layout too. */
export const LANDMARK_FEATURE_DIM = 392;

export type FlattenInput = {
  hands: { x: number; y: number; z?: number }[][];
  handLabels?: (string | undefined)[];
  pose: { x: number; y: number; z?: number }[] | undefined;
  face: { x: number; y: number; z?: number }[] | undefined;
};

const POSE_COUNT = 33;
const LEFT_HAND_START = 501;
const RIGHT_HAND_START = 522;

function writePoint(
  out: number[],
  holisticIndex: number,
  pt: { x: number; y: number; z?: number } | undefined,
): void {
  if (!pt || holisticIndex < 0 || holisticIndex >= HOLISTIC_LANDMARK_COUNT) {
    return;
  }
  const base = holisticIndex * 3;
  out[base] = pt.x;
  out[base + 1] = pt.y;
  out[base + 2] = pt.z ?? 0;
}

function handSide(label: string | undefined): "left" | "right" | null {
  if (!label) {
    return null;
  }
  const text = label.toLowerCase();
  if (text.includes("left")) {
    return "left";
  }
  if (text.includes("right")) {
    return "right";
  }
  return null;
}

export function flattenLandmarkFrame(input: FlattenInput): number[] {
  const out = Array<number>(HOLISTIC_FLAT_DIM).fill(0);

  input.pose?.slice(0, POSE_COUNT).forEach((pt, i) => {
    writePoint(out, i, pt);
  });

  input.face?.forEach((pt, i) => {
    writePoint(out, POSE_COUNT + i, pt);
  });

  const hands = input.hands.slice(0, 2);
  const labels = input.handLabels ?? [];
  let leftHand: typeof hands[number] | undefined;
  let rightHand: typeof hands[number] | undefined;
  const unlabeled: typeof hands = [];

  hands.forEach((hand, idx) => {
    const side = handSide(labels[idx]);
    if (side === "left") {
      leftHand = hand;
    } else if (side === "right") {
      rightHand = hand;
    } else {
      unlabeled.push(hand);
    }
  });

  for (const hand of unlabeled) {
    if (!leftHand) {
      leftHand = hand;
    } else if (!rightHand) {
      rightHand = hand;
    }
  }

  leftHand?.slice(0, 21).forEach((pt, i) => {
    writePoint(out, LEFT_HAND_START + i, pt);
  });
  rightHand?.slice(0, 21).forEach((pt, i) => {
    writePoint(out, RIGHT_HAND_START + i, pt);
  });

  return out;
}
