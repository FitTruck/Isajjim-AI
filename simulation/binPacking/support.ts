/**
 * 70% 지지 규칙 검사 로직
 *
 * 공중 배치 방지를 위한 지지 검사
 * - 바닥 배치(y=0)는 항상 지지됨
 * - 그 외에는 아래 박스들의 윗면과 겹치는 영역이 70% 이상이어야 함
 */

import { PlacedBox } from './types';
import { DEFAULT_SUPPORT_RATIO, FLOOR_THRESHOLD, TOLERANCE } from './constants';

/**
 * X-Z 평면에서 두 영역의 겹침 면적 계산
 *
 * @param x1 - 첫 번째 박스 중심 X
 * @param z1 - 첫 번째 박스 중심 Z
 * @param w1 - 첫 번째 박스 너비
 * @param d1 - 첫 번째 박스 깊이
 * @param box - 비교할 배치된 박스
 * @returns 겹침 면적 (cm²)
 */
export function calculate2DOverlap(
  x1: number,
  z1: number,
  w1: number,
  d1: number,
  box: PlacedBox
): number {
  const xMin1 = x1 - w1 / 2;
  const xMax1 = x1 + w1 / 2;
  const zMin1 = z1 - d1 / 2;
  const zMax1 = z1 + d1 / 2;

  const xMin2 = box.x - box.width / 2;
  const xMax2 = box.x + box.width / 2;
  const zMin2 = box.z - box.depth / 2;
  const zMax2 = box.z + box.depth / 2;

  const xOverlap = Math.max(0, Math.min(xMax1, xMax2) - Math.max(xMin1, xMin2));
  const zOverlap = Math.max(0, Math.min(zMax1, zMax2) - Math.max(zMin1, zMin2));

  return xOverlap * zOverlap;
}

/**
 * 70% 지지 규칙 검사
 *
 * @param x - 배치할 위치 중심 X
 * @param y - 배치할 위치 바닥 Y
 * @param z - 배치할 위치 중심 Z
 * @param w - 박스 너비
 * @param d - 박스 깊이
 * @param placedBoxes - 이미 배치된 박스들
 * @param minRatio - 최소 지지 비율 (기본 0.7)
 * @returns 지지 조건 충족 여부
 */
export function checkSupport(
  x: number,
  y: number,
  z: number,
  w: number,
  d: number,
  placedBoxes: PlacedBox[],
  minRatio: number = DEFAULT_SUPPORT_RATIO
): boolean {
  // 바닥 배치는 항상 지지됨
  if (y < FLOOR_THRESHOLD) {
    return true;
  }

  const baseArea = w * d;
  if (baseArea <= 0) {
    return false;
  }

  let supportedArea = 0;

  for (const box of placedBoxes) {
    // 아래 박스의 윗면이 현재 위치와 맞닿는지 확인
    const boxTopY = box.y + box.height;
    if (Math.abs(boxTopY - y) < TOLERANCE) {
      supportedArea += calculate2DOverlap(x, z, w, d, box);
    }
  }

  const supportRatio = supportedArea / baseArea;
  return supportRatio >= minRatio;
}

/**
 * 충돌 검사 (AABB)
 *
 * @param x - 새 박스 중심 X
 * @param y - 새 박스 바닥 Y
 * @param z - 새 박스 중심 Z
 * @param w - 새 박스 너비
 * @param d - 새 박스 깊이
 * @param h - 새 박스 높이
 * @param placedBoxes - 이미 배치된 박스들
 * @returns 충돌 여부 (true = 충돌)
 */
export function checkOverlap(
  x: number,
  y: number,
  z: number,
  w: number,
  d: number,
  h: number,
  placedBoxes: PlacedBox[]
): boolean {
  // 새 박스 경계 (tolerance 만큼 축소하여 경계 접촉 허용)
  const newXMin = x - w / 2 + TOLERANCE;
  const newXMax = x + w / 2 - TOLERANCE;
  const newYMin = y + TOLERANCE;
  const newYMax = y + h - TOLERANCE;
  const newZMin = z - d / 2 + TOLERANCE;
  const newZMax = z + d / 2 - TOLERANCE;

  for (const box of placedBoxes) {
    // 기존 박스 경계 (tolerance 만큼 축소)
    const boxXMin = box.x - box.width / 2 + TOLERANCE;
    const boxXMax = box.x + box.width / 2 - TOLERANCE;
    const boxYMin = box.y + TOLERANCE;
    const boxYMax = box.y + box.height - TOLERANCE;
    const boxZMin = box.z - box.depth / 2 + TOLERANCE;
    const boxZMax = box.z + box.depth / 2 - TOLERANCE;

    // AABB 충돌 검사
    if (
      newXMin < boxXMax &&
      newXMax > boxXMin &&
      newYMin < boxYMax &&
      newYMax > boxYMin &&
      newZMin < boxZMax &&
      newZMax > boxZMin
    ) {
      return true;
    }
  }

  return false;
}

/**
 * 경계 내 배치 가능 여부 검사
 *
 * @param x - 배치할 위치 중심 X
 * @param y - 배치할 위치 바닥 Y
 * @param z - 배치할 위치 중심 Z
 * @param w - 박스 너비
 * @param d - 박스 깊이
 * @param h - 박스 높이
 * @param truckW - 트럭 너비
 * @param truckD - 트럭 깊이
 * @param truckH - 트럭 높이
 * @returns 경계 내 여부
 */
export function checkBoundary(
  x: number,
  y: number,
  z: number,
  w: number,
  d: number,
  h: number,
  truckW: number,
  truckD: number,
  truckH: number
): boolean {
  const halfW = w / 2;
  const halfD = d / 2;

  // X축 경계 (좌우)
  if (x - halfW < -truckW / 2) return false;
  if (x + halfW > truckW / 2) return false;

  // Z축 경계 (앞뒤)
  if (z - halfD < -truckD / 2) return false;
  if (z + halfD > truckD / 2) return false;

  // Y축 경계 (바닥, 천장)
  if (y < 0) return false;
  if (y + h > truckH) return false;

  return true;
}
