/**
 * OBB Packer 메인 클래스
 *
 * Extreme Points (EP) 알고리즘 기반 3D Bin Packing
 * - Python obb_packer.py의 extreme_points_pack 로직을 TypeScript로 변환
 * - 70% 지지 규칙 적용
 * - 수평 회전만 허용 (LWH, WLH)
 * - Corner-first 배치 지원
 */

import {
  OBBItem,
  PlacedBox,
  PackingResult,
  ExtremePoint,
  TruckPreset,
  Orientation,
  PackingOptions,
} from './types';
import { TRUCK_PRESETS, TRUCK_ORDER, DEFAULT_SUPPORT_RATIO, TOLERANCE } from './constants';
import { checkSupport, checkOverlap, checkBoundary } from './support';

/**
 * 회전 적용된 치수 반환
 */
function getRotatedDims(
  width: number,
  depth: number,
  height: number,
  orientation: Orientation
): [number, number, number] {
  if (orientation === Orientation.WLH) {
    // 90도 수평 회전: width와 depth 교환
    return [depth, width, height];
  }
  // 기본 방향
  return [width, depth, height];
}

/**
 * 부피 계산
 */
function calculateVolume(item: OBBItem): number {
  return item.width * item.depth * item.height;
}

/**
 * EP 정렬 비교 함수 (Z → X → Y 우선순위)
 * 뒤쪽(Z 작은 값)부터, 왼쪽부터 오른쪽으로, 마지막으로 위로 쌓기
 */
function compareExtremePoints(a: ExtremePoint, b: ExtremePoint): number {
  if (a.z !== b.z) return a.z - b.z;
  if (a.x !== b.x) return a.x - b.x;
  return a.y - b.y;
}

/**
 * EP가 다른 박스 내부에 있는지 확인 (경계는 유효)
 */
function isEPValid(ep: ExtremePoint, placed: PlacedBox[]): boolean {
  for (const box of placed) {
    const boxXMin = box.x - box.width / 2 + TOLERANCE;
    const boxXMax = box.x + box.width / 2 - TOLERANCE;
    const boxYMin = box.y + TOLERANCE;
    const boxYMax = box.y + box.height - TOLERANCE;
    const boxZMin = box.z - box.depth / 2 + TOLERANCE;
    const boxZMax = box.z + box.depth / 2 - TOLERANCE;

    // EP가 박스 내부에 완전히 들어가 있는 경우만 무효
    if (
      ep.x > boxXMin &&
      ep.x < boxXMax &&
      ep.y > boxYMin &&
      ep.y < boxYMax &&
      ep.z > boxZMin &&
      ep.z < boxZMax
    ) {
      return false;
    }
  }
  return true;
}

/**
 * 배치 후 새 EP 생성
 * - EP1: +X 방향 (오른쪽)
 * - EP2: +Z 방향 (앞쪽)
 * - EP3: +Y 방향 (위쪽)
 */
function generateNewExtremePoints(
  box: PlacedBox,
  truckW: number,
  truckD: number,
  truckH: number
): ExtremePoint[] {
  const newEPs: ExtremePoint[] = [];

  // EP1: 오른쪽 (+X)
  const epX = box.x + box.width / 2;
  if (epX < truckW / 2) {
    newEPs.push({
      x: epX,
      y: box.y,
      z: box.z - box.depth / 2,
    });
  }

  // EP2: 앞쪽 (+Z)
  const epZ = box.z + box.depth / 2;
  if (epZ < truckD / 2) {
    newEPs.push({
      x: box.x - box.width / 2,
      y: box.y,
      z: epZ,
    });
  }

  // EP3: 위쪽 (+Y)
  const epY = box.y + box.height;
  if (epY < truckH) {
    newEPs.push({
      x: box.x - box.width / 2,
      y: epY,
      z: box.z - box.depth / 2,
    });
  }

  return newEPs;
}

/**
 * 뒤쪽-왼쪽 코너 배치 시도
 */
function tryCornerPlacement(
  item: OBBItem,
  placed: PlacedBox[],
  truckW: number,
  truckD: number,
  truckH: number
): {
  orientation: Orientation;
  dims: [number, number, number];
  position: [number, number, number];
} | null {
  const orientations = [Orientation.LWH, Orientation.WLH];

  for (const orientation of orientations) {
    const [w, d, h] = getRotatedDims(item.width, item.depth, item.height, orientation);

    // 뒤쪽-왼쪽 코너 위치 계산
    const cx = -truckW / 2 + w / 2;
    const cy = 0;
    const cz = -truckD / 2 + d / 2;

    // 경계 검사
    if (!checkBoundary(cx, cy, cz, w, d, h, truckW, truckD, truckH)) {
      continue;
    }

    // 충돌 검사
    if (checkOverlap(cx, cy, cz, w, d, h, placed)) {
      continue;
    }

    // 자연스러운 방향 우선 (depth >= width)
    if (d >= w) {
      return {
        orientation,
        dims: [w, d, h],
        position: [cx, cy, cz],
      };
    }
  }

  // 자연스러운 방향이 없으면 첫 번째 유효한 배치 반환
  for (const orientation of orientations) {
    const [w, d, h] = getRotatedDims(item.width, item.depth, item.height, orientation);
    const cx = -truckW / 2 + w / 2;
    const cy = 0;
    const cz = -truckD / 2 + d / 2;

    if (checkBoundary(cx, cy, cz, w, d, h, truckW, truckD, truckH)) {
      if (!checkOverlap(cx, cy, cz, w, d, h, placed)) {
        return {
          orientation,
          dims: [w, d, h],
          position: [cx, cy, cz],
        };
      }
    }
  }

  return null;
}

/**
 * 최적 배치 위치 찾기
 */
function findBestPlacement(
  item: OBBItem,
  extremePoints: ExtremePoint[],
  placed: PlacedBox[],
  truckW: number,
  truckD: number,
  truckH: number,
  supportRatio: number
): {
  ep: ExtremePoint;
  orientation: Orientation;
  dims: [number, number, number];
} | null {
  let bestPlacement: {
    ep: ExtremePoint;
    orientation: Orientation;
    dims: [number, number, number];
  } | null = null;
  let bestScore: [number, number, number, number] = [Infinity, Infinity, Infinity, 1];

  const orientations = [Orientation.LWH, Orientation.WLH];

  for (const ep of extremePoints) {
    for (const orientation of orientations) {
      const [w, d, h] = getRotatedDims(item.width, item.depth, item.height, orientation);

      // EP에서 중심 좌표 계산
      const cx = ep.x + w / 2;
      const cy = ep.y;
      const cz = ep.z + d / 2;

      // 1. 경계 검사
      if (!checkBoundary(cx, cy, cz, w, d, h, truckW, truckD, truckH)) {
        continue;
      }

      // 2. 충돌 검사
      if (checkOverlap(cx, cy, cz, w, d, h, placed)) {
        continue;
      }

      // 3. 지지 검사 (70% 규칙)
      if (!checkSupport(cx, cy, cz, w, d, placed, supportRatio)) {
        continue;
      }

      // 점수 계산 (Z → X → Y 우선순위, 자연스러운 방향 마지막)
      const natural = d >= w ? 0 : 1;
      const score: [number, number, number, number] = [cz, cx, cy, natural];

      // 점수 비교 (낮을수록 좋음)
      if (
        score[0] < bestScore[0] ||
        (score[0] === bestScore[0] && score[1] < bestScore[1]) ||
        (score[0] === bestScore[0] && score[1] === bestScore[1] && score[2] < bestScore[2]) ||
        (score[0] === bestScore[0] &&
          score[1] === bestScore[1] &&
          score[2] === bestScore[2] &&
          score[3] < bestScore[3])
      ) {
        bestScore = score;
        bestPlacement = { ep, orientation, dims: [w, d, h] };
      }
    }
  }

  return bestPlacement;
}

/**
 * Extreme Points 알고리즘으로 3D Bin Packing 수행
 */
export function extremePointsPack(
  items: OBBItem[],
  truckDims: TruckPreset,
  options: PackingOptions = {}
): PackingResult {
  const { supportRatio = DEFAULT_SUPPORT_RATIO, cornerFirst = true } = options;

  const truckW = truckDims.width;
  const truckD = truckDims.depth;
  const truckH = truckDims.height;
  const truckVolume = truckW * truckD * truckH;

  // 부피 순 정렬 (큰 것 먼저)
  const sortedItems = [...items].sort((a, b) => calculateVolume(b) - calculateVolume(a));

  // 초기 EP: 뒤쪽-왼쪽-바닥 코너
  let extremePoints: ExtremePoint[] = [{ x: -truckW / 2, y: 0, z: -truckD / 2 }];

  const placed: PlacedBox[] = [];
  const unplaced: string[] = [];
  let placedVolume = 0;
  let cornerPhaseDone = !cornerFirst;

  for (const item of sortedItems) {
    let box: PlacedBox | null = null;

    // Phase 1: Corner-first 배치
    if (!cornerPhaseDone) {
      const cornerResult = tryCornerPlacement(item, placed, truckW, truckD, truckH);

      if (cornerResult) {
        const { orientation, dims, position } = cornerResult;
        const [w, d, h] = dims;
        const [cx, cy, cz] = position;

        box = {
          itemId: item.id,
          x: cx,
          y: cy,
          z: cz,
          width: w,
          depth: d,
          height: h,
          orientation,
        };

        placed.push(box);
        placedVolume += w * d * h;

        // 새 EP 생성
        const newEPs = generateNewExtremePoints(box, truckW, truckD, truckH);
        extremePoints.push(...newEPs);
        extremePoints = extremePoints.filter((ep) => isEPValid(ep, placed));
        extremePoints.sort(compareExtremePoints);

        // 코너 단계 완료
        cornerPhaseDone = true;
        continue;
      }
    }

    // Phase 2: EP 기반 일반 배치
    const placement = findBestPlacement(
      item,
      extremePoints,
      placed,
      truckW,
      truckD,
      truckH,
      supportRatio
    );

    if (!placement) {
      unplaced.push(item.id);
      continue;
    }

    const { ep, orientation, dims } = placement;
    const [w, d, h] = dims;
    const cx = ep.x + w / 2;
    const cy = ep.y;
    const cz = ep.z + d / 2;

    box = {
      itemId: item.id,
      x: cx,
      y: cy,
      z: cz,
      width: w,
      depth: d,
      height: h,
      orientation,
    };

    placed.push(box);
    placedVolume += w * d * h;

    // 새 EP 생성
    const newEPs = generateNewExtremePoints(box, truckW, truckD, truckH);
    extremePoints.push(...newEPs);

    // 무효 EP 제거
    extremePoints = extremePoints.filter((ep) => isEPValid(ep, placed));

    // EP 정렬 (Z → X → Y)
    extremePoints.sort(compareExtremePoints);
  }

  const volumeUtilization = truckVolume > 0 ? (placedVolume / truckVolume) * 100 : 0;

  return {
    success: unplaced.length === 0,
    truckType: '',
    placedItems: placed,
    unplacedItems: unplaced,
    volumeUtilization: Math.round(volumeUtilization * 10) / 10,
    message: `${placed.length}개 배치, ${unplaced.length}개 미배치`,
  };
}

/**
 * 모든 아이템을 배치할 수 있는 가장 작은 트럭 선택
 */
export function selectSmallestFittingTruck(
  items: OBBItem[],
  options: PackingOptions = {}
): { truckType: string; result: PackingResult } {
  for (const truckType of TRUCK_ORDER) {
    const truckDims = TRUCK_PRESETS[truckType];
    const result = extremePointsPack(items, truckDims, options);
    result.truckType = truckType;

    if (result.success) {
      return { truckType, result };
    }
  }

  // 모든 트럭에서 실패 → 최대 트럭 결과 반환
  const result = extremePointsPack(items, TRUCK_PRESETS['5ton'], options);
  result.truckType = '5ton';

  return { truckType: '5ton', result };
}

/**
 * 단일 트럭 또는 자동 선택 패킹
 */
export function optimizeOBB(
  items: OBBItem[],
  truckType?: string,
  options: PackingOptions = {}
): PackingResult {
  if (truckType && TRUCK_PRESETS[truckType]) {
    const result = extremePointsPack(items, TRUCK_PRESETS[truckType], options);
    result.truckType = truckType;
    return result;
  }

  const { result } = selectSmallestFittingTruck(items, options);
  return result;
}
