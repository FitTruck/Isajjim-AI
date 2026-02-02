/**
 * 3D Bin Packing 모듈
 *
 * React Native용 3D 적재 최적화 라이브러리
 * Python obb_packer.py의 Extreme Points 알고리즘을 TypeScript로 변환
 *
 * @example
 * ```typescript
 * import { optimizeOBB, optimizeOBBMulti, TRUCK_PRESETS } from './binPacking';
 *
 * // 단일 트럭 자동 선택
 * const items = [
 *   { id: 'sofa', width: 200, depth: 90, height: 85 },
 *   { id: 'table', width: 120, depth: 80, height: 75 },
 * ];
 * const result = optimizeOBB(items);
 *
 * // 특정 트럭 지정
 * const result2 = optimizeOBB(items, '2.5ton');
 *
 * // 멀티 트럭 (모든 아이템 배치)
 * const multiResult = optimizeOBBMulti(items);
 * ```
 */

// Types
export {
  TruckPreset,
  OBBItem,
  PlacedBox,
  ExtremePoint,
  PackingResult,
  TruckPlacement,
  MultiTruckResult,
  PackingOptions,
  Orientation,
} from './types';

// Constants
export { TRUCK_PRESETS, TRUCK_ORDER, DEFAULT_SUPPORT_RATIO, TOLERANCE } from './constants';

// Support functions
export { checkSupport, checkOverlap, checkBoundary, calculate2DOverlap } from './support';

// Packer functions
export { extremePointsPack, selectSmallestFittingTruck, optimizeOBB } from './packer';

// Multi-truck functions
export { selectTrucksForAllItems, optimizeOBBMulti } from './multiTruck';
