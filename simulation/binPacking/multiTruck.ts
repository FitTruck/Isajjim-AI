/**
 * 멀티 트럭 선택 로직
 *
 * 모든 아이템을 배치할 수 있는 최소 트럭 조합 선택
 * - 단일 트럭 시도: 1ton → 2.5ton → 5ton
 * - 5ton에도 안 들어가면 조합: 5ton + 1ton → 5ton + 2.5ton → 5ton + 5ton
 * - 미배치가 없을 때까지 반복
 */

import { OBBItem, MultiTruckResult, TruckPlacement, PackingOptions } from './types';
import { TRUCK_PRESETS, TRUCK_ORDER, DEFAULT_SUPPORT_RATIO } from './constants';
import { extremePointsPack } from './packer';

/**
 * 모든 아이템을 배치할 수 있는 최소 트럭 조합 선택
 */
export function selectTrucksForAllItems(
  items: OBBItem[],
  options: PackingOptions = {}
): MultiTruckResult {
  const { supportRatio = DEFAULT_SUPPORT_RATIO, cornerFirst = true } = options;

  // 1. 단일 트럭 시도
  for (const truckType of TRUCK_ORDER) {
    const truckDims = TRUCK_PRESETS[truckType];
    const result = extremePointsPack(items, truckDims, { supportRatio, cornerFirst });

    if (result.success) {
      const truckVolume = truckDims.width * truckDims.depth * truckDims.height;
      const placedVolume = result.placedItems.reduce(
        (sum, p) => sum + p.width * p.depth * p.height,
        0
      );
      const utilization = truckVolume > 0 ? (placedVolume / truckVolume) * 100 : 0;

      return {
        success: true,
        trucks: [
          {
            type: truckType,
            placements: result.placedItems,
            utilization: Math.round(utilization * 10) / 10,
          },
        ],
        totalTrucks: 1,
        unplacedItems: [],
        message: `1대 트럭 사용: ${truckType}`,
      };
    }
  }

  // 2. 5ton에도 안 들어가면 멀티 트럭 조합
  const trucksResult: TruckPlacement[] = [];
  let remainingItems = [...items];

  // 첫 번째 트럭: 5ton
  const firstTruckDims = TRUCK_PRESETS['5ton'];
  const firstResult = extremePointsPack(remainingItems, firstTruckDims, { supportRatio, cornerFirst });

  const firstVolume = firstTruckDims.width * firstTruckDims.depth * firstTruckDims.height;
  const firstPlacedVolume = firstResult.placedItems.reduce(
    (sum, p) => sum + p.width * p.depth * p.height,
    0
  );
  const firstUtilization = firstVolume > 0 ? (firstPlacedVolume / firstVolume) * 100 : 0;

  trucksResult.push({
    type: '5ton',
    placements: firstResult.placedItems,
    utilization: Math.round(firstUtilization * 10) / 10,
  });

  // 배치된 아이템 제거
  const placedIds = new Set(firstResult.placedItems.map((p) => p.itemId));
  remainingItems = remainingItems.filter((item) => !placedIds.has(item.id));

  // 3. 미배치 아이템이 있으면 추가 트럭 할당
  while (remainingItems.length > 0) {
    let placedAny = false;

    for (const addTruckType of TRUCK_ORDER) {
      if (remainingItems.length === 0) break;

      const addTruckDims = TRUCK_PRESETS[addTruckType];
      const addResult = extremePointsPack(remainingItems, addTruckDims, { supportRatio, cornerFirst });

      if (addResult.placedItems.length > 0) {
        const addVolume = addTruckDims.width * addTruckDims.depth * addTruckDims.height;
        const addPlacedVolume = addResult.placedItems.reduce(
          (sum, p) => sum + p.width * p.depth * p.height,
          0
        );
        const addUtilization = addVolume > 0 ? (addPlacedVolume / addVolume) * 100 : 0;

        trucksResult.push({
          type: addTruckType,
          placements: addResult.placedItems,
          utilization: Math.round(addUtilization * 10) / 10,
        });

        // 배치된 아이템 제거
        const newPlacedIds = new Set(addResult.placedItems.map((p) => p.itemId));
        remainingItems = remainingItems.filter((item) => !newPlacedIds.has(item.id));
        placedAny = true;

        if (addResult.success) {
          // 모든 remaining이 배치됨
          break;
        }
      }
    }

    if (!placedAny) {
      // 더 이상 배치할 수 없음
      break;
    }
  }

  // 결과 메시지 생성
  const truckNames = trucksResult.map((t) => t.type);
  const message = `${trucksResult.length}대 트럭 사용: ${truckNames.join(' + ')}`;

  const unplacedIds = remainingItems.map((item) => item.id);

  return {
    success: unplacedIds.length === 0,
    trucks: trucksResult,
    totalTrucks: trucksResult.length,
    unplacedItems: unplacedIds,
    message,
  };
}

/**
 * 멀티 트럭 OBB 최적화 메인 함수
 *
 * 자동으로 최소 트럭 조합을 선택하여 모든 아이템을 배치합니다.
 */
export function optimizeOBBMulti(items: OBBItem[], options: PackingOptions = {}): MultiTruckResult {
  return selectTrucksForAllItems(items, options);
}
