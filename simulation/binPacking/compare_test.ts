/**
 * Python-TypeScript 비교 테스트
 *
 * 동일한 입력 데이터로 Python과 TypeScript의 결과를 비교
 * 결과를 JSON으로 출력하여 Python 스크립트에서 비교
 */

import { optimizeOBB, optimizeOBBMulti, OBBItem } from './index';

// 테스트 케이스 1: 기본 가구 셋
const testCase1: OBBItem[] = [
  { id: 'bed', width: 200, depth: 180, height: 40 },
  { id: 'wardrobe', width: 150, depth: 60, height: 200 },
  { id: 'desk', width: 120, depth: 60, height: 75 },
  { id: 'chair1', width: 50, depth: 50, height: 80 },
  { id: 'chair2', width: 50, depth: 50, height: 80 },
  { id: 'nightstand', width: 50, depth: 40, height: 55 },
];

// 테스트 케이스 2: 큰 가구 위주
const testCase2: OBBItem[] = [
  { id: 'sofa_3seat', width: 250, depth: 100, height: 85 },
  { id: 'dining_table', width: 180, depth: 90, height: 75 },
  { id: 'bookshelf', width: 120, depth: 40, height: 200 },
  { id: 'tv_stand', width: 180, depth: 45, height: 50 },
];

// 테스트 케이스 3: 많은 작은 가구
const testCase3: OBBItem[] = [];
for (let i = 0; i < 10; i++) {
  testCase3.push({ id: `box_${i}`, width: 60, depth: 40, height: 50 });
}

// 실행 및 결과 출력
function runTest(name: string, items: OBBItem[]) {
  const singleResult = optimizeOBB(items);
  const multiResult = optimizeOBBMulti(items);

  return {
    name,
    items: items.map((i) => ({
      id: i.id,
      width: i.width,
      depth: i.depth,
      height: i.height,
    })),
    singleTruck: {
      truckType: singleResult.truckType,
      success: singleResult.success,
      volumeUtilization: singleResult.volumeUtilization,
      placedCount: singleResult.placedItems.length,
      unplacedCount: singleResult.unplacedItems.length,
      placements: singleResult.placedItems.map((p) => ({
        itemId: p.itemId,
        x: Math.round(p.x * 100) / 100,
        y: Math.round(p.y * 100) / 100,
        z: Math.round(p.z * 100) / 100,
        width: p.width,
        depth: p.depth,
        height: p.height,
        orientation: p.orientation,
      })),
    },
    multiTruck: {
      totalTrucks: multiResult.totalTrucks,
      success: multiResult.success,
      trucks: multiResult.trucks.map((t) => ({
        type: t.type,
        utilization: t.utilization,
        placedCount: t.placements.length,
      })),
    },
  };
}

const results = {
  testCase1: runTest('기본 가구 셋', testCase1),
  testCase2: runTest('큰 가구 위주', testCase2),
  testCase3: runTest('많은 작은 가구', testCase3),
};

console.log(JSON.stringify(results, null, 2));
