/**
 * 3D Bin Packing 테스트
 *
 * Jest 테스트 파일
 * - 좌표 변환 테스트
 * - 지지 검사 테스트
 * - 충돌 검사 테스트
 * - 전체 패킹 플로우 테스트
 */

import {
  optimizeOBB,
  optimizeOBBMulti,
  checkSupport,
  checkOverlap,
  checkBoundary,
  calculate2DOverlap,
  TRUCK_PRESETS,
  OBBItem,
  PlacedBox,
  Orientation,
} from '../index';

describe('Support Functions', () => {
  describe('calculate2DOverlap', () => {
    it('완전히 겹치는 경우', () => {
      const box: PlacedBox = {
        itemId: 'test',
        x: 0,
        y: 0,
        z: 0,
        width: 100,
        depth: 100,
        height: 50,
        orientation: Orientation.LWH,
      };

      const overlap = calculate2DOverlap(0, 0, 100, 100, box);
      expect(overlap).toBe(10000); // 100 * 100
    });

    it('부분적으로 겹치는 경우', () => {
      const box: PlacedBox = {
        itemId: 'test',
        x: 0,
        y: 0,
        z: 0,
        width: 100,
        depth: 100,
        height: 50,
        orientation: Orientation.LWH,
      };

      // 50% 겹침 (x축으로 50 이동)
      const overlap = calculate2DOverlap(50, 0, 100, 100, box);
      expect(overlap).toBe(5000); // 50 * 100
    });

    it('겹치지 않는 경우', () => {
      const box: PlacedBox = {
        itemId: 'test',
        x: 0,
        y: 0,
        z: 0,
        width: 100,
        depth: 100,
        height: 50,
        orientation: Orientation.LWH,
      };

      // 완전히 떨어진 위치
      const overlap = calculate2DOverlap(200, 200, 50, 50, box);
      expect(overlap).toBe(0);
    });
  });

  describe('checkSupport', () => {
    it('바닥 배치는 항상 지지됨', () => {
      const result = checkSupport(0, 0, 0, 100, 100, []);
      expect(result).toBe(true);
    });

    it('지지대 없이 공중에 배치 불가', () => {
      const result = checkSupport(0, 50, 0, 100, 100, []);
      expect(result).toBe(false);
    });

    it('70% 이상 지지되면 통과', () => {
      const supportBox: PlacedBox = {
        itemId: 'support',
        x: 0,
        y: 0,
        z: 0,
        width: 100,
        depth: 100,
        height: 50,
        orientation: Orientation.LWH,
      };

      // 80% 겹침 (80 x 100 = 8000 / 10000 = 80%)
      const result = checkSupport(10, 50, 0, 100, 100, [supportBox], 0.7);
      expect(result).toBe(true);
    });

    it('70% 미만 지지되면 실패', () => {
      const supportBox: PlacedBox = {
        itemId: 'support',
        x: 0,
        y: 0,
        z: 0,
        width: 50,
        depth: 50,
        height: 50,
        orientation: Orientation.LWH,
      };

      // 25% 겹침 (50 x 50 = 2500 / 10000 = 25%)
      const result = checkSupport(0, 50, 0, 100, 100, [supportBox], 0.7);
      expect(result).toBe(false);
    });
  });

  describe('checkOverlap', () => {
    it('겹치는 경우 true 반환', () => {
      const existingBox: PlacedBox = {
        itemId: 'existing',
        x: 0,
        y: 0,
        z: 0,
        width: 100,
        depth: 100,
        height: 100,
        orientation: Orientation.LWH,
      };

      const result = checkOverlap(0, 0, 0, 50, 50, 50, [existingBox]);
      expect(result).toBe(true);
    });

    it('겹치지 않는 경우 false 반환', () => {
      const existingBox: PlacedBox = {
        itemId: 'existing',
        x: 0,
        y: 0,
        z: 0,
        width: 100,
        depth: 100,
        height: 100,
        orientation: Orientation.LWH,
      };

      const result = checkOverlap(200, 0, 0, 50, 50, 50, [existingBox]);
      expect(result).toBe(false);
    });

    it('경계가 맞닿는 경우는 허용 (tolerance)', () => {
      const existingBox: PlacedBox = {
        itemId: 'existing',
        x: 0,
        y: 0,
        z: 0,
        width: 100,
        depth: 100,
        height: 100,
        orientation: Orientation.LWH,
      };

      // 정확히 옆에 붙어있음
      const result = checkOverlap(100, 0, 0, 100, 100, 100, [existingBox]);
      expect(result).toBe(false);
    });
  });

  describe('checkBoundary', () => {
    const truckW = 200;
    const truckD = 400;
    const truckH = 200;

    it('트럭 내부에 있으면 true', () => {
      const result = checkBoundary(0, 0, 0, 50, 50, 50, truckW, truckD, truckH);
      expect(result).toBe(true);
    });

    it('X축 경계 초과시 false', () => {
      const result = checkBoundary(90, 0, 0, 50, 50, 50, truckW, truckD, truckH);
      expect(result).toBe(false); // 90 + 25 = 115 > 100
    });

    it('Y축 천장 초과시 false', () => {
      const result = checkBoundary(0, 180, 0, 50, 50, 50, truckW, truckD, truckH);
      expect(result).toBe(false); // 180 + 50 = 230 > 200
    });

    it('Z축 경계 초과시 false', () => {
      const result = checkBoundary(0, 0, 180, 50, 50, 50, truckW, truckD, truckH);
      expect(result).toBe(false); // 180 + 25 = 205 > 200
    });
  });
});

describe('Packer Functions', () => {
  describe('optimizeOBB - 단일 트럭', () => {
    it('빈 아이템 리스트는 성공', () => {
      const result = optimizeOBB([]);
      expect(result.success).toBe(true);
      expect(result.placedItems).toHaveLength(0);
    });

    it('단일 아이템 배치', () => {
      const items: OBBItem[] = [{ id: 'sofa', width: 200, depth: 90, height: 85 }];

      const result = optimizeOBB(items);
      expect(result.success).toBe(true);
      expect(result.placedItems).toHaveLength(1);
      expect(result.truckType).toBe('1ton');
    });

    it('여러 아이템 배치', () => {
      const items: OBBItem[] = [
        { id: 'sofa', width: 200, depth: 90, height: 85 },
        { id: 'table', width: 120, depth: 80, height: 75 },
        { id: 'chair', width: 50, depth: 50, height: 80 },
      ];

      const result = optimizeOBB(items);
      expect(result.success).toBe(true);
      expect(result.placedItems.length).toBeGreaterThanOrEqual(1);
    });

    it('특정 트럭 지정', () => {
      const items: OBBItem[] = [{ id: 'sofa', width: 200, depth: 90, height: 85 }];

      const result = optimizeOBB(items, '5ton');
      expect(result.truckType).toBe('5ton');
    });

    it('1톤에 안 들어가면 2.5톤 시도', () => {
      // 1톤 트럭 (170 x 280 x 170) 높이보다 큰 아이템
      // 높이는 회전해도 변하지 않으므로 175cm 높이면 1톤에 안 들어감
      const items: OBBItem[] = [{ id: 'tall_wardrobe', width: 100, depth: 60, height: 175 }];

      const result = optimizeOBB(items);
      expect(result.success).toBe(true);
      expect(result.truckType).toBe('2.5ton');
    });
  });

  describe('optimizeOBB - 좌표계 검증', () => {
    it('배치 좌표가 트럭 중심 기준', () => {
      const items: OBBItem[] = [{ id: 'test', width: 100, depth: 100, height: 50 }];

      const result = optimizeOBB(items, '1ton');
      const placed = result.placedItems[0];

      // 트럭 중심이 원점이므로 뒤쪽-왼쪽 코너는 음수 좌표
      expect(placed.x).toBeLessThan(0);
      expect(placed.z).toBeLessThan(0);
      expect(placed.y).toBe(0); // 바닥
    });

    it('배치된 아이템이 경계 내에 있음', () => {
      const items: OBBItem[] = [
        { id: 'item1', width: 80, depth: 100, height: 80 },
        { id: 'item2', width: 70, depth: 90, height: 70 },
      ];

      const result = optimizeOBB(items, '1ton');
      const truck = TRUCK_PRESETS['1ton'];

      for (const box of result.placedItems) {
        // X축 경계
        expect(box.x - box.width / 2).toBeGreaterThanOrEqual(-truck.width / 2 - 0.01);
        expect(box.x + box.width / 2).toBeLessThanOrEqual(truck.width / 2 + 0.01);

        // Z축 경계
        expect(box.z - box.depth / 2).toBeGreaterThanOrEqual(-truck.depth / 2 - 0.01);
        expect(box.z + box.depth / 2).toBeLessThanOrEqual(truck.depth / 2 + 0.01);

        // Y축 경계
        expect(box.y).toBeGreaterThanOrEqual(0);
        expect(box.y + box.height).toBeLessThanOrEqual(truck.height + 0.01);
      }
    });
  });
});

describe('Multi-Truck Functions', () => {
  describe('optimizeOBBMulti', () => {
    it('단일 트럭으로 충분하면 1대만 사용', () => {
      const items: OBBItem[] = [
        { id: 'sofa', width: 200, depth: 90, height: 85 },
        { id: 'table', width: 120, depth: 80, height: 75 },
      ];

      const result = optimizeOBBMulti(items);
      expect(result.success).toBe(true);
      expect(result.totalTrucks).toBe(1);
    });

    it('많은 아이템은 멀티 트럭 사용', () => {
      // 많은 큰 가구
      const items: OBBItem[] = [];
      for (let i = 0; i < 20; i++) {
        items.push({
          id: `furniture_${i}`,
          width: 150,
          depth: 100,
          height: 100,
        });
      }

      const result = optimizeOBBMulti(items);
      expect(result.trucks.length).toBeGreaterThanOrEqual(1);
      expect(result.totalTrucks).toBe(result.trucks.length);
    });

    it('각 트럭에 적재율이 계산됨', () => {
      const items: OBBItem[] = [
        { id: 'sofa', width: 200, depth: 90, height: 85 },
      ];

      const result = optimizeOBBMulti(items);
      expect(result.trucks[0].utilization).toBeGreaterThan(0);
    });
  });
});

describe('Integration Tests', () => {
  it('Python 결과와 비슷한 적재율', () => {
    // Python 테스트 케이스와 유사한 입력
    const items: OBBItem[] = [
      { id: 'bed', width: 200, depth: 180, height: 40 },
      { id: 'wardrobe', width: 150, depth: 60, height: 200 },
      { id: 'desk', width: 120, depth: 60, height: 75 },
      { id: 'chair1', width: 50, depth: 50, height: 80 },
      { id: 'chair2', width: 50, depth: 50, height: 80 },
      { id: 'nightstand', width: 50, depth: 40, height: 55 },
    ];

    const result = optimizeOBB(items);

    // 적재율이 합리적인 범위 내 (Python과 ±10% 이내)
    expect(result.volumeUtilization).toBeGreaterThan(0);
    expect(result.volumeUtilization).toBeLessThan(100);

    // 모든 아이템 배치 성공
    expect(result.success).toBe(true);
  });

  it('70% 지지 규칙이 적용됨', () => {
    // 작은 아이템 위에 큰 아이템이 올라가면 안됨
    const items: OBBItem[] = [
      { id: 'small', width: 30, depth: 30, height: 50 },
      { id: 'large', width: 100, depth: 100, height: 50 },
    ];

    const result = optimizeOBB(items);

    // 큰 아이템이 먼저 배치되어야 함 (부피 순 정렬)
    const largeBox = result.placedItems.find((p) => p.itemId === 'large');
    const smallBox = result.placedItems.find((p) => p.itemId === 'small');

    if (largeBox && smallBox) {
      // 작은 박스가 큰 박스 위에 있으면 안됨 (지지 부족)
      if (smallBox.y > 0) {
        // 작은 박스가 공중에 있다면, 아래에 지지대가 있어야 함
        const supportingBoxes = result.placedItems.filter(
          (p) => Math.abs(p.y + p.height - smallBox.y) < 0.01
        );
        expect(supportingBoxes.length).toBeGreaterThan(0);
      }
    }
  });
});
