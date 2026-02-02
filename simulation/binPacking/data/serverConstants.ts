/**
 * 서버 상수 정의
 */

/** 트럭 프리셋 (m 단위) - 프론트엔드용 */
export const TRUCK_SPECS_M = {
  '1ton': { name: '1톤', width: 1.7, depth: 2.8, height: 1.7 },
  '2.5ton': { name: '2.5톤', width: 2.0, depth: 4.3, height: 1.9 },
  '5ton': { name: '5톤', width: 2.3, depth: 6.2, height: 2.4 },
} as const;

export type TruckType = keyof typeof TRUCK_SPECS_M;
