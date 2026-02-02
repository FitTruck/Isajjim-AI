/**
 * 3D Bin Packing 타입 정의
 *
 * Python obb_packer.py의 타입을 TypeScript로 변환
 * - PlacedBox: 배치된 박스 정보
 * - PackingResult: 단일 트럭 패킹 결과
 * - MultiTruckResult: 멀티 트럭 패킹 결과
 */

/**
 * 트럭 프리셋 치수 (cm)
 */
export interface TruckPreset {
  width: number;
  depth: number;
  height: number;
}

/**
 * OBB 아이템 (입력용)
 */
export interface OBBItem {
  id: string;
  width: number; // X축 (cm)
  depth: number; // Z축 (cm)
  height: number; // Y축 (cm)
}

/**
 * 회전 방향 (수평 회전만 허용)
 * LWH: 기본 방향 (Length, Width, Height)
 * WLH: 90도 수평 회전 (Width, Length, Height)
 */
export enum Orientation {
  LWH = 0, // 기본 (L, W, H) - 원래 방향
  WLH = 2, // 90도 회전 (W, L, H) - 수평 회전
}

/**
 * 배치된 박스 정보
 *
 * 좌표계: 트럭 중심이 원점 (Python과 동일)
 * - X: 좌우 (-width/2 ~ +width/2)
 * - Y: 바닥 기준 (0 ~ height)
 * - Z: 앞뒤 (-depth/2 ~ +depth/2)
 */
export interface PlacedBox {
  itemId: string;
  x: number; // 중심 좌표
  y: number; // 바닥 좌표 (중심 아님)
  z: number; // 중심 좌표
  width: number; // X축
  depth: number; // Z축
  height: number; // Y축
  orientation: Orientation;
}

/**
 * Extreme Point (EP) - 배치 후보 위치
 */
export interface ExtremePoint {
  x: number;
  y: number;
  z: number;
}

/**
 * 단일 트럭 패킹 결과
 */
export interface PackingResult {
  success: boolean;
  truckType: string;
  placedItems: PlacedBox[];
  unplacedItems: string[];
  volumeUtilization: number;
  message: string;
}

/**
 * 트럭 배치 정보 (멀티 트럭용)
 */
export interface TruckPlacement {
  type: string;
  placements: PlacedBox[];
  utilization: number;
}

/**
 * 멀티 트럭 패킹 결과
 */
export interface MultiTruckResult {
  success: boolean;
  trucks: TruckPlacement[];
  totalTrucks: number;
  unplacedItems: string[];
  message: string;
}

/**
 * 패킹 옵션
 */
export interface PackingOptions {
  supportRatio?: number; // 지지 비율 (기본 0.7 = 70%)
  allowTilt?: boolean; // 6방향 회전 허용 (기본 false)
  cornerFirst?: boolean; // 코너 우선 배치 (기본 true)
}
