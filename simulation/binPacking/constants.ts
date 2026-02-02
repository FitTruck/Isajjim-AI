/**
 * 트럭 프리셋 및 상수
 *
 * Python obb_packer.py의 TRUCK_PRESETS_CM과 동일
 */

import { TruckPreset } from './types';

/**
 * 트럭 크기 (cm 단위)
 * - 1ton: 소형 트럭
 * - 2.5ton: 중형 트럭
 * - 5ton: 대형 트럭
 */
export const TRUCK_PRESETS: Record<string, TruckPreset> = {
  '1ton': { width: 170, depth: 280, height: 170 },
  '2.5ton': { width: 200, depth: 430, height: 190 },
  '5ton': { width: 230, depth: 620, height: 240 },
};

/**
 * 트럭 선택 순서 (작은 것부터)
 */
export const TRUCK_ORDER: string[] = ['1ton', '2.5ton', '5ton'];

/**
 * 기본 지지 비율 (70%)
 */
export const DEFAULT_SUPPORT_RATIO = 0.7;

/**
 * 경계/충돌 검사 허용 오차 (cm)
 */
export const TOLERANCE = 0.01;

/**
 * 최소 지지 검사용 높이 (바닥 판정)
 */
export const FLOOR_THRESHOLD = 0.001;
