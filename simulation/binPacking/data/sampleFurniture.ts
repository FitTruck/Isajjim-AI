/**
 * 샘플 가구 데이터 (PLY 파일 포함)
 */

export interface FurnitureItem {
  id: string;
  label: string;
  type: string;
  width: number;
  depth: number;
  height: number;
  ply_url: string;
}

/** 샘플 가구 목록 생성 */
export function getSampleFurniture(estimateId: number): FurnitureItem[] {
  return [
    {
      id: `${estimateId}_bed_001`,
      label: 'bed',
      type: 'BED',
      width: 1.3,
      depth: 2.28,
      height: 0.59,
      ply_url: '/assets/aligned/2_BED_1.ply',
    },
    {
      id: `${estimateId}_bed_002`,
      label: 'bed',
      type: 'BED',
      width: 1.82,
      depth: 2.6,
      height: 0.65,
      ply_url: '/assets/aligned/1_BED_6.ply',
    },
    {
      id: `${estimateId}_sofa_001`,
      label: 'sofa',
      type: 'SOFA',
      width: 1.04,
      depth: 1.11,
      height: 0.98,
      ply_url: '/assets/aligned/9_SOFA_1.ply',
    },
    {
      id: `${estimateId}_sofa_002`,
      label: 'sofa',
      type: 'SOFA',
      width: 2.34,
      depth: 1.17,
      height: 1.04,
      ply_url: '/assets/aligned/12_SOFA_0.ply',
    },
    {
      id: `${estimateId}_table_001`,
      label: 'coffee table',
      type: 'COFFEE_TABLE',
      width: 1.04,
      depth: 0.65,
      height: 0.59,
      ply_url: '/assets/aligned/4_COFFEE_TABLE_1.ply',
    },
    {
      id: `${estimateId}_chair_001`,
      label: 'chair',
      type: 'CHAIR_STOOL',
      width: 0.52,
      depth: 0.52,
      height: 0.59,
      ply_url: '/assets/aligned/4_CHAIR_STOOL_3.ply',
    },
    {
      id: `${estimateId}_cabinet_001`,
      label: 'cabinet',
      type: 'CABINET',
      width: 0.78,
      depth: 0.52,
      height: 1.56,
      ply_url: '/assets/aligned/8_CABINET_1.ply',
    },
    {
      id: `${estimateId}_nightstand_001`,
      label: 'nightstand',
      type: 'NIGHTSTAND',
      width: 0.52,
      depth: 0.52,
      height: 0.72,
      ply_url: '/assets/aligned/2_NIGHTSTAND_0.ply',
    },
    {
      id: `${estimateId}_tv_001`,
      label: 'television',
      type: 'MONITOR_TV',
      width: 1.59,
      depth: 0.09,
      height: 0.91,
      ply_url: '/assets/aligned/2_MONITOR_TV_3.ply',
    },
    {
      id: `${estimateId}_plant_001`,
      label: 'potted plant',
      type: 'POTTED_PLANT',
      width: 0.46,
      depth: 0.46,
      height: 1.17,
      ply_url: '/assets/aligned/2_POTTED_PLANT_4.ply',
    },
  ];
}
