/**
 * TypeScript Bin Packing 시각화 서버
 *
 * Express 서버로 3D Bin Packing 시각화 제공
 * - GET /: 시뮬레이터 HTML 페이지
 * - POST /api/optimize: 단일 트럭 최적화
 * - POST /api/optimize-multi: 멀티 트럭 최적화
 */

import express, { Request, Response } from 'express';
import cors from 'cors';
import path from 'path';
import { optimizeOBB, optimizeOBBMulti, OBBItem, PackingOptions, PlacedBox } from './index';
import { TRUCK_SPECS_M, TruckType, getSampleFurniture } from './data';

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// ==================== Static Files ====================

app.use('/static', express.static(path.join(__dirname, 'static')));

// PLY assets - float 변환된 PLY 사용 (Three.js 호환성)
const ASSETS_DIR = path.join(__dirname, '..', 'assets');
app.use('/assets/aligned', express.static(path.join(ASSETS_DIR, 'aligned_float')));
app.use('/assets', express.static(ASSETS_DIR));

// ==================== Helper Functions ====================

/** 단위 변환 스케일 계산 */
function getScale(unit: 'm' | 'cm'): number {
  return unit === 'm' ? 100 : 1;
}

/** 아이템 스케일 변환 (m -> cm) */
function scaleItems(items: OBBItem[], scale: number): OBBItem[] {
  return items.map((item) => ({
    id: item.id,
    width: item.width * scale,
    depth: item.depth * scale,
    height: item.height * scale,
  }));
}

/** 배치 결과 스케일 역변환 (cm -> m) */
function scalePlacements(placements: PlacedBox[], scale: number) {
  return placements.map((p) => ({
    id: p.itemId,
    x: p.x / scale,
    y: p.y / scale,
    z: p.z / scale,
    width: p.width / scale,
    depth: p.depth / scale,
    height: p.height / scale,
    orientation: p.orientation,
  }));
}

// ==================== API Routes ====================

/** 시뮬레이터 HTML 페이지 */
app.get('/', (_req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, 'static', 'simulator.html'));
});

/** 트럭 프리셋 조회 */
app.get('/api/trucks', (_req: Request, res: Response) => {
  res.json(TRUCK_SPECS_M);
});

/** 시뮬레이션 데이터 (샘플 가구) */
app.get('/api/data/:estimateId', (req: Request, res: Response) => {
  const estimateId = parseInt(req.params.estimateId as string) || 123;
  const truckType = (req.query.truck_type as TruckType) || '2.5ton';
  const truck = TRUCK_SPECS_M[truckType] || TRUCK_SPECS_M['2.5ton'];

  res.json({
    estimate_id: estimateId,
    truck,
    furniture: getSampleFurniture(estimateId),
  });
});

/** 단일 트럭 최적화 API */
interface OptimizeRequest {
  items: OBBItem[];
  truck_type?: string;
  unit?: 'm' | 'cm';
  support_ratio?: number;
}

app.post('/api/optimize', (req: Request<unknown, unknown, OptimizeRequest>, res: Response) => {
  try {
    const { items, truck_type, unit = 'm', support_ratio = 0.7 } = req.body;

    if (!items || !Array.isArray(items)) {
      res.status(400).json({ error: 'items array is required' });
      return;
    }

    const scale = getScale(unit);
    const scaledItems = scaleItems(items, scale);
    const options: PackingOptions = { supportRatio: support_ratio };
    const result = optimizeOBB(scaledItems, truck_type, options);

    res.json({
      success: result.success,
      truck_type: result.truckType,
      placements: scalePlacements(result.placedItems, scale),
      unplaced_ids: result.unplacedItems,
      volume_utilization: result.volumeUtilization,
      message: result.message,
    });
  } catch (error) {
    console.error('Optimize error:', error);
    res.status(500).json({ error: 'Optimization failed' });
  }
});

/** 멀티 트럭 최적화 API */
app.post('/api/optimize-multi', (req: Request<unknown, unknown, OptimizeRequest>, res: Response) => {
  try {
    const { items, unit = 'm', support_ratio = 0.7 } = req.body;

    if (!items || !Array.isArray(items)) {
      res.status(400).json({ error: 'items array is required' });
      return;
    }

    const scale = getScale(unit);
    const scaledItems = scaleItems(items, scale);
    const options: PackingOptions = { supportRatio: support_ratio };
    const result = optimizeOBBMulti(scaledItems, options);

    res.json({
      success: result.success,
      trucks: result.trucks.map((truck) => ({
        type: truck.type,
        utilization: truck.utilization,
        placements: scalePlacements(truck.placements, scale),
      })),
      total_trucks: result.totalTrucks,
      unplaced_ids: result.unplacedItems,
      message: result.message,
    });
  } catch (error) {
    console.error('Multi-truck optimize error:', error);
    res.status(500).json({ error: 'Multi-truck optimization failed' });
  }
});

// ==================== Server Start ====================

app.listen(PORT, () => {
  console.log(`\n🚀 TypeScript Bin Packing Server`);
  console.log(`   http://localhost:${PORT}`);
  console.log(`\n📦 API Endpoints:`);
  console.log(`   GET  /           - Simulator page`);
  console.log(`   GET  /api/trucks - Truck presets`);
  console.log(`   POST /api/optimize       - Single truck optimization`);
  console.log(`   POST /api/optimize-multi - Multi-truck optimization`);
  console.log(`\nPress Ctrl+C to stop\n`);
});
