"""
Worker Protocol for Persistent 3D Worker Pool

워커와 매니저 간 통신 프로토콜 정의.
JSON 기반 메시지 교환으로 stdin/stdout 통신.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from enum import Enum
import json


class MessageType(str, Enum):
    """메시지 타입"""
    INIT = "init"           # 워커 초기화 완료 신호
    TASK = "task"           # 3D 생성 요청
    RESULT = "result"       # 처리 결과
    HEARTBEAT = "heartbeat" # 헬스체크
    SHUTDOWN = "shutdown"   # 워커 종료 요청


@dataclass
class TaskMessage:
    """3D 생성 작업 요청 메시지"""
    task_id: str
    image_b64: str
    mask_b64: str
    seed: int = 42
    skip_gif: bool = True
    volume_only: bool = False  # Phase 4: Skip GLB/mesh if only volume needed

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.TASK.value,
            "data": asdict(self)
        })

    @classmethod
    def from_dict(cls, data: Dict) -> "TaskMessage":
        return cls(**data)


@dataclass
class ResultMessage:
    """3D 생성 결과 메시지"""
    task_id: str
    success: bool
    ply_b64: Optional[str] = None
    ply_size_bytes: Optional[int] = None
    gif_b64: Optional[str] = None
    gif_size_bytes: Optional[int] = None
    mesh_url: Optional[str] = None
    error: Optional[str] = None
    processing_time_seconds: float = 0.0
    # 워커 내부 부피 계산 결과 (최적화: PLY 전송 대신 결과만 전송)
    dimensions: Optional[Dict] = None  # {"volume": float, "bounding_box": {"width", "depth", "height"}}

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.RESULT.value,
            "data": asdict(self)
        })

    @classmethod
    def from_dict(cls, data: Dict) -> "ResultMessage":
        return cls(**data)


@dataclass
class InitMessage:
    """워커 초기화 완료 메시지"""
    worker_id: int
    gpu_id: int
    model_loaded: bool
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.INIT.value,
            "data": asdict(self)
        })

    @classmethod
    def from_dict(cls, data: Dict) -> "InitMessage":
        return cls(**data)


@dataclass
class HeartbeatMessage:
    """헬스체크 메시지"""
    worker_id: int
    status: str = "alive"
    gpu_memory_used_mb: float = 0.0

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.HEARTBEAT.value,
            "data": asdict(self)
        })

    @classmethod
    def from_dict(cls, data: Dict) -> "HeartbeatMessage":
        return cls(**data)


@dataclass
class ShutdownMessage:
    """워커 종료 요청 메시지"""
    reason: str = "normal"

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.SHUTDOWN.value,
            "data": asdict(self)
        })


def parse_message(json_str: str) -> Dict[str, Any]:
    """JSON 메시지를 파싱하여 타입과 데이터 반환"""
    try:
        msg = json.loads(json_str.strip())
        return {
            "type": MessageType(msg.get("type")),
            "data": msg.get("data", {})
        }
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "type": None,
            "data": {"error": str(e), "raw": json_str}
        }
