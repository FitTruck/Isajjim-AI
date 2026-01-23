"""
Tests for ai/subprocess/worker_protocol.py

Worker protocol message tests:
- Message serialization/deserialization
- Message type validation
- JSON parsing
"""

import pytest
import json

from ai.subprocess.worker_protocol import (
    MessageType,
    TaskMessage,
    ResultMessage,
    InitMessage,
    HeartbeatMessage,
    ShutdownMessage,
    parse_message,
)


class TestMessageType:
    """MessageType enum tests"""

    def test_message_types_defined(self):
        """All message types are defined"""
        assert MessageType.INIT.value == "init"
        assert MessageType.TASK.value == "task"
        assert MessageType.RESULT.value == "result"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessageType.SHUTDOWN.value == "shutdown"


class TestTaskMessage:
    """TaskMessage dataclass tests"""

    def test_create_task_message(self):
        """Create task message with required fields"""
        msg = TaskMessage(
            task_id="task-123",
            image_b64="base64_image_data",
            mask_b64="base64_mask_data"
        )
        assert msg.task_id == "task-123"
        assert msg.image_b64 == "base64_image_data"
        assert msg.mask_b64 == "base64_mask_data"
        assert msg.seed == 42  # default
        assert msg.skip_gif is True  # default
        assert msg.volume_only is False  # default

    def test_create_task_message_with_options(self):
        """Create task message with custom options"""
        msg = TaskMessage(
            task_id="task-456",
            image_b64="img",
            mask_b64="mask",
            seed=123,
            skip_gif=False,
            volume_only=True
        )
        assert msg.seed == 123
        assert msg.skip_gif is False
        assert msg.volume_only is True

    def test_task_message_to_json(self):
        """TaskMessage serialization to JSON"""
        msg = TaskMessage(
            task_id="task-789",
            image_b64="img",
            mask_b64="mask",
            seed=42
        )
        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "task"
        assert parsed["data"]["task_id"] == "task-789"
        assert parsed["data"]["image_b64"] == "img"
        assert parsed["data"]["seed"] == 42

    def test_task_message_from_dict(self):
        """TaskMessage deserialization from dict"""
        data = {
            "task_id": "task-abc",
            "image_b64": "image_data",
            "mask_b64": "mask_data",
            "seed": 100,
            "skip_gif": False,
            "volume_only": True
        }
        msg = TaskMessage.from_dict(data)

        assert msg.task_id == "task-abc"
        assert msg.seed == 100
        assert msg.skip_gif is False
        assert msg.volume_only is True


class TestResultMessage:
    """ResultMessage dataclass tests"""

    def test_create_success_result(self):
        """Create successful result message"""
        msg = ResultMessage(
            task_id="result-123",
            success=True,
            ply_b64="ply_data",
            ply_size_bytes=1000,
            processing_time_seconds=2.5
        )
        assert msg.success is True
        assert msg.ply_b64 == "ply_data"
        assert msg.ply_size_bytes == 1000
        assert msg.error is None

    def test_create_failed_result(self):
        """Create failed result message"""
        msg = ResultMessage(
            task_id="result-456",
            success=False,
            error="Processing failed"
        )
        assert msg.success is False
        assert msg.error == "Processing failed"
        assert msg.ply_b64 is None

    def test_result_message_with_dimensions(self):
        """Result message with volume dimensions"""
        msg = ResultMessage(
            task_id="result-789",
            success=True,
            dimensions={
                "volume": 1.5,
                "bounding_box": {"width": 1.0, "depth": 1.0, "height": 1.5}
            }
        )
        assert msg.dimensions is not None
        assert msg.dimensions["volume"] == 1.5
        assert msg.dimensions["bounding_box"]["width"] == 1.0

    def test_result_message_to_json(self):
        """ResultMessage serialization to JSON"""
        msg = ResultMessage(
            task_id="result-json",
            success=True,
            ply_b64="ply",
            processing_time_seconds=1.5
        )
        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "result"
        assert parsed["data"]["task_id"] == "result-json"
        assert parsed["data"]["success"] is True
        assert parsed["data"]["processing_time_seconds"] == 1.5

    def test_result_message_from_dict(self):
        """ResultMessage deserialization from dict"""
        data = {
            "task_id": "result-dict",
            "success": True,
            "ply_b64": "ply_data",
            "ply_size_bytes": 500,
            "gif_b64": "gif_data",
            "gif_size_bytes": 200,
            "mesh_url": "/assets/mesh.glb",
            "error": None,
            "processing_time_seconds": 3.0,
            "dimensions": {"volume": 2.0}
        }
        msg = ResultMessage.from_dict(data)

        assert msg.task_id == "result-dict"
        assert msg.ply_size_bytes == 500
        assert msg.gif_b64 == "gif_data"
        assert msg.mesh_url == "/assets/mesh.glb"
        assert msg.dimensions["volume"] == 2.0


class TestInitMessage:
    """InitMessage dataclass tests"""

    def test_create_init_message_success(self):
        """Create successful init message"""
        msg = InitMessage(
            worker_id=0,
            gpu_id=0,
            model_loaded=True
        )
        assert msg.worker_id == 0
        assert msg.gpu_id == 0
        assert msg.model_loaded is True
        assert msg.error is None

    def test_create_init_message_failed(self):
        """Create failed init message"""
        msg = InitMessage(
            worker_id=1,
            gpu_id=1,
            model_loaded=False,
            error="GPU memory insufficient"
        )
        assert msg.model_loaded is False
        assert msg.error == "GPU memory insufficient"

    def test_init_message_to_json(self):
        """InitMessage serialization to JSON"""
        msg = InitMessage(
            worker_id=2,
            gpu_id=2,
            model_loaded=True
        )
        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "init"
        assert parsed["data"]["worker_id"] == 2
        assert parsed["data"]["model_loaded"] is True

    def test_init_message_from_dict(self):
        """InitMessage deserialization from dict"""
        data = {
            "worker_id": 3,
            "gpu_id": 3,
            "model_loaded": True,
            "error": None
        }
        msg = InitMessage.from_dict(data)

        assert msg.worker_id == 3
        assert msg.gpu_id == 3


class TestHeartbeatMessage:
    """HeartbeatMessage dataclass tests"""

    def test_create_heartbeat_message(self):
        """Create heartbeat message with defaults"""
        msg = HeartbeatMessage(worker_id=0)
        assert msg.worker_id == 0
        assert msg.status == "alive"
        assert msg.gpu_memory_used_mb == 0.0

    def test_create_heartbeat_with_memory(self):
        """Create heartbeat with GPU memory info"""
        msg = HeartbeatMessage(
            worker_id=1,
            status="alive",
            gpu_memory_used_mb=4096.5
        )
        assert msg.gpu_memory_used_mb == 4096.5

    def test_heartbeat_message_to_json(self):
        """HeartbeatMessage serialization to JSON"""
        msg = HeartbeatMessage(
            worker_id=4,
            status="busy",
            gpu_memory_used_mb=8192.0
        )
        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "heartbeat"
        assert parsed["data"]["worker_id"] == 4
        assert parsed["data"]["status"] == "busy"

    def test_heartbeat_message_from_dict(self):
        """HeartbeatMessage deserialization from dict"""
        data = {
            "worker_id": 5,
            "status": "idle",
            "gpu_memory_used_mb": 2048.0
        }
        msg = HeartbeatMessage.from_dict(data)

        assert msg.worker_id == 5
        assert msg.status == "idle"


class TestShutdownMessage:
    """ShutdownMessage dataclass tests"""

    def test_create_shutdown_message_default(self):
        """Create shutdown message with default reason"""
        msg = ShutdownMessage()
        assert msg.reason == "normal"

    def test_create_shutdown_message_custom(self):
        """Create shutdown message with custom reason"""
        msg = ShutdownMessage(reason="timeout")
        assert msg.reason == "timeout"

    def test_shutdown_message_to_json(self):
        """ShutdownMessage serialization to JSON"""
        msg = ShutdownMessage(reason="error")
        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "shutdown"
        assert parsed["data"]["reason"] == "error"


class TestParseMessage:
    """parse_message function tests"""

    def test_parse_valid_task_message(self):
        """Parse valid task message JSON"""
        json_str = json.dumps({
            "type": "task",
            "data": {"task_id": "test", "image_b64": "img", "mask_b64": "mask"}
        })
        result = parse_message(json_str)

        assert result["type"] == MessageType.TASK
        assert result["data"]["task_id"] == "test"

    def test_parse_valid_result_message(self):
        """Parse valid result message JSON"""
        json_str = json.dumps({
            "type": "result",
            "data": {"task_id": "test", "success": True}
        })
        result = parse_message(json_str)

        assert result["type"] == MessageType.RESULT
        assert result["data"]["success"] is True

    def test_parse_valid_init_message(self):
        """Parse valid init message JSON"""
        json_str = json.dumps({
            "type": "init",
            "data": {"worker_id": 0, "gpu_id": 0, "model_loaded": True}
        })
        result = parse_message(json_str)

        assert result["type"] == MessageType.INIT
        assert result["data"]["model_loaded"] is True

    def test_parse_valid_heartbeat_message(self):
        """Parse valid heartbeat message JSON"""
        json_str = json.dumps({
            "type": "heartbeat",
            "data": {"worker_id": 0, "status": "alive"}
        })
        result = parse_message(json_str)

        assert result["type"] == MessageType.HEARTBEAT

    def test_parse_valid_shutdown_message(self):
        """Parse valid shutdown message JSON"""
        json_str = json.dumps({
            "type": "shutdown",
            "data": {"reason": "normal"}
        })
        result = parse_message(json_str)

        assert result["type"] == MessageType.SHUTDOWN

    def test_parse_invalid_json(self):
        """Parse invalid JSON returns error"""
        result = parse_message("not valid json {")

        assert result["type"] is None
        assert "error" in result["data"]
        assert "raw" in result["data"]

    def test_parse_invalid_message_type(self):
        """Parse invalid message type returns error"""
        json_str = json.dumps({
            "type": "unknown_type",
            "data": {}
        })
        result = parse_message(json_str)

        assert result["type"] is None
        assert "error" in result["data"]

    def test_parse_message_with_whitespace(self):
        """Parse message with leading/trailing whitespace"""
        json_str = "  \n" + json.dumps({
            "type": "task",
            "data": {"task_id": "test", "image_b64": "img", "mask_b64": "mask"}
        }) + "  \n"
        result = parse_message(json_str)

        assert result["type"] == MessageType.TASK

    def test_parse_message_empty_data(self):
        """Parse message with missing data field"""
        json_str = json.dumps({"type": "task"})
        result = parse_message(json_str)

        assert result["type"] == MessageType.TASK
        assert result["data"] == {}
