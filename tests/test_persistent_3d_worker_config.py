"""
Persistent 3D Worker 설정 테스트

STAGE1_INFERENCE_STEPS, STAGE2_INFERENCE_STEPS 등 성능 최적화 설정을 검증합니다.
"""

import pytest
import sys
import os


class TestPersistent3DWorkerConfig:
    """persistent_3d_worker.py 설정 테스트"""

    def test_stage1_inference_steps_is_14(self):
        """STAGE1_INFERENCE_STEPS가 14로 설정되어 있어야 함 (속도/정확도 균형)"""
        # persistent_3d_worker.py 파일에서 설정 읽기
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..", "ai", "subprocess", "persistent_3d_worker.py"
        )

        with open(worker_path, "r") as f:
            content = f.read()

        # STAGE1_INFERENCE_STEPS = 14 라인 찾기
        import re
        match = re.search(r"STAGE1_INFERENCE_STEPS\s*=\s*(\d+)", content)

        assert match is not None, "STAGE1_INFERENCE_STEPS 설정을 찾을 수 없습니다"
        stage1_steps = int(match.group(1))
        assert stage1_steps == 14, f"STAGE1_INFERENCE_STEPS는 14여야 합니다 (현재: {stage1_steps})"

    def test_stage2_inference_steps_is_8(self):
        """STAGE2_INFERENCE_STEPS가 8로 설정되어 있어야 함 (속도 우선)"""
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..", "ai", "subprocess", "persistent_3d_worker.py"
        )

        with open(worker_path, "r") as f:
            content = f.read()

        import re
        match = re.search(r"STAGE2_INFERENCE_STEPS\s*=\s*(\d+)", content)

        assert match is not None, "STAGE2_INFERENCE_STEPS 설정을 찾을 수 없습니다"
        stage2_steps = int(match.group(1))
        assert stage2_steps == 8, f"STAGE2_INFERENCE_STEPS는 8이어야 합니다 (현재: {stage2_steps})"

    def test_gaussian_only_mode_enabled(self):
        """GAUSSIAN_ONLY_MODE가 True로 설정되어 있어야 함 (부피 계산 최적화)"""
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..", "ai", "subprocess", "persistent_3d_worker.py"
        )

        with open(worker_path, "r") as f:
            content = f.read()

        import re
        match = re.search(r"GAUSSIAN_ONLY_MODE\s*=\s*(True|False)", content)

        assert match is not None, "GAUSSIAN_ONLY_MODE 설정을 찾을 수 없습니다"
        gaussian_only = match.group(1) == "True"
        assert gaussian_only, "GAUSSIAN_ONLY_MODE는 True여야 합니다"

    def test_binary_ply_enabled(self):
        """USE_BINARY_PLY가 True로 설정되어 있어야 함 (I/O 최적화)"""
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..", "ai", "subprocess", "persistent_3d_worker.py"
        )

        with open(worker_path, "r") as f:
            content = f.read()

        import re
        match = re.search(r"USE_BINARY_PLY\s*=\s*(True|False)", content)

        assert match is not None, "USE_BINARY_PLY 설정을 찾을 수 없습니다"
        binary_ply = match.group(1) == "True"
        assert binary_ply, "USE_BINARY_PLY는 True여야 합니다"

    def test_max_image_size_disabled(self):
        """MAX_IMAGE_SIZE가 None으로 설정되어 있어야 함 (부피 정확도 유지)"""
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..", "ai", "subprocess", "persistent_3d_worker.py"
        )

        with open(worker_path, "r") as f:
            content = f.read()

        import re
        match = re.search(r"MAX_IMAGE_SIZE\s*=\s*(None|\d+)", content)

        assert match is not None, "MAX_IMAGE_SIZE 설정을 찾을 수 없습니다"
        max_size = match.group(1)
        assert max_size == "None", f"MAX_IMAGE_SIZE는 None이어야 합니다 (현재: {max_size})"

    def test_stage1_steps_in_valid_range(self):
        """STAGE1_INFERENCE_STEPS가 유효한 범위(8-30) 내에 있어야 함"""
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..", "ai", "subprocess", "persistent_3d_worker.py"
        )

        with open(worker_path, "r") as f:
            content = f.read()

        import re
        match = re.search(r"STAGE1_INFERENCE_STEPS\s*=\s*(\d+)", content)

        assert match is not None
        stage1_steps = int(match.group(1))

        # 유효 범위: 8 이상, 30 이하
        assert 8 <= stage1_steps <= 30, (
            f"STAGE1_INFERENCE_STEPS는 8-30 범위여야 합니다 (현재: {stage1_steps})"
        )

    def test_stage2_steps_in_valid_range(self):
        """STAGE2_INFERENCE_STEPS가 유효한 범위(4-16) 내에 있어야 함"""
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..", "ai", "subprocess", "persistent_3d_worker.py"
        )

        with open(worker_path, "r") as f:
            content = f.read()

        import re
        match = re.search(r"STAGE2_INFERENCE_STEPS\s*=\s*(\d+)", content)

        assert match is not None
        stage2_steps = int(match.group(1))

        # 유효 범위: 4 이상, 16 이하
        assert 4 <= stage2_steps <= 16, (
            f"STAGE2_INFERENCE_STEPS는 4-16 범위여야 합니다 (현재: {stage2_steps})"
        )


class TestConfigurationComments:
    """설정에 대한 문서화 테스트"""

    def test_stage1_has_comment(self):
        """STAGE1_INFERENCE_STEPS 설정에 설명 주석이 있어야 함"""
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..", "ai", "subprocess", "persistent_3d_worker.py"
        )

        with open(worker_path, "r") as f:
            content = f.read()

        # STAGE1_INFERENCE_STEPS 라인에 주석이 있는지 확인
        import re
        match = re.search(r"STAGE1_INFERENCE_STEPS\s*=\s*\d+\s*#\s*.+", content)

        assert match is not None, "STAGE1_INFERENCE_STEPS에 설명 주석이 필요합니다"

    def test_performance_config_section_exists(self):
        """Performance Optimization Configuration 섹션이 존재해야 함"""
        worker_path = os.path.join(
            os.path.dirname(__file__),
            "..", "ai", "subprocess", "persistent_3d_worker.py"
        )

        with open(worker_path, "r") as f:
            content = f.read()

        assert "Performance Optimization Configuration" in content, (
            "Performance Optimization Configuration 섹션이 필요합니다"
        )
