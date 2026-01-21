"""
Tests for ai/config.py

Config 모듈의 단위 테스트:
- 기본 설정값 검증
- GPU 관련 함수 테스트
- 환경 변수 생성 테스트
"""

from unittest.mock import patch


class TestConfigBasics:
    """Config 기본 설정 테스트"""

    def test_config_import(self):
        """Config 클래스 임포트 테스트"""
        from ai.config import Config
        assert Config is not None

    def test_device_is_string(self):
        """DEVICE가 문자열인지 확인"""
        from ai.config import Config
        assert isinstance(Config.DEVICE, str)
        assert Config.DEVICE in ["cuda", "cpu"]

    def test_default_gpu_id(self):
        """기본 GPU ID 확인"""
        from ai.config import Config
        assert Config.DEFAULT_GPU_ID == 0

    def test_multi_gpu_enabled(self):
        """Multi-GPU 설정 확인"""
        from ai.config import Config
        assert isinstance(Config.ENABLE_MULTI_GPU, bool)

    def test_max_images_per_gpu(self):
        """GPU당 최대 이미지 수 확인"""
        from ai.config import Config
        assert Config.MAX_IMAGES_PER_GPU >= 1

    def test_yolo_model_path(self):
        """YOLO 모델 경로 확인"""
        from ai.config import Config
        assert Config.YOLO_MODEL_PATH == 'yoloe-26x-seg.pt'

    def test_font_settings(self):
        """폰트 설정 확인"""
        from ai.config import Config
        assert Config.FONT_PATH == "fonts/NanumGothic-Regular.ttf"
        assert Config.FONT_SIZE_LARGE > 0
        assert Config.FONT_SIZE_SMALL > 0

    def test_detection_thresholds(self):
        """탐지 임계값 확인"""
        from ai.config import Config
        assert 0 < Config.CONF_THRESHOLD_MAIN < 1
        assert 0 < Config.CONF_THRESHOLD_SMALL < 1
        assert Config.CONF_THRESHOLD_SMALL <= Config.CONF_THRESHOLD_MAIN


class TestGetDevice:
    """get_device 함수 테스트"""

    def test_get_device_with_none(self):
        """gpu_id가 None일 때 기본 디바이스 반환"""
        from ai.config import Config
        device = Config.get_device(None)
        assert isinstance(device, str)

    def test_get_device_with_specific_id(self):
        """특정 GPU ID로 디바이스 문자열 생성"""
        from ai.config import Config
        with patch.object(Config, 'DEVICE', 'cuda'):
            with patch('torch.cuda.is_available', return_value=True):
                device = Config.get_device(0)
                assert device == "cuda:0"

                device = Config.get_device(1)
                assert device == "cuda:1"

    def test_get_device_no_cuda(self):
        """CUDA 사용 불가 시 CPU 반환"""
        from ai.config import Config
        with patch('torch.cuda.is_available', return_value=False):
            device = Config.get_device(0)
            assert device == "cpu"


class TestGetAvailableGpus:
    """get_available_gpus 함수 테스트"""

    def test_returns_list(self):
        """리스트 반환 확인"""
        from ai.config import Config
        result = Config.get_available_gpus()
        assert isinstance(result, list)

    def test_with_custom_gpu_ids(self):
        """GPU_IDS가 설정된 경우"""
        from ai.config import Config
        original = Config.GPU_IDS
        try:
            Config.GPU_IDS = [0, 2, 4]
            result = Config.get_available_gpus()
            assert result == [0, 2, 4]
        finally:
            Config.GPU_IDS = original

    def test_no_cuda_returns_empty(self):
        """CUDA 사용 불가 시 빈 리스트"""
        from ai.config import Config
        original = Config.GPU_IDS
        try:
            Config.GPU_IDS = None
            with patch('torch.cuda.is_available', return_value=False):
                result = Config.get_available_gpus()
                assert result == []
        finally:
            Config.GPU_IDS = original


class TestGetSpconvEnvVars:
    """get_spconv_env_vars 함수 테스트"""

    def test_returns_dict(self):
        """딕셔너리 반환 확인"""
        from ai.config import Config
        result = Config.get_spconv_env_vars(0)
        assert isinstance(result, dict)

    def test_cuda_visible_devices(self):
        """CUDA_VISIBLE_DEVICES 설정 확인"""
        from ai.config import Config

        result = Config.get_spconv_env_vars(0)
        assert result["CUDA_VISIBLE_DEVICES"] == "0"

        result = Config.get_spconv_env_vars(3)
        assert result["CUDA_VISIBLE_DEVICES"] == "3"

    def test_spconv_tune_device_always_zero(self):
        """SPCONV_TUNE_DEVICE는 항상 0"""
        from ai.config import Config

        for gpu_id in [0, 1, 2, 3]:
            result = Config.get_spconv_env_vars(gpu_id)
            assert result["SPCONV_TUNE_DEVICE"] == "0"

    def test_spconv_algo_time_limit(self):
        """SPCONV_ALGO_TIME_LIMIT 설정 확인"""
        from ai.config import Config
        result = Config.get_spconv_env_vars(0)
        assert "SPCONV_ALGO_TIME_LIMIT" in result
        assert result["SPCONV_ALGO_TIME_LIMIT"] == "100"


class TestCheckDependencies:
    """check_dependencies 함수 테스트"""

    def test_check_dependencies_runs(self):
        """check_dependencies 실행 확인"""
        from ai.config import Config
        # 예외 없이 실행되어야 함
        Config.check_dependencies()

    def test_check_dependencies_with_missing_font(self):
        """폰트 파일이 없을 때 경고 출력"""
        from ai.config import Config

        original_path = Config.FONT_PATH
        try:
            Config.FONT_PATH = "/nonexistent/path/font.ttf"
            # 예외 없이 실행되어야 함 (경고만 출력)
            Config.check_dependencies()
        finally:
            Config.FONT_PATH = original_path
