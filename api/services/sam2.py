"""
SAM 2 Model Service

SAM 2 모델 초기화 및 관리
"""

from transformers import Sam2Processor, Sam2Model
from api.config import device

# Global model instances
_model = None
_processor = None


def initialize_model():
    """Initialize SAM 2 model and processor from Hugging Face"""
    global _model, _processor

    try:
        model_id = "facebook/sam2.1-hiera-large"
        print(f"Loading model from {model_id}...")

        _processor = Sam2Processor.from_pretrained(model_id)
        _model = Sam2Model.from_pretrained(model_id).to(device)

        print("SAM 2 model and processor initialized successfully")

    except Exception as e:
        print(f"Error initializing model: {e}")
        raise


def get_model():
    """Get the SAM 2 model instance"""
    return _model


def get_processor():
    """Get the SAM 2 processor instance"""
    return _processor


def is_model_loaded() -> bool:
    """Check if model is loaded"""
    return _model is not None and _processor is not None
