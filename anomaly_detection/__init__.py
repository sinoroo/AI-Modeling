"""
Anomaly Detection Pipeline Package

Modules:
- config: Configuration management
- data_loader: Data loading and EDA
- preprocessing: Data preprocessing and feature engineering
- model_training: Model training (classical ML + DL)
- evaluation: Model evaluation and comparison
- inference_serial: Real-time serial inference
"""

__version__ = "1.0.0"
__author__ = "AI/ML Engineering Team"

# 일부 선택적 의존성(matplotlib/torch 등)이 환경에 따라 import 실패할 수 있으므로
# 서브모듈을 지연(lazy) import 한다. 이렇게 하면 가벼운 config 만 필요할 때
# matplotlib/torch 를 끌어오지 않아, 한 모듈의 ABI 문제(numpy 2.x vs 구버전
# matplotlib)가 패키지 전체를 깨뜨리지 않는다. (PEP 562 module __getattr__)
import importlib as _importlib

_SUBMODULES = (
    "config",
    "data_loader",
    "preprocessing",
    "model_training",
    "evaluation",
    "inference_serial",
    "mlflow_utils",
    "feature_extraction",
)

__all__ = list(_SUBMODULES)


def __getattr__(name):  # PEP 562: 접근 시점에 해당 서브모듈만 로드
    if name in _SUBMODULES:
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES))


