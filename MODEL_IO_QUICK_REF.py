"""
모델 입출력 형식 빠른 참조 (Quick Reference)
"""

import json

# 모든 모델의 입출력 형식 요약
MODEL_IO_SPECS = {
    "data_flow": {
        "stage_1": {
            "name": "CSV 로드",
            "input": "CSV 파일들 (메타데이터 + 진동 데이터)",
            "output": "DataFrame (n_rows, 3)",
            "example": "train_normal_000.csv → (499, 3) [time, vibration, label]"
        },
        "stage_2": {
            "name": "전처리",
            "input": "DataFrame (4990, 3)",
            "output": "3D Numpy Array (n_windows, 64, 1)",
            "operations": ["결측치 처리", "이상치 제거", "정규화", "윈도우 생성"],
            "example": "(4990, 3) → (154, 64, 1)"
        },
        "stage_3a": {
            "name": "Classical ML (RF/IF/SVM)",
            "input": "(154, 64, 1) 3D Array",
            "transformation": "Flatten → (154, 64) 2D Array",
            "output_predict": "(27,) - 클래스 레이블",
            "output_proba": "(27, 2) - 확률",
            "models": ["RandomForest", "IsolationForest", "OneClassSVM"]
        },
        "stage_3b": {
            "name": "Deep Learning (Autoencoder/LSTM)",
            "input": "(154, 64, 1) 3D Array (KEEP 3D)",
            "output_forward": "Reconstruction (same shape as reshaped input)",
            "output_error": "(27,) - Reconstruction Error",
            "models": ["Autoencoder", "LSTM"]
        }
    },
    
    "models": {
        "RandomForest": {
            "type": "Classical ML - Supervised",
            "requires_labels": True,
            "input_shape": "(154, 64)",
            "input_dtype": "float32/64",
            "input_range": "Normalized [-3, 3]",
            "batch_processing": "Full batch",
            
            "predict_output": {
                "shape": "(27,)",
                "dtype": "int64",
                "values": "[0, 1]",
                "meaning": "Class labels (0=normal, 1=anomaly)"
            },
            
            "predict_proba_output": {
                "shape": "(27, 2)",
                "dtype": "float64",
                "values": "[0.0 ~ 1.0]",
                "meaning": "Probability for each class [P(normal), P(anomaly)]"
            },
            
            "metrics": {
                "accuracy": "float [0.0 ~ 1.0]",
                "precision": "float [0.0 ~ 1.0]",
                "recall": "float [0.0 ~ 1.0]",
                "f1": "float [0.0 ~ 1.0]",
                "roc_auc": "float [0.0 ~ 1.0]"
            },
            
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 15,
                "random_state": 42
            },
            
            "json_output": {
                "model_name": "RandomForest",
                "timestamp": "ISO 8601",
                "metrics": "dict (see metrics above)",
                "confusion_matrix": "list [[TN, FP], [FN, TP]]",
                "predictions_count": "int",
                "probabilities": {"shape": "[27, 2]", "min": 0.0, "max": 1.0, "mean": 0.5}
            }
        },
        
        "IsolationForest": {
            "type": "Classical ML - Unsupervised",
            "requires_labels": False,
            "input_shape": "(154, 64)",
            "input_dtype": "float32/64",
            "input_range": "Normalized [-3, 3]",
            "batch_processing": "Full batch",
            
            "predict_output": {
                "shape": "(27,)",
                "dtype": "int64",
                "values": "[1, -1] → Converted to [0, 1]",
                "meaning": "Class labels (1=normal→0, -1=anomaly→1)"
            },
            
            "decision_function": {
                "shape": "(27,)",
                "dtype": "float64",
                "values": "[-∞ ~ ∞]",
                "meaning": "Anomaly score (lower = more anomalous)"
            },
            
            "metrics": {
                "Lower performance than RF": "60~70% accuracy"
            },
            
            "hyperparameters": {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42
            }
        },
        
        "OneClassSVM": {
            "type": "Classical ML - Unsupervised",
            "requires_labels": False,
            "input_shape": "(154, 64)",
            "input_dtype": "float32/64",
            "input_range": "Normalized [-3, 3]",
            "batch_processing": "Full batch",
            
            "predict_output": {
                "shape": "(27,)",
                "dtype": "int64",
                "values": "[1, -1] → Converted to [0, 1]",
                "meaning": "Class labels (1=normal→0, -1=anomaly→1)"
            },
            
            "decision_function": {
                "shape": "(27,)",
                "dtype": "float64",
                "values": "[-∞ ~ ∞]",
                "meaning": "Distance from hyperplane"
            },
            
            "metrics": {
                "Lower performance than RF": "40~60% accuracy (needs tuning)"
            },
            
            "hyperparameters": {
                "kernel": "rbf",
                "gamma": "auto",
                "nu": 0.05
            }
        },
        
        "Autoencoder": {
            "type": "Deep Learning - Unsupervised",
            "requires_labels": False,
            "input_shape": "(154, 64, 1)",
            "input_reshape": "(154, 64, 1) → (154, 64) before forward pass",
            "input_dtype": "float32",
            "input_range": "Normalized [-3, 3]",
            "batch_processing": "Batches of 16 (configurable)",
            
            "architecture": {
                "encoder": "Linear(64→128)→ReLU→Linear(128→64)→ReLU→Linear(64→32)",
                "decoder": "Linear(32→64)→ReLU→Linear(64→128)→ReLU→Linear(128→64)",
                "bottleneck_dim": 32,
                "loss": "MSE (Mean Squared Error)"
            },
            
            "forward_output": {
                "shape": "(batch_size, 64)",
                "dtype": "torch.float32",
                "values": "Reconstructed input values",
                "example": "tensor([[-0.0065, -0.0184, ...], ...])"
            },
            
            "reconstruction_error": {
                "calculation": "mean((input - output)^2) for each sample",
                "shape": "(27,)",
                "dtype": "float64",
                "values": "[0.0 ~ ∞]",
                "interpretation": {
                    "low": "< threshold → normal (0)",
                    "high": "> threshold → anomaly (1)",
                    "threshold": "95-percentile"
                }
            },
            
            "metrics": {
                "accuracy": "float [0.0 ~ 1.0]",
                "precision": "float [0.0 ~ 1.0]",
                "recall": "float [0.0 ~ 1.0]",
                "f1": "float [0.0 ~ 1.0]",
                "NO_roc_auc": "Not applicable for reconstruction error"
            },
            
            "training_config": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "device": "CPU or GPU"
            },
            
            "json_output": {
                "model_name": "Autoencoder",
                "timestamp": "ISO 8601",
                "metrics": "dict",
                "confusion_matrix": "list",
                "predictions_count": 27,
                "reconstruction_error": {
                    "threshold": "float",
                    "min": "float",
                    "max": "float",
                    "mean": "float",
                    "std": "float",
                    "percentile_95": "float"
                }
            }
        },
        
        "LSTM": {
            "type": "Deep Learning - Unsupervised",
            "requires_labels": False,
            "importance": "KEEP 3D - DO NOT FLATTEN",
            "input_shape": "(154, 64, 1)",
            "input_dtype": "float32",
            "input_range": "Normalized [-3, 3]",
            "batch_processing": "Batches of 16 (configurable)",
            
            "architecture": {
                "lstm_layers": 2,
                "input_size": 1,
                "hidden_size": 64,
                "batch_first": True,
                "dropout": 0.2,
                "fc_layer": "Linear(64 → 1)"
            },
            
            "forward_pass": {
                "step1": "(batch_size, 64, 1) → LSTM",
                "lstm_output": "(batch_size, 64, 64) [64 hidden dims]",
                "step2": "Get last hidden state → (batch_size, 64)",
                "step3": "FC Layer → (batch_size, 1)"
            },
            
            "forward_output": {
                "shape": "(batch_size, 1)",
                "dtype": "torch.float32",
                "values": "Last time step predictions",
                "example": "tensor([[-0.0145], [0.0087], [-0.0023], ...])"
            },
            
            "reconstruction_error": {
                "calculation": "mean((input_mean - output)^2) for each sample",
                "input_mean": "mean input over sequence dimension",
                "shape": "(27,)",
                "dtype": "float64",
                "values": "[0.0 ~ ∞]",
                "threshold": "95-percentile"
            },
            
            "metrics": {
                "accuracy": "float [0.0 ~ 1.0]",
                "precision": "float [0.0 ~ 1.0]",
                "recall": "float [0.0 ~ 1.0]",
                "f1": "float [0.0 ~ 1.0]",
                "NO_roc_auc": "Not applicable"
            },
            
            "training_config": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "device": "CPU or GPU (slow on CPU)"
            },
            
            "json_output": {
                "model_name": "LSTM",
                "timestamp": "ISO 8601",
                "metrics": "dict",
                "confusion_matrix": "list",
                "predictions_count": 27,
                "reconstruction_error": {
                    "threshold": "float",
                    "min": "float",
                    "max": "float",
                    "mean": "float",
                    "std": "float",
                    "percentile_95": "float"
                }
            }
        }
    },
    
    "summary_table": {
        "headers": ["Model", "Type", "Labels", "Input Shape", "Output Shape", "Performance"],
        "rows": [
            ["RandomForest", "Classical", "Yes", "(154, 64)", "(27,)", "Excellent 90%+"],
            ["IsolationForest", "Classical", "No", "(154, 64)", "(27,)", "Moderate 60-70%"],
            ["OneClassSVM", "Classical", "No", "(154, 64)", "(27,)", "Moderate 40-60%"],
            ["Autoencoder", "Deep Learning", "No", "(154, 64, 1)", "Error: (27,)", "Good 70-80%"],
            ["LSTM", "Deep Learning", "No", "(154, 64, 1) 3D", "Error: (27,)", "Very Good 80-90%"]
        ]
    }
}


if __name__ == "__main__":
    print("=" * 80)
    print("MODEL INPUT/OUTPUT FORMAT QUICK REFERENCE")
    print("=" * 80)
    print(json.dumps(MODEL_IO_SPECS, indent=2, ensure_ascii=False))
