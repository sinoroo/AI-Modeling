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
            "name": "전처리 및 특성 생성",
            "input": "DataFrame (4990, 3)",
            "output": "3D Numpy Array (n_windows, 64, 6)",
            "operations": ["결측치 처리", "이상치 제거", "윈도우 생성", "통계 특성 생성"],
            "example": "(4990, 3) → (154, 64, 6) [vibration, RMS, Peak, Crest, Kurtosis, Skewness]"
        },
        "stage_3a": {
            "name": "Classical ML (RF/IF/SVM)",
            "input": "(154, 64, 6) 3D Array",
            "transformation": "Flatten → (154, 384) 2D Array (64 samples × 6 features)",
            "output_predict": "(27,) - 클래스 레이블",
            "output_proba": "(27, 2) - 확률",
            "models": ["RandomForest", "IsolationForest", "OneClassSVM"]
        },
        "stage_3b": {
            "name": "Deep Learning (Autoencoder/LSTM)",
            "input": "(154, 64, 6) 3D Array (KEEP 3D)",
            "output_forward": "Reconstruction (same shape as input)",
            "output_error": "(27,) - Reconstruction Error",
            "models": ["Autoencoder", "LSTM"]
        }
    },
    
    "models": {
        "RandomForest": {
            "type": "Classical ML - Supervised",
            "requires_labels": True,
            "input_shape": "(154, 384)",
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
