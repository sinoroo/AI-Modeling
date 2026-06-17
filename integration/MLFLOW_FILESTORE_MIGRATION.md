# MLflow 파일스토어 유지보수 모드 가이드

## 🔴 **에러 메시지**

```
mlflow.exceptions.MlflowException: The filesystem tracking backend (e.g., './mlruns') 
is in maintenance mode and will not receive further updates. Please migrate to a 
database backend (e.g., 'sqlite:///mlflow.db') to access the latest MLflow features.
```

---

## 📝 **문제 설명**

### MLflow 2.0 이후 변경사항

MLflow는 파일시스템 기반 백엔드(`./mlruns`)의 **유지보수 모드** 전환 결정했습니다.

**원인:**
- 파일시스템 방식은 **확장성, 동시성, 성능** 면에서 제한적
- 데이터베이스 백엔드가 더 강력하고 안정적
- 새로운 기능들은 데이터베이스 백엔드에서만 작동

---

## ✅ **해결 방법**

### **방법 1: 환경변수 설정 (임시방편)**

가장 빠른 해결책이지만 임시방편입니다.

#### Windows PowerShell
```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
python main.py --train
```

#### Windows Command Prompt
```cmd
set MLFLOW_ALLOW_FILE_STORE=true
python main.py --train
```

#### Linux/Mac
```bash
export MLFLOW_ALLOW_FILE_STORE=true
python main.py --train
```

**장점:** 가장 간단, 기존 데이터 유지
**단점:** 임시방편, 향후 지원 안될 수 있음

---

### **방법 2: SQLite로 마이그레이션 (권장)** ⭐ 

이 프로젝트에 이미 적용되었습니다.

#### Step 1: 기존 데이터 마이그레이션

```bash
mlflow migrate-filestore -c sqlite:///mlflow.db ./mlruns
```

**출력 예:**
```
Migrating experiment and run datasets from ./mlruns to sqlite:///mlflow.db
Migrating Experiment id 0...
✓ Migration complete
```

#### Step 2: 코드 변경 (이미 적용됨) ✅

```python
# 이전
mlflow.set_tracking_uri("file:./mlruns")

# 현재 (적용됨)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

#### Step 3: MLflow UI 실행

```bash
# SQLite 백엔드 사용
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 또는 (더 간단)
mlflow ui
# 기본값이 sqlite:///mlflow.db로 작동
```

**장점:**
- 최신 MLflow 기능 지원
- 더 빠른 성능
- 동시 접근 안전
- 프로덕션 권장

**단점:**
- 마이그레이션 필요 (한 번만)

---

### **방법 3: 원격 MLflow 서버 (선택사항)**

프로덕션 환경에서는 원격 서버 사용 권장:

```python
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

---

## 🚀 **빠른 마이그레이션 가이드**

### 단계 1: 기존 데이터 확인
```bash
ls -la mlruns/
# 또는 (Windows)
dir mlruns
```

### 단계 2: 마이그레이션 실행
```bash
mlflow migrate-filestore -c sqlite:///mlflow.db ./mlruns
```

### 단계 3: 마이그레이션 확인
```bash
ls -la
# mlflow.db 파일이 생성됨
```

### 단계 4: MLflow UI 확인
```bash
mlflow ui
# http://localhost:5000 에서 모든 데이터가 보임
```

---

## 📊 **비교 분석**

| 항목 | 파일시스템 | SQLite | 원격 서버 |
|------|---------|--------|--------|
| **설정 복잡도** | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| **성능** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **확장성** | ❌ | ✅ | ✅ |
| **동시 접근** | ❌ | ✅ | ✅ |
| **최신 기능** | ❌ | ✅ | ✅ |
| **권장 용도** | 테스트 | 개발/소규모 | 프로덕션 |
| **유지보수** | ❌ | ✅ | ✅ |

---

## 💾 **파일 구조 변경**

### 마이그레이션 전
```
project/
├── mlruns/           # 여러 폴더와 파일들
│   ├── 0/
│   ├── 1/
│   └── ...
└── main.py
```

### 마이그레이션 후
```
project/
├── mlflow.db         # SQLite 데이터베이스 (단일 파일)
├── mlruns/           # (선택적으로 보관 가능)
│   ├── 0/
│   ├── 1/
│   └── ...
└── main.py
```

---

## 🔍 **데이터 검증**

마이그레이션이 성공했는지 확인:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 실험 확인
experiments = mlflow.search_experiments()
print(f"Experiments: {len(experiments)}")

# 실행 확인
runs = mlflow.search_runs(experiment_ids=["0"])
print(f"Runs: {len(runs)}")

# 각 실행의 메트릭 확인
for run in runs[:3]:
    print(f"Run {run.info.run_id}: {run.data.metrics}")
```

---

## ❓ **자주 묻는 질문**

### Q1: 기존 데이터가 손실되나요?
**A:** 아니요, `mlflow migrate-filestore` 명령으로 손실 없이 마이그레이션됩니다.

### Q2: SQLite 파일이 손상되면?
**A:** 백업을 해두는 것이 좋습니다:
```bash
cp mlflow.db mlflow.db.backup
```

### Q3: 기존 mlruns 폴더는 삭제해도 되나요?
**A:** 안전한 백업 후 삭제 가능:
```bash
mv mlruns mlruns_backup
```

### Q4: 다중 사용자 환경에서는?
**A:** SQLite도 기본적으로 지원하지만, 동시 쓰기가 많으면 원격 서버 추천:
```python
mlflow.set_tracking_uri("postgresql://user:pass@localhost/mlflow")
```

---

## 🛠️ **트러블슈팅**

### 에러: "Database is locked"
```python
# SQLite 리셋
import os
os.remove("mlflow.db")  # 시작부터 자동 생성됨
```

### 에러: "Cannot connect to SQLite"
```bash
# 권한 확인
ls -la mlflow.db
chmod 644 mlflow.db
```

### 마이그레이션 실패
```bash
# 자세한 로그 보기
mlflow migrate-filestore -c sqlite:///mlflow.db ./mlruns -v
```

---

## 📚 **추가 리소스**

- [MLflow 공식 마이그레이션 가이드](https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store)
- [MLflow SQLite 설명서](https://mlflow.org/docs/latest/tracking.html#backend-stores)
- [MLflow URI 스키마](https://mlflow.org/docs/latest/tracking.html#tracking-uri)

---

## ✨ **이 프로젝트의 변경사항**

이 프로젝트에는 다음과 같이 이미 변경되었습니다:

### 1. mlflow_utils.py
```python
# 기본값을 SQLite로 변경
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

### 2. main.py
```python
# 명시적으로 SQLite 지정
tracking_uri="sqlite:///mlflow.db"
```

### 3. mlflow_bentoml_example.py
```python
# 모든 tracker에 SQLite 적용
tracking_uri="sqlite:///mlflow.db"
```

---

## 🎯 **권장 (현재 프로젝트 상황)**

✅ **현재 설정: SQLite (`sqlite:///mlflow.db`)**

이 프로젝트는 이미 SQLite로 설정되어 있습니다.

### 실행 방법 (그대로 사용 가능)
```bash
python main.py --train
```

### MLflow UI 확인
```bash
mlflow ui
# http://localhost:5000
```

---

**축하합니다! 🎉 MLflow가 최신 표준에 맞게 구성되었습니다.**
