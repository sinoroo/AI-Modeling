"""
MLflow 파일스토어 -> SQLite 마이그레이션 자동 스크립트

이 스크립트는:
1. 기존 mlruns 데이터 확인
2. SQLite로 마이그레이션
3. 마이그레이션 검증
3. 백업 생성
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# 색상 정의
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(msg: str, status: str = "INFO"):
    """상태 메시지 출력."""
    if status == "SUCCESS":
        print(f"{Colors.GREEN}✓ {msg}{Colors.END}")
    elif status == "WARNING":
        print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")
    elif status == "ERROR":
        print(f"{Colors.RED}✗ {msg}{Colors.END}")
    else:  # INFO
        print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def check_mlruns_exists() -> bool:
    """mlruns 디렉토리 존재 확인."""
    if os.path.isdir("mlruns"):
        print_status("mlruns 디렉토리 찾음", "INFO")
        size = sum(
            f.stat().st_size for f in Path("mlruns").rglob("*") if f.is_file()
        )
        size_mb = size / (1024 * 1024)
        print_status(f"데이터 크기: {size_mb:.2f} MB", "INFO")
        return True
    else:
        print_status("mlruns 디렉토리 없음 (새로 생성됨)", "WARNING")
        return False

def backup_mlruns():
    """mlruns 백업 생성."""
    if not os.path.isdir("mlruns"):
        print_status("백업할 mlruns 없음", "INFO")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"mlruns_backup_{timestamp}"
    
    try:
        shutil.copytree("mlruns", backup_dir)
        print_status(f"백업 생성됨: {backup_dir}", "SUCCESS")
    except Exception as e:
        print_status(f"백업 생성 실패: {e}", "ERROR")

def migrate_to_sqlite():
    """MLflow 마이그레이션 실행."""
    print_status("MLflow 파일스토어 -> SQLite 마이그레이션 시작...", "INFO")
    
    # mlruns 없으면 스킵
    if not os.path.isdir("mlruns"):
        print_status("mlruns 없음, 마이그레이션 스킵", "WARNING")
        return True
    
    try:
        # 마이그레이션 명령 실행
        result = subprocess.run(
            [
                "mlflow", "migrate-filestore",
                "-c", "sqlite:///../integration/mlflow.db",
                "../integration/mlruns"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_status("마이그레이션 완료", "SUCCESS")
            print(result.stdout)
            return True
        else:
            print_status(f"마이그레이션 실패: {result.stderr}", "ERROR")
            return False
            
    except FileNotFoundError:
        print_status("mlflow 명령어를 찾을 수 없음", "ERROR")
        print_status("MLflow가 설치되어 있는지 확인하세요: pip install mlflow", "INFO")
        return False
    except Exception as e:
        print_status(f"마이그레이션 실패: {e}", "ERROR")
        return False

def verify_migration():
    """마이그레이션 검증."""
    print_status("마이그레이션 검증 중...", "INFO")
    
    try:
        import mlflow
        
        # SQLite 연결
        mlflow.set_tracking_uri("sqlite:///../integration/mlflow.db")
        
        # 실험 목록 조회
        experiments = mlflow.search_experiments()
        print_status(f"실험 개수: {len(experiments)}", "SUCCESS")
        
        # 각 실험의 실행 확인
        total_runs = 0
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            total_runs += len(runs)
        
        print_status(f"총 실행 개수: {total_runs}", "SUCCESS")
        
        if len(experiments) > 0 or total_runs > 0:
            print_status("✓ 마이그레이션 검증 성공", "SUCCESS")
            return True
        else:
            print_status("⚠ 데이터가 없음 (새로 시작됨)", "WARNING")
            return True
            
    except Exception as e:
        print_status(f"검증 실패: {e}", "ERROR")
        return False

def check_sqlite_db():
    """SQLite 데이터베이스 확인."""
    if os.path.exists("../integration/mlflow.db"):
        size = os.path.getsize("../integration/mlflow.db")
        size_mb = size / (1024 * 1024)
        print_status(f"SQLite DB 파일 크기: {size_mb:.2f} MB", "SUCCESS")
        return True
    else:
        print_status("SQLite DB 파일 생성됨", "INFO")
        return False

def main():
    """메인 함수."""
    print("\n" + "=" * 70)
    print("MLflow 파일스토어 -> SQLite 마이그레이션")
    print("=" * 70)
    
    # Step 1: 현재 상태 확인
    print("\n[Step 1] 현재 상태 확인...")
    has_mlruns = check_mlruns_exists()
    has_db = check_sqlite_db()
    
    # Step 2: 백업 생성
    if has_mlruns:
        print("\n[Step 2] 백업 생성 중...")
        backup_mlruns()
    
    # Step 3: 마이그레이션
    if has_mlruns:
        print("\n[Step 3] 마이그레이션 실행 중...")
        migration_success = migrate_to_sqlite()
        
        if not migration_success:
            print_status("마이그레이션에 실패했습니다", "ERROR")
            print_status("다음을 시도해보세요:", "INFO")
            print("  1. MLflow 재설치: pip install --upgrade mlflow")
            print("  2. 수동 마이그레이션: mlflow migrate-filestore -c sqlite:///../integration/mlflow.db ../integration/mlruns")
            return 1
    
    # Step 4: 검증
    print("\n[Step 4] 마이그레이션 검증...")
    if not verify_migration():
        print_status("검증 실패", "ERROR")
        return 1
    
    # Step 5: 완료 메시지
    print("\n" + "=" * 70)
    print_status("마이그레이션 완료!", "SUCCESS")
    print("=" * 70)
    
    print("\n다음으로 할 일:")
    print("  1. MLflow UI 시작: mlflow ui")
    print("  2. 브라우저에서 확인: http://localhost:5000")
    print("  3. 모델 학습 실행: python main.py --train")
    
    print("\n백업 폴더:")
    for item in os.listdir("."):
        if item.startswith("mlruns_backup_"):
            print(f"  - {item}")
    
    print("\n✨ 모든 준비가 완료되었습니다!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
