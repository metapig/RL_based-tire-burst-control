# 강화학습 기반 차량 타이어 파열 안정화 제어

[中文](../README.md) | [English](README.en.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)

이 프로젝트는 CarSim과 PyTorch를 사용하여 타이어 파열 이후 차량의 궤적과 안정성을 제어하는 연속 행동 PPO 정책을 학습합니다. 초기 속도, 노면 마찰계수, 차량 질량, 파열 타이어 위치 및 파열 정도를 무작위화합니다. 현재 구조는 학습 시 특권 환경 정보를 사용하며, 최근 상태 이력만으로 동일한 잠재 표현을 추정하는 history encoder를 제공합니다.

> 이 저장소에는 CarSim, CarSim DLL, `simfile.sim`, 로컬 데이터베이스 파일, 학습된 가중치 또는 TensorBoard 로그가 포함되지 않습니다. 전체 시뮬레이션에는 유효한 CarSim 설치와 로컬 설정이 필요합니다.

## 방법 개요

- **알고리즘:** 5차원 연속 제어를 위한 PPO + GAE.
- **행동:** 전륜 조향각과 네 바퀴 제어 모멘트. 물리 스케일은 `(3, 400, 400, 400, 400)`입니다.
- **유계 정책:** 가우시안 잠재 행동에 `tanh`와 물리 스케일을 적용합니다. 일관된 PPO log-prob 계산을 위해 pre-tanh 행동을 저장합니다.
- **탐색:** 행동 차원별 전역 학습 가능 `log_std`를 사용하며 설정 범위로 제한합니다.
- **특권 학습:** 파열 위치/정도, 마찰계수 및 차량 질량을 잠재 변수로 인코딩합니다.
- **이력 적응:** 1-D CNN이 25 스텝 상태 이력으로부터 특권 잠재 변수를 예측합니다.
- **커리큘럼:** 에피소드가 증가함에 따라 속도, 저마찰 및 파열 정도 범위를 확장합니다.

## 저장소 구조

```text
tire_burst/
├── generated_tire_tables/     # 25/50/75/100% Fx/Fy/Mz 파열 테이블
├── scripts/                   # 학습, 데이터 수집 및 비교 진입점
├── src/
│   ├── agents/                # PPO 및 history encoder
│   ├── envs/                  # CarSim 래퍼와 타이어 파열 환경
│   ├── models/                # 상태, reward 및 파라미터
│   ├── mpc/                   # LPV-MPC 참조 구현
│   └── utils/                 # VS Solver 및 타이어 테이블 도구
├── tests/
├── docs/
├── TRAINING_COMMANDS.md
└── README.md
```

로컬 실행 시 `checkpoints/`, `datasets/`, `runs/`, `simfile.sim`이 추가로 생성되며 모두 `.gitignore`에서 제외됩니다.

## 요구 사항

- Windows 및 Python 3.10+
- CarSim 2023. 현재 통합 코드는 CarSim 2023.2 인터페이스를 기준으로 합니다.
- PyTorch, NumPy, TensorBoard
- LPV-MPC 모듈용 CasADi
- `pefile`, `tomlkit`, `pytest`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy torch tensorboard casadi pefile tomlkit pytest
```

아직 의존성 버전이 고정되어 있지 않습니다. 재현 가능한 실험을 위해 Python, PyTorch, CUDA, CarSim 및 패키지 버전을 기록하십시오.

## CarSim 로컬 설정

1. 유효한 `simfile.sim`을 프로젝트 루트에 배치합니다.
2. 다음 파일의 개발 PC 절대 경로를 로컬 환경에 맞게 수정합니다.
   - `src/envs/tire_burst_env.py`
   - `src/utils/generate_burst_force_tables.py`
   - `src/utils/get_dll_functions.py`
3. 시뮬레이션이 예상 순서의 import 채널 6개와 export 변수 최소 8개를 제공하는지 확인합니다.
4. 환경 reset은 선택된 파열 정도 테이블을 설정된 CarSim 타이어 데이터셋에 기록하므로 먼저 데이터베이스를 백업하십시오.

`generated_tire_tables/manifest.json`의 경로는 데이터 출처 기록이며 이식 가능한 설정값이 아닙니다.

## 빠른 시작

```powershell
python -m pytest -q
python -m scripts.main --mode train --max_episodes 4000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_squashed.pth"
python -m scripts.main --mode resume --max_episodes 1000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_squashed.pth"
python -m scripts.main --mode test --checkpoint "checkpoints\ppo_squashed.pth"
```

데이터 수집, history encoder 학습, paired 평가, TensorBoard 및 타이어 테이블 명령은 [TRAINING_COMMANDS.md](../TRAINING_COMMANDS.md)를 참고하십시오.

## 이력 적응

```powershell
python -m scripts.main --mode collect_history_data --checkpoint "checkpoints\ppo_squashed.pth" --adaptation_episodes 500 --dataset "datasets\adaptation.pt"
python -m scripts.train_history_encoder --train --dataset "datasets\adaptation.pt" --checkpoint "checkpoints\history_encoder.pth" --epochs 30 --batch_size 64 --learning_rate 0.0005
python -m scripts.compare_oracle_history --checkpoint "checkpoints\ppo_squashed.pth" --history_checkpoint "checkpoints\history_encoder.pth" --episodes 20
```

## Reward와 통계적 해석

`src/models/reward_model.py`는 속도, 궤적, 안정성, 제어 크기, 행동 평활성 및 비정상 종료 성분을 결합합니다. reward를 변경한 뒤 학습을 재개하면 이전 정책을 초기값으로 사용하는 새로운 최적화 목표가 됩니다. 서로 다른 reward로 계산된 episode return을 직접 비교하면 안 됩니다.

평가는 고정되고 독립적인 paired 시나리오 집합에서 수행해야 합니다. 궤적 오차, 요 응답, 횡속도, 행동 RMS/포화율, 실패율 및 꼬리 성능을 비교하십시오. 현재 진입점은 완전한 명령행 seed 제어를 제공하지 않습니다.

## 제외 파일, 상태 및 라이선스

TensorBoard 로그, checkpoint, 데이터셋, `.sim/.par/.dll` CarSim 파일, 캐시, 가상환경 및 편집기 메타데이터는 버전 관리에서 제외됩니다. 모델을 공개할 때는 GitHub Release, Git LFS 또는 외부 모델 레지스트리를 사용하고 정확한 commit, reward 버전, 학습 명령 및 평가 결과를 함께 제공하십시오.

이 프로젝트는 연구용 프로토타입이며 양산 차량 제어 소프트웨어가 아닙니다. 실차 실험에는 독립적인 안전 제약, 고장 안전 장치, 검증 및 HIL 테스트가 필요합니다.

현재 라이선스가 포함되어 있지 않습니다. 공개 배포 전 적절한 `LICENSE`를 추가하십시오. 라이선스가 없으면 저작권은 저자에게 유지됩니다.
