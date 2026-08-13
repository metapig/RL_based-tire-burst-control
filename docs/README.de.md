# Fahrzeugstabilisierung nach Reifenplatzer mit Reinforcement Learning

[中文](../README.md) | [English](README.en.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)

Dieses Projekt trainiert mit CarSim und PyTorch einen PPO-Regler mit kontinuierlichem Aktionsraum. Ziel ist die Fahrstabilisierung und Spurhaltung nach einem Reifenplatzer. Anfangsgeschwindigkeit, Reibwert, Fahrzeugmasse, betroffenes Rad und Schadensgrad werden randomisiert. Beim Training werden privilegierte Umgebungsinformationen verwendet; ein zusätzlicher History Encoder schätzt deren latente Darstellung aus vergangenen Zuständen.

> CarSim, DLLs, `simfile.sim`, lokale Datenbankdateien, Modellgewichte und TensorBoard-Protokolle sind nicht Bestandteil dieses Repositories. Für die vollständige Simulation sind eine gültige CarSim-Installation und eine lokale Konfiguration erforderlich.

## Methode

- **Algorithmus:** PPO mit GAE und fünf kontinuierlichen Aktionen.
- **Aktionen:** Vorderradlenkwinkel und vier Radmomente mit den Skalen `(3, 400, 400, 400, 400)`.
- **Beschränkte Policy:** Gaußsche latente Aktionen werden mit `tanh` begrenzt und physikalisch skaliert. Die pre-tanh Aktionen werden für konsistente PPO-Log-Wahrscheinlichkeiten gespeichert.
- **Exploration:** ein globaler lernbarer `log_std` je Aktionsdimension mit festgelegten Grenzen.
- **Privilegiertes Training:** Reifenposition, Schadensgrad, Reibwert und Masse werden in eine latente Variable codiert.
- **Historische Adaption:** Ein 1-D-CNN schätzt die privilegierte latente Variable aus 25 vergangenen Zuständen.
- **Curriculum:** Geschwindigkeitsbereich, niedrige Reibwerte und Schadensgrade werden mit der Episodenzahl erweitert.

## Verzeichnisstruktur

```text
tire_burst/
├── generated_tire_tables/     # Fx/Fy/Mz-Tabellen für 25/50/75/100 %
├── scripts/                   # Training, Datensammlung und Vergleiche
├── src/
│   ├── agents/                # PPO und History Encoder
│   ├── envs/                  # CarSim-Wrapper und Reifenplatzer-Umgebung
│   ├── models/                # Zustand, Reward und Parameter
│   ├── mpc/                   # LPV-MPC-Referenz
│   └── utils/                 # VS Solver und Reifentabellen-Werkzeuge
├── tests/
├── docs/
├── TRAINING_COMMANDS.md
└── README.md
```

Lokale Läufe erzeugen zusätzlich `checkpoints/`, `datasets/`, `runs/` und `simfile.sim`. Diese Dateien werden durch `.gitignore` ausgeschlossen.

## Voraussetzungen

- Windows und Python 3.10+
- CarSim 2023; die aktuelle Integration orientiert sich an CarSim 2023.2
- PyTorch, NumPy und TensorBoard
- CasADi für das LPV-MPC-Modul
- `pefile`, `tomlkit` und `pytest`

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy torch tensorboard casadi pefile tomlkit pytest
```

Die Abhängigkeitsversionen sind noch nicht fixiert. Für reproduzierbare Experimente sollten Python-, PyTorch-, CUDA-, CarSim- und Paketversionen dokumentiert werden.

## CarSim-Konfiguration

1. Eine gültige `simfile.sim` im Projektstamm ablegen.
2. Die lokalen absoluten Pfade in folgenden Dateien anpassen:
   - `src/envs/tire_burst_env.py`
   - `src/utils/generate_burst_force_tables.py`
   - `src/utils/get_dll_functions.py`
3. Die Simulation muss 6 Importkanäle und mindestens 8 Exportvariablen in der erwarteten Reihenfolge bereitstellen.
4. Vor dem Training die CarSim-Datenbank sichern, da ein Reset die ausgewählten Reifentabellen in lokale Datensätze schreibt.

Die Pfade in `generated_tire_tables/manifest.json` dokumentieren nur die Herkunft der Daten und sind keine portable Konfiguration.

## Schnellstart

```powershell
python -m pytest -q
python -m scripts.main --mode train --max_episodes 4000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_squashed.pth"
python -m scripts.main --mode resume --max_episodes 1000 --checkpoint_interval 100 --checkpoint "checkpoints\ppo_squashed.pth"
python -m scripts.main --mode test --checkpoint "checkpoints\ppo_squashed.pth"
```

Weitere Befehle für Datensammlung, History-Encoder, paarweise Auswertung, TensorBoard und Reifentabellen stehen in [TRAINING_COMMANDS.md](../TRAINING_COMMANDS.md).

## History-Adaption

```powershell
python -m scripts.main --mode collect_history_data --checkpoint "checkpoints\ppo_squashed.pth" --adaptation_episodes 500 --dataset "datasets\adaptation.pt"
python -m scripts.train_history_encoder --train --dataset "datasets\adaptation.pt" --checkpoint "checkpoints\history_encoder.pth" --epochs 30 --batch_size 64 --learning_rate 0.0005
python -m scripts.compare_oracle_history --checkpoint "checkpoints\ppo_squashed.pth" --history_checkpoint "checkpoints\history_encoder.pth" --episodes 20
```

## Reward und statistische Interpretation

`src/models/reward_model.py` kombiniert Geschwindigkeit, Spurhaltung, Stabilität, Stellaufwand, Aktionsglätte und abnormalen Abbruch. Wird der Reward geändert und das Training fortgesetzt, entsteht ein neues Optimierungsziel mit der alten Policy als Initialisierung. Returns aus verschiedenen Reward-Definitionen dürfen nicht direkt verglichen werden.

Für die Bewertung sollte ein fester, unabhängiger und gepaarter Szenariosatz verwendet werden. Relevante Größen sind Spurfehler, Gierrate, Quergeschwindigkeit, RMS- und Sättigungswerte der Aktionen, Ausfallrate und Randverhalten.

## Ignorierte Dateien, Status und Lizenz

TensorBoard-Läufe, Checkpoints, Datensätze, `.sim/.par/.dll`-Dateien, Caches, virtuelle Umgebungen und lokale Editor-Dateien werden nicht versioniert. Modelle sollten gemeinsam mit Commit, Reward-Version, Trainingsbefehl und Bewertung über Releases, Git LFS oder ein Modellregister veröffentlicht werden.

Das Projekt ist ein Forschungsprototyp und keine serienreife Fahrzeugsoftware. Für reale Versuche sind unabhängige Sicherheitsmechanismen, Verifikation und Hardware-in-the-Loop-Tests erforderlich.

Derzeit ist keine Lizenz enthalten. Vor einer öffentlichen Weitergabe sollte eine passende `LICENSE` ergänzt werden; ohne Lizenz bleiben alle Rechte beim Autor.
