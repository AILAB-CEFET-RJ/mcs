"""Executa os baselines sobre os oito datasets metodologicamente corrigidos."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIRECTORY = PROJECT_ROOT / "runs" / "stage1_baseline_method_corrected_v2"
OUTPUT_TAG = "BASELINE_METHOD_CORRECTED_V2"

CONFIGS = [
    "config/train_rj_daily.yaml",
    "config/train_rj_daily_casesonly.yaml",
    "config/train_natal_daily.yaml",
    "config/train_natal_daily_casesonly.yaml",
    "config/train_rj_weekly.yaml",
    "config/train_rj_weekly_casesonly.yaml",
    "config/train_natal_weekly.yaml",
    "config/train_natal_weekly_casesonly.yaml",
]


def run_baseline(config: str) -> None:
    config_path = PROJECT_ROOT / config
    if not config_path.is_file():
        raise SystemExit(f"Configuração não encontrada: {config_path}")

    log_path = LOG_DIRECTORY / f"{config_path.stem}.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train_pipeline.py"),
        "--config",
        str(config_path),
        "--output-tag",
        OUTPUT_TAG,
    ]
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"

    print(f"\n=== Baseline: {config} ===", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        raise SystemExit(
            f"Falha no baseline {config}. Consulte o log: {log_path}"
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        run_baseline(config)

    print(f"\nBaselines concluídos para as {len(CONFIGS)} configurações.")
    print(f"Tag dos resultados: {OUTPUT_TAG}")
    print(f"Logs: {LOG_DIRECTORY}")


if __name__ == "__main__":
    main()
