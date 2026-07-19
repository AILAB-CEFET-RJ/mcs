"""Reconstrói os oito datasets após as correções metodológicas.

Os datasets FULL são sempre construídos antes dos CASEONLY correspondentes,
pois estes últimos são filtrados pelo suporte de datas e unidades do FULL.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIRECTORY = PROJECT_ROOT / "runs" / "dataset_build_method_corrected_v2"

CONFIGS = [
    "config/config_rj_daily.yaml",
    "config/config_rj_daily_casesonly.yaml",
    "config/config_natal_daily.yaml",
    "config/config_natal_daily_casesonly.yaml",
    "config/config_rj_weekly.yaml",
    "config/config_rj_weekly_casesonly.yaml",
    "config/config_natal_weekly.yaml",
    "config/config_natal_weekly_casesonly.yaml",
]


def run_build(config: str) -> None:
    config_path = PROJECT_ROOT / config
    log_path = LOG_DIRECTORY / f"{config_path.stem}.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "data_handling" / "build_dataset.py"),
        "--config",
        str(config_path),
    ]

    print(f"\n=== Construindo {config} ===", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
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
            f"Falha ao construir {config}. Consulte o log: {log_path}"
        )


def main() -> None:
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        run_build(config)

    print(f"\nConstrução corrigida concluída para os {len(CONFIGS)} datasets.")
    print(f"Logs: {LOG_DIRECTORY}")


if __name__ == "__main__":
    main()
