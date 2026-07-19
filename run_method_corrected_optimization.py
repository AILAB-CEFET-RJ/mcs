"""Executa a otimização e os treinamentos finais metodologicamente corrigidos."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "experiments.yaml"
LOG_DIRECTORY = PROJECT_ROOT / "runs" / "stage3_method_corrected_v2"
LOG_PATH = LOG_DIRECTORY / "optimization_pipeline.log"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if not CONFIG_PATH.is_file():
        raise SystemExit(f"Configuração não encontrada: {CONFIG_PATH}")

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "optimization_pipeline.py"),
        "--config",
        str(CONFIG_PATH),
    ]

    print("=== Otimização dos 24 experimentos corrigidos ===", flush=True)
    print(f"Configuração: {CONFIG_PATH}", flush=True)
    print(f"Log: {LOG_PATH}", flush=True)

    with LOG_PATH.open("w", encoding="utf-8") as log_file:
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
            f"A otimização falhou com código {return_code}. Consulte: {LOG_PATH}"
        )

    print("\nOtimização e treinamentos finais concluídos.")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
