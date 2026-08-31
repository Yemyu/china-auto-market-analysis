#!/usr/bin/env python3
"""Create the project-local, reproducibly constrained Airflow environment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = PROJECT_ROOT / ".airflow-venv"
AIRFLOW_VERSION = "3.3.1"
SUPPORTED_PYTHON = (3, 13)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    args = parser.parse_args()
    if sys.version_info[:2] != SUPPORTED_PYTHON:
        raise RuntimeError(
            f"This lock targets Python {SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}; "
            f"current interpreter is {sys.version_info.major}.{sys.version_info.minor}"
        )
    environment = args.env.resolve()
    if environment == PROJECT_ROOT or PROJECT_ROOT not in environment.parents:
        raise ValueError("The Airflow environment must be a dedicated directory inside the project")
    if not (environment / "bin" / "python").exists():
        run([sys.executable, "-m", "venv", str(environment)])
    python = environment / "bin" / "python"
    constraint_url = (
        "https://raw.githubusercontent.com/apache/airflow/"
        f"constraints-{AIRFLOW_VERSION}/constraints-3.13.txt"
    )
    uv = shutil.which("uv")
    if uv:
        run([
            uv, "pip", "install", "--python", str(python),
            f"apache-airflow=={AIRFLOW_VERSION}", "--constraint", constraint_url,
        ])
    else:
        run([
            str(python), "-m", "pip", "install",
            f"apache-airflow=={AIRFLOW_VERSION}", "--constraint", constraint_url,
        ])
    # Dag parsing only needs this package's stdlib orchestration module. Add
    # src through a .pth file instead of installing project metadata that
    # declares analytical dependencies; those remain in the primary .venv.
    if uv:
        subprocess.run(
            [uv, "pip", "uninstall", "--python", str(python),
             "china-auto-market-analysis"],
            cwd=PROJECT_ROOT,
            check=False,
        )
    else:
        subprocess.run(
            [str(python), "-m", "pip", "uninstall", "-y", "china-auto-market-analysis"],
            cwd=PROJECT_ROOT,
            check=False,
        )
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    import_root = PROJECT_ROOT / ".airflow-pythonpath"
    import_root.mkdir(exist_ok=True)
    package_link = import_root / "china_auto_market"
    if package_link.is_symlink() and package_link.resolve() != PROJECT_ROOT / "src" / "china_auto_market":
        package_link.unlink()
    if not package_link.exists():
        package_link.symlink_to(PROJECT_ROOT / "src" / "china_auto_market", target_is_directory=True)
    Path(purelib, "china_auto_market_project.pth").write_text(
        f"{import_root}\n", encoding="utf-8"
    )
    run([str(python), "-m", "pip", "check"])
    run([str(python), "-c", "import airflow; print(airflow.__version__)"])


if __name__ == "__main__":
    main()
