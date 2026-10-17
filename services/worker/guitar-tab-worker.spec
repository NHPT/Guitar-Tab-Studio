import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


worker_root = Path(SPECPATH).resolve()
hidden_imports = []
runtime_hooks = []
packages = ["basic_pitch", "demucs", "stringtrace_ml", "yt_dlp"]
if os.environ.get("GTS_COMMERCIAL_BUILD") != "1":
    packages.append("muscriptor")
for package in packages:
    hidden_imports.extend(collect_submodules(package))
if os.environ.get("GTS_COMMERCIAL_BUILD") != "1":
    hidden_imports.extend(
        [
            "mt3_infer.adapters.mr_mt3",
            "mt3_infer.adapters.yourmt3",
            "mt3_infer.adapters.vocab_utils",
            "mt3_infer.models.mr_mt3.t5",
            "mt3_infer.models.yourmt3.inference_loader",
        ]
    )
    runtime_hooks.append(str(worker_root / "pyinstaller_mr_mt3_runtime.py"))
excluded_packages = ["matplotlib", "notebook", "pytest"]
if os.environ.get("GTS_COMMERCIAL_BUILD") == "1":
    excluded_packages.extend(
        [
            "muscriptor",
            "mt3_infer",
        ]
    )
else:
    excluded_packages.extend(
        [
            "mt3_infer.adapters.mt3_pytorch",
            "mt3_infer.models.mt3_pytorch",
        ]
    )

analysis = Analysis(
    [str(worker_root / "worker.py")],
    pathex=[str(worker_root)],
    binaries=[],
    datas=collect_data_files("basic_pitch"),
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=excluded_packages,
    noarchive=False,
)

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="guitar-tab-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="guitar-tab-worker",
)
