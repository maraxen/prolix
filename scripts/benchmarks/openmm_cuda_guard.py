#!/usr/bin/env python3
"""Anti-Fallback OpenMM CUDA Platform Sensor, Provenance Recorder, & CUDA MPS Multi-Replica Harness.

Enforces that every OpenMM benchmark invocation runs on genuine OpenMM `CUDA`
(`libOpenMMCUDA.so`) rather than silently falling back to `OpenCL`, `CPU`, or `Reference`.
Also provides a reproducible NVIDIA Multi-Process Service (`CUDA MPS`) multi-replica
benchmark runner (`B = 1, 2, 4, 8, 16, 32` concurrent OpenMM CUDA worker processes
synchronized via `multiprocessing.Barrier` after JIT/PME compilation).
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def find_lib_openmm_cuda_path() -> str | None:
    """Locate `libOpenMMCUDA.so` via `/proc/self/maps`, OpenMM plugin dirs, site-packages, and LD_LIBRARY_PATH."""
    try:
        import openmm as mm

        # 1. Check /proc/self/maps first (proves libOpenMMCUDA.so is mapped in process memory)
        maps_path = Path("/proc/self/maps")
        if maps_path.exists():
            for line in maps_path.read_text(errors="ignore").splitlines():
                if "libOpenMMCUDA" in line:
                    parts = line.split()
                    if len(parts) >= 6 and parts[-1].startswith("/"):
                        return parts[-1]

        candidates: list[Path] = []
        env_plug = os.environ.get("OPENMM_PLUGIN_DIR", "").strip()
        if env_plug:
            for p in env_plug.split(":"):
                if p.strip():
                    candidates.append(Path(p.strip()))
        default_dir = mm.Platform.getDefaultPluginsDirectory()
        if default_dir:
            candidates.append(Path(default_dir))
        mm_file = getattr(mm, "__file__", None)
        if mm_file:
            mm_root = Path(mm_file).resolve().parent
            site_pkgs = mm_root.parent
            candidates.extend(
                [
                    mm_root / "lib" / "plugins",
                    mm_root / "plugins",
                    site_pkgs / "openmm.libs",
                    site_pkgs.parent / "lib" / "plugins",
                    Path(sys.prefix) / "lib" / "plugins",
                ]
            )
            for pkg_dir in site_pkgs.glob("*openmm*"):
                if pkg_dir.is_dir():
                    candidates.append(pkg_dir)
                    candidates.append(pkg_dir / "lib")
                    candidates.append(pkg_dir / "lib" / "plugins")
                    candidates.append(pkg_dir / "plugins")
        for ld_dir in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
            if ld_dir.strip():
                candidates.append(Path(ld_dir.strip()))
                candidates.append(Path(ld_dir.strip()) / "plugins")

        for d in candidates:
            if not d.is_dir():
                continue
            direct = d / "libOpenMMCUDA.so"
            if direct.exists():
                return str(direct.resolve())
            for match in d.rglob("libOpenMMCUDA*.so*"):
                if match.is_file():
                    return str(match.resolve())
    except Exception:
        pass
    found = ctypes.util.find_library("OpenMMCUDA")
    return found


def is_cuda_mps_active() -> bool:
    """Return True if an `nvidia-cuda-mps-control` or `nvidia-cuda-mps-server` process is active."""
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    if os.path.exists(os.path.join(pipe_dir, "control")):
        return True
    try:
        out = subprocess.run(
            ["pgrep", "-f", "nvidia-cuda-mps"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def get_openmm_platform_inventory() -> dict[str, Any]:
    """Return registered OpenMM platforms, plugin load failures, and `libOpenMMCUDA.so` path."""
    try:
        import openmm as mm
    except ImportError as exc:
        return {
            "openmm_version": None,
            "python_executable": sys.executable,
            "default_plugin_dir": None,
            "env_plugin_dir": os.environ.get("OPENMM_PLUGIN_DIR") or None,
            "platforms": [],
            "has_cuda_platform": False,
            "lib_openmm_cuda_path": None,
            "plugin_load_failures": [f"ImportError: {exc}"],
            "mps_active": is_cuda_mps_active(),
        }

    plugin_dir = mm.Platform.getDefaultPluginsDirectory()
    if plugin_dir and os.path.isdir(plugin_dir):
        try:
            mm.Platform.loadPluginsFromDirectory(plugin_dir)
        except Exception:
            pass
    env_plug = os.environ.get("OPENMM_PLUGIN_DIR", "").strip()
    if env_plug and os.path.isdir(env_plug):
        try:
            mm.Platform.loadPluginsFromDirectory(env_plug)
        except Exception:
            pass

    platforms = [
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    ]
    failures = list(mm.Platform.getPluginLoadFailures())
    lib_cuda = find_lib_openmm_cuda_path()
    return {
        "openmm_version": getattr(mm, "__version__", "unknown"),
        "python_executable": sys.executable,
        "default_plugin_dir": plugin_dir,
        "env_plugin_dir": env_plug or None,
        "platforms": platforms,
        "has_cuda_platform": "CUDA" in platforms,
        "lib_openmm_cuda_path": lib_cuda,
        "plugin_load_failures": failures,
        "mps_active": is_cuda_mps_active(),
    }


def ensure_openmm_cuda_installed(
    python_exe: str | None = None,
    auto_repair: bool = True,
) -> dict[str, Any]:
    """Verify `'CUDA'` is in OpenMM platforms; optionally install `OpenMM-CUDA-12` via `uv pip`."""
    inv = get_openmm_platform_inventory()
    if inv["has_cuda_platform"] and inv["lib_openmm_cuda_path"] is not None:
        return {"repaired": False, **inv}
    if not auto_repair:
        return {"repaired": False, **inv}

    py = python_exe or sys.executable
    uv_bin = shutil.which("uv")
    ver = inv.get("openmm_version")
    cuda_pkg = f"OpenMM-CUDA-12=={ver}" if ver and ver != "unknown" else "OpenMM-CUDA-12"
    cmds: list[list[str]] = []
    if uv_bin:
        cmds.append([uv_bin, "pip", "install", "--python", py, cuda_pkg])
        cmds.append([uv_bin, "pip", "install", "--python", py, "--upgrade", "openmm[cuda12]"])
    cmds.append([py, "-m", "pip", "install", cuda_pkg])

    repair_logs: list[str] = []
    for cmd in cmds:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            repair_logs.append(
                f"$ {' '.join(cmd)} -> rc={proc.returncode}\n{proc.stdout[-400:]}\n{proc.stderr[-400:]}"
            )
            if proc.returncode == 0:
                break
        except Exception as exc:
            repair_logs.append(f"$ {' '.join(cmd)} -> exc={exc}")

    # Re-query in a fresh subprocess so newly installed shared libraries are loaded cleanly
    check_code = (
        "import json; from scripts.benchmarks.openmm_cuda_guard import get_openmm_platform_inventory; "
        "print(json.dumps(get_openmm_platform_inventory()))"
    )
    try:
        sub = subprocess.run(
            [py, "-c", check_code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if sub.returncode == 0 and sub.stdout.strip():
            inv_after = json.loads(sub.stdout.strip().splitlines()[-1])
            return {"repaired": True, "repair_logs": repair_logs, **inv_after}
    except Exception:
        pass
    inv_after = get_openmm_platform_inventory()
    return {"repaired": True, "repair_logs": repair_logs, **inv_after}


def extract_openmm_context_provenance(ctx: Any) -> dict[str, Any]:
    """Extract hardware, compiler, driver, and library provenance from a live OpenMM `Context`."""
    plat = ctx.getPlatform()
    plat_name = plat.getName()
    prop_names = set(plat.getPropertyNames())

    def _prop(keys: tuple[str, ...]) -> str | None:
        for k in keys:
            if k in prop_names:
                try:
                    val = plat.getPropertyValue(ctx, k)
                    if val is not None and str(val) != "":
                        return str(val)
                except Exception:
                    pass
        return None

    return {
        "platform_name": plat_name,
        "device_name": _prop(("DeviceName", "OpenCLDeviceName")),
        "device_index": _prop(("DeviceIndex", "OpenCLDeviceIndex")),
        "precision": _prop(("Precision", "CudaPrecision", "OpenCLPrecision")),
        "cuda_Compiler": _prop(("Compiler", "CudaCompiler")),
        "CudaDriverVersion": _prop(("CudaDriverVersion", "DriverVersion")),
        "lib_openmm_cuda_path": find_lib_openmm_cuda_path(),
        "mps_active": is_cuda_mps_active(),
        "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        "plugin_load_failures": list(__import__("openmm").Platform.getPluginLoadFailures()),
    }


def verify_openmm_cuda_context(
    ctx: Any,
    expected_precision: str = "single",
    *,
    allow_opencl_fallback: bool = False,
    allow_cpu_fallback: bool = False,
) -> dict[str, Any]:
    """Verify an existing OpenMM `Context` (or `Simulation`) is running on `CUDA` with `expected_precision`.

    Raises a hard `RuntimeError` if the context is not on `CUDA` (unless explicitly allowed)
    or if the precision property does not match `expected_precision`.
    """
    if hasattr(ctx, "context"):
        ctx = ctx.context
    prov = extract_openmm_context_provenance(ctx)
    plat_name = prov["platform_name"]

    if plat_name != "CUDA":
        if plat_name == "OpenCL" and allow_opencl_fallback:
            return prov
        if plat_name in ("CPU", "Reference") and allow_cpu_fallback:
            return prov
        raise RuntimeError(
            f"OpenMM Anti-Fallback Sensor tripped: expected platform 'CUDA', but context initialized on "
            f"'{plat_name}'. Provenance={json.dumps(prov, default=str)}. "
            f"Pass --allow-opencl-fallback (or allow_opencl_fallback=True) only for explicit non-CUDA debugging."
        )

    actual_prec = (prov.get("precision") or "").strip().lower()
    exp_prec = expected_precision.strip().lower()
    if actual_prec != exp_prec:
        raise RuntimeError(
            f"OpenMM Anti-Fallback Sensor tripped: expected CUDA Precision='{exp_prec}', "
            f"got '{actual_prec}'. Provenance={json.dumps(prov, default=str)}"
        )

    if prov.get("lib_openmm_cuda_path") is None:
        raise RuntimeError(
            f"OpenMM Anti-Fallback Sensor tripped: platform reported 'CUDA' but `libOpenMMCUDA.so` "
            f"could not be resolved on disk. Provenance={json.dumps(prov, default=str)}"
        )
    return prov


def require_openmm_cuda_context(
    system: Any,
    integrator: Any,
    expected_precision: str = "single",
    *,
    allow_opencl_fallback: bool = False,
    allow_cpu_fallback: bool = False,
    device_index: int | str | None = None,
    extra_properties: dict[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create an `openmm.Context` guaranteed to run on `CUDA` with `expected_precision` and return `(ctx, provenance)`."""
    import openmm as mm

    inv = get_openmm_platform_inventory()
    target_platform_name = "CUDA"
    if not inv["has_cuda_platform"]:
        if allow_opencl_fallback and "OpenCL" in inv["platforms"]:
            target_platform_name = "OpenCL"
        elif allow_cpu_fallback and "CPU" in inv["platforms"]:
            target_platform_name = "CPU"
        elif allow_cpu_fallback and "Reference" in inv["platforms"]:
            target_platform_name = "Reference"
        else:
            raise RuntimeError(
                f"OpenMM Anti-Fallback Sensor tripped: 'CUDA' platform is not registered in OpenMM! "
                f"Registered platforms={inv['platforms']}, lib_openmm_cuda_path={inv['lib_openmm_cuda_path']}, "
                f"plugin_load_failures={inv['plugin_load_failures']}, python={inv['python_executable']}. "
                f"Install `openmm[cuda12]` (`OpenMM-CUDA-12`) or pass `--allow-opencl-fallback`."
            )

    platform = mm.Platform.getPlatformByName(target_platform_name)
    props: dict[str, str] = {}
    if target_platform_name in ("CUDA", "OpenCL"):
        props["Precision"] = expected_precision.strip().lower()
        if device_index is not None:
            props["DeviceIndex"] = str(device_index)
    if extra_properties:
        props.update(extra_properties)

    ctx = mm.Context(system, integrator, platform, props) if props else mm.Context(system, integrator, platform)
    prov = verify_openmm_cuda_context(
        ctx,
        expected_precision=expected_precision,
        allow_opencl_fallback=allow_opencl_fallback,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    return ctx, prov


def query_gpu_compute_apps_memory_mb(pids: list[int] | None = None) -> dict[str, Any]:
    """Query `nvidia-smi --query-compute-apps=pid,used_memory` and sum GPU memory across target PIDs."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return {"total_used_memory_mb": 0.0, "per_pid_mb": {}}
        per_pid: dict[int, float] = {}
        pid_filter = set(pids) if pids is not None else None
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                pid_i = int(parts[0])
                mem_f = float(parts[1])
            except ValueError:
                continue
            if pid_filter is None or pid_i in pid_filter:
                per_pid[pid_i] = per_pid.get(pid_i, 0.0) + mem_f
        return {
            "total_used_memory_mb": float(sum(per_pid.values())),
            "per_pid_mb": {str(k): float(v) for k, v in per_pid.items()},
        }
    except Exception:
        return {"total_used_memory_mb": 0.0, "per_pid_mb": {}}


class CudaMpsServerContext:
    """Context manager that starts an isolated `nvidia-cuda-mps-control -d` daemon if not already running."""

    def __init__(self, job_tag: str | None = None, enabled: bool = True):
        self.enabled = bool(enabled)
        tag = job_tag or os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        self.base_dir = Path(tempfile.gettempdir()) / f"mps_{tag}_{os.getpid()}"
        self.pipe_dir = self.base_dir / "pipe"
        self.log_dir = self.base_dir / "log"
        self.started_by_us = False
        self.prev_pipe_env: str | None = None
        self.prev_log_env: str | None = None

    def __enter__(self) -> dict[str, Any]:
        if not self.enabled or os.environ.get("PROLIX_DISABLE_MPS_DAEMON") == "1":
            return {"mps_active": is_cuda_mps_active(), "started_by_harness": False}
        if is_cuda_mps_active():
            return {
                "mps_active": True,
                "started_by_harness": False,
                "pipe_dir": os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps"),
                "log_dir": os.environ.get("CUDA_MPS_LOG_DIRECTORY", "/var/log/nvidia-mps"),
            }
        mps_bin = shutil.which("nvidia-cuda-mps-control")
        if not mps_bin:
            return {
                "mps_active": False,
                "started_by_harness": False,
                "error": "nvidia-cuda-mps-control binary not found in PATH",
            }
        self.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prev_pipe_env = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        self.prev_log_env = os.environ.get("CUDA_MPS_LOG_DIRECTORY")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(self.pipe_dir)
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(self.log_dir)
        try:
            proc = subprocess.run(
                [mps_bin, "-d"],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.started_by_us = proc.returncode == 0
            time.sleep(0.1)
            if self.started_by_us:
                # Probe whether nvidia-cuda-mps-server can attach to the GPU within 6s
                # (can fail/hang if parent already holds a non-MPS context or on MIG slices).
                probe = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--check"],
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    timeout=6,
                    check=False,
                )
                if probe.returncode != 0 or "\"live_cuda_context_verified\": true" not in probe.stdout:
                    self.__exit__(None, None, None)
                    self.started_by_us = False
                    return {
                        "mps_active": False,
                        "started_by_harness": False,
                        "fallback_reason": "mps_server_probe_failed_or_blocked_by_active_context",
                    }
            return {
                "mps_active": is_cuda_mps_active() or self.started_by_us,
                "started_by_harness": self.started_by_us,
                "pipe_dir": str(self.pipe_dir),
                "log_dir": str(self.log_dir),
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        except Exception as exc:
            return {
                "mps_active": False,
                "started_by_harness": False,
                "error": str(exc),
            }

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.started_by_us:
            mps_bin = shutil.which("nvidia-cuda-mps-control")
            if mps_bin:
                try:
                    subprocess.run(
                        [mps_bin],
                        input="quit\n",
                        env=os.environ.copy(),
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                except Exception:
                    pass
        if self.prev_pipe_env is not None:
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = self.prev_pipe_env
        elif self.started_by_us:
            os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
        if self.prev_log_env is not None:
            os.environ["CUDA_MPS_LOG_DIRECTORY"] = self.prev_log_env
        elif self.started_by_us:
            os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
        shutil.rmtree(self.base_dir, ignore_errors=True)


def _mps_openmm_worker(
    rank: int,
    system_xml_list: list[str],
    positions_nm_list: list[np.ndarray],
    precision: str,
    dt_fs: float,
    gamma_ps: float,
    temp_k: float,
    n_warmup: int,
    n_steps: int,
    allow_opencl_fallback: bool,
    allow_cpu_fallback: bool,
    ready_barrier: Any,
    start_barrier: Any,
    done_barrier: Any,
    out_queue: Any,
) -> None:
    """Worker process for concurrent OpenMM CUDA + MPS benchmarking."""
    try:
        import openmm as mm
        import openmm.unit as unit

        sys_xml = system_xml_list[rank % len(system_xml_list)]
        pos_nm = positions_nm_list[rank % len(positions_nm_list)]
        system = mm.XmlSerializer.deserialize(sys_xml)
        integrator = mm.LangevinMiddleIntegrator(
            temp_k * unit.kelvin,
            gamma_ps / unit.picosecond,
            (dt_fs * 1e-3) * unit.picoseconds,
        )
        integrator.setRandomNumberSeed(1000 + rank)
        ctx, prov = require_openmm_cuda_context(
            system,
            integrator,
            expected_precision=precision,
            allow_opencl_fallback=allow_opencl_fallback,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        ctx.setPositions(pos_nm * unit.nanometer)
        ctx.setVelocitiesToTemperature(temp_k * unit.kelvin, 2000 + rank)
        # Warmup & JIT / PME plan compilation BEFORE barrier synchronization
        integrator.step(int(n_warmup))
        _ = ctx.getState(getEnergy=True).getPotentialEnergy()

        ready_barrier.wait(timeout=45)
        start_barrier.wait(timeout=45)
        t0 = time.perf_counter()
        integrator.step(int(n_steps))
        state_end = ctx.getState(getEnergy=True)
        pe_kj = float(state_end.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        elapsed_s = time.perf_counter() - t0
        done_barrier.wait(timeout=45)

        del ctx, integrator
        out_queue.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "ok": True,
                "elapsed_s": float(elapsed_s),
                "potential_energy_kj_mol": pe_kj,
                "provenance": prov,
            }
        )
    except Exception as exc:
        try:
            ready_barrier.abort()
        except Exception:
            pass
        try:
            start_barrier.abort()
        except Exception:
            pass
        try:
            done_barrier.abort()
        except Exception:
            pass
        out_queue.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def run_openmm_cuda_mps_replicas(
    system_or_list: Any,
    positions_nm_or_list: Any,
    *,
    replicas_B: int | None = None,
    precision: str = "single",
    dt_fs: float = 2.0,
    gamma_ps: float = 1.0,
    temp_k: float = 300.0,
    n_warmup: int = 50,
    n_steps: int = 500,
    use_mps_daemon: bool = True,
    use_mps: bool | None = None,
    allow_opencl_fallback: bool = False,
    allow_cpu_fallback: bool = False,
) -> dict[str, Any]:
    """Run `B` synchronized concurrent OpenMM `CUDA` worker processes under NVIDIA `CUDA MPS`.

    Accepts either a single `(openmm.System, positions_nm)` (homogeneous `B` replicas)
    or lists `([System_0, ...], [pos_0, ...])` (heterogeneous congeneric or ragged screen).
    """
    import numpy as np
    import openmm as mm
    import openmm.unit as unit

    if use_mps is not None:
        use_mps_daemon = bool(use_mps)
    if replicas_B is not None:
        B = int(replicas_B)
    elif isinstance(system_or_list, (list, tuple)):
        B = len(system_or_list)
    else:
        B = 1
    assert B >= 1, f"replicas_B must be >= 1, got {B}"

    sys_list = system_or_list if isinstance(system_or_list, (list, tuple)) else [system_or_list]
    pos_list_raw = (
        positions_nm_or_list
        if isinstance(positions_nm_or_list, (list, tuple))
        else [positions_nm_or_list]
    )

    xml_list: list[str] = []
    for s in sys_list:
        xml_list.append(s if isinstance(s, str) else mm.XmlSerializer.serialize(s))

    pos_nm_list: list[np.ndarray] = []
    for p in pos_list_raw:
        if hasattr(p, "value_in_unit"):
            pos_nm_list.append(np.asarray(p.value_in_unit(unit.nanometer), dtype=np.float32))
        else:
            pos_nm_list.append(np.asarray(p, dtype=np.float32))

    if os.environ.get("_OPENMM_MPS_COORDINATOR_SUBPROC") != "1":
        import pickle
        with tempfile.TemporaryDirectory(prefix="omm_mps_coord_") as td:
            in_pkl = Path(td) / "task.pkl"
            out_json = Path(td) / "result.json"
            in_pkl.write_bytes(
                pickle.dumps(
                    {
                        "xml_list": xml_list,
                        "pos_nm_list": pos_nm_list,
                        "replicas_B": B,
                        "precision": precision,
                        "dt_fs": float(dt_fs),
                        "gamma_ps": float(gamma_ps),
                        "temp_k": float(temp_k),
                        "n_warmup": int(n_warmup),
                        "n_steps": int(n_steps),
                        "use_mps_daemon": bool(use_mps_daemon),
                        "allow_opencl_fallback": bool(allow_opencl_fallback),
                        "allow_cpu_fallback": bool(allow_cpu_fallback),
                    }
                )
            )
            env_sub = os.environ.copy()
            env_sub["_OPENMM_MPS_COORDINATOR_SUBPROC"] = "1"
            proc_c = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mps-coordinator",
                    str(in_pkl),
                    str(out_json),
                ],
                env=env_sub,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if out_json.exists():
                return json.loads(out_json.read_text())
            err_txt = (proc_c.stderr or proc_c.stdout or f"returncode={proc_c.returncode}").strip()
            if not allow_opencl_fallback and not allow_cpu_fallback and "Anti-Fallback Sensor" in err_txt:
                raise RuntimeError(err_txt)
            return {
                "available": False,
                "ok": False,
                "replicas_B": B,
                "precision": precision,
                "error": err_txt[-600:],
            }

    ctx_mp = mp.get_context("spawn")
    with CudaMpsServerContext(enabled=(use_mps_daemon and B > 1)) as mps_info:
        ready_barrier = ctx_mp.Barrier(B + 1)
        start_barrier = ctx_mp.Barrier(B + 1)
        done_barrier = ctx_mp.Barrier(B + 1)
        out_q = ctx_mp.Queue()

        procs = [
            ctx_mp.Process(
                target=_mps_openmm_worker,
                args=(
                    r,
                    xml_list,
                    pos_nm_list,
                    precision,
                    float(dt_fs),
                    float(gamma_ps),
                    float(temp_k),
                    int(n_warmup),
                    int(n_steps),
                    bool(allow_opencl_fallback),
                    bool(allow_cpu_fallback),
                    ready_barrier,
                    start_barrier,
                    done_barrier,
                    out_q,
                ),
            )
            for r in range(B)
        ]
        for p in procs:
            p.start()

        worker_pids = [p.pid for p in procs if p.pid is not None]
        try:
            ready_barrier.wait(timeout=45)
            mem_snapshot = query_gpu_compute_apps_memory_mb(worker_pids)
            start_barrier.wait(timeout=45)
            t0_wall = time.perf_counter()
            done_barrier.wait(timeout=45)
            wall_elapsed_s = time.perf_counter() - t0_wall
        except Exception as bar_exc:
            wall_elapsed_s = float("nan")
            mem_snapshot = query_gpu_compute_apps_memory_mb(worker_pids)
            for p in procs:
                if p.is_alive():
                    p.terminate()
            return {
                "available": False,
                "ok": False,
                "replicas_B": B,
                "precision": precision,
                "mps_info": mps_info,
                "gpu_memory_mb": mem_snapshot.get("total_used_memory_mb", 0.0),
                "error": f"Barrier synchronization failed across B={B} OpenMM MPS workers: {bar_exc}",
            }

        results = []
        for _ in range(B):
            try:
                results.append(out_q.get(timeout=30))
            except Exception:
                break
        for p in procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()

    failures = [r for r in results if not r.get("ok", False)]
    if failures or len(results) < B:
        err_msg = "; ".join(r.get("error", "unknown error") for r in failures) or "Incomplete worker results"
        if not allow_opencl_fallback and not allow_cpu_fallback and "Anti-Fallback Sensor" in err_msg:
            raise RuntimeError(err_msg)
        return {
            "available": False,
            "ok": False,
            "replicas_B": B,
            "precision": precision,
            "mps_info": mps_info,
            "gpu_memory_mb": mem_snapshot.get("total_used_memory_mb", 0.0),
            "error": err_msg,
        }

    max_worker_s = max(float(r["elapsed_s"]) for r in results)
    eff_wall_s = max(float(wall_elapsed_s), max_worker_s)
    ms_per_step_batch = (eff_wall_s / n_steps) * 1000.0
    ms_per_step_per_rep = ms_per_step_batch / B
    ps_per_step = dt_fs * 1e-3
    ns_per_day_per_rep = ((86400.0 * 1000.0) / ms_per_step_batch) * (ps_per_step * 1e-3)
    aggregate_ns_per_day = ns_per_day_per_rep * B
    total_mem_mb = float(mem_snapshot.get("total_used_memory_mb", 0.0))

    return {
        "available": True,
        "ok": True,
        "replicas_B": B,
        "precision": precision,
        "n_steps": int(n_steps),
        "dt_fs": float(dt_fs),
        "wall_elapsed_s": float(eff_wall_s),
        "ms_per_step_batch": float(ms_per_step_batch),
        "ms_per_step_per_replica": float(ms_per_step_per_rep),
        "per_replica_ns_per_day": float(ns_per_day_per_rep),
        "aggregate_ns_per_day": float(aggregate_ns_per_day),
        "total_gpu_memory_mb": total_mem_mb,
        "memory_mb_per_replica": total_mem_mb / B if B > 0 else 0.0,
        "mps_info": mps_info,
        "provenance": results[0].get("provenance", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify OpenMM CUDA platform & plugin installation and optionally auto-repair."
    )
    parser.add_argument("--check", action="store_true", help="Check OpenMM CUDA platform and exit.")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Install/repair openmm[cuda12] if CUDA platform is missing.",
    )
    parser.add_argument("--allow-opencl-fallback", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--mps-coordinator", nargs=2, metavar=("IN_PKL", "OUT_JSON"))
    args = parser.parse_args()

    if args.mps_coordinator:
        import pickle
        in_p, out_p = Path(args.mps_coordinator[0]), Path(args.mps_coordinator[1])
        task = pickle.loads(in_p.read_bytes())
        os.environ["_OPENMM_MPS_COORDINATOR_SUBPROC"] = "1"
        res = run_openmm_cuda_mps_replicas(
            task["xml_list"],
            task["pos_nm_list"],
            replicas_B=task["replicas_B"],
            precision=task["precision"],
            dt_fs=task["dt_fs"],
            gamma_ps=task["gamma_ps"],
            temp_k=task["temp_k"],
            n_warmup=task["n_warmup"],
            n_steps=task["n_steps"],
            use_mps_daemon=task["use_mps_daemon"],
            allow_opencl_fallback=task["allow_opencl_fallback"],
            allow_cpu_fallback=task["allow_cpu_fallback"],
        )
        out_p.write_text(json.dumps(res, default=str))
        return

    inv = ensure_openmm_cuda_installed(auto_repair=args.repair)
    if args.check and inv.get("has_cuda_platform", False):
        try:
            import openmm as mm

            sys_2 = mm.System()
            sys_2.addParticle(12.0)
            sys_2.addParticle(12.0)
            nb = mm.NonbondedForce()
            nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
            nb.addParticle(0.1, 0.3, 0.5)
            nb.addParticle(-0.1, 0.3, 0.5)
            sys_2.addForce(nb)
            integ = mm.VerletIntegrator(0.001)
            ctx, prov = require_openmm_cuda_context(
                sys_2,
                integ,
                expected_precision="single",
                allow_opencl_fallback=args.allow_opencl_fallback,
                allow_cpu_fallback=args.allow_cpu_fallback,
            )
            ctx.setPositions([mm.Vec3(0.0, 0.0, 0.0), mm.Vec3(0.5, 0.0, 0.0)])
            integ.step(1)
            del ctx, integ
            inv["live_cuda_context_verified"] = True
            inv["context_provenance"] = prov
        except Exception as exc:
            inv["live_cuda_context_verified"] = False
            inv["context_error"] = str(exc)
    print(json.dumps(inv, indent=2, default=str))
    if args.check and not inv.get("has_cuda_platform", False):
        if not (args.allow_opencl_fallback or args.allow_cpu_fallback):
            raise SystemExit(
                "ERROR: OpenMM CUDA platform is NOT available and neither --allow-opencl-fallback "
                "nor --allow-cpu-fallback was passed."
            )


if __name__ == "__main__":
    main()

