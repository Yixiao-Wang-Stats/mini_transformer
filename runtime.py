"""Environment and dataset checks shared by scripts and notebooks."""

import ast
import inspect
import os
from pathlib import Path
import pickle
import re
import socket
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parent


def _run_scontrol(*arguments):
    try:
        result = subprocess.run(
            ["scontrol", *arguments], check=True, capture_output=True,
            text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RuntimeError(
            "Cannot verify the Slurm allocation with scontrol. Start Jupyter inside "
            f"a working CS Slurm allocation, then retry. Details: {detail.strip()}"
        ) from error
    return result.stdout.strip()


def _is_cs_cluster(hostname, fqdn):
    if (hostname.startswith("compsci-")
            or re.fullmatch(r"(?:fitz-\d+|linux-?\d+|gpu-compute\d+)", hostname)
            or fqdn.lower().endswith(".cs.duke.edu")
            or os.environ.get("SLURM_CLUSTER_NAME") == "compsci_cluster"):
        return True
    # This also identifies login hosts with other names; ordinary laptops have no CS config.
    try:
        config = Path("/etc/slurm/slurm.conf").read_text()
    except OSError:
        return False
    return bool(re.search(r"^\s*ClusterName\s*=\s*compsci_cluster\s*$", config, re.MULTILINE))


def validate_execution_environment(device="auto"):
    """Return the verified execution context; reject CS login nodes and invalid jobs.

    A local computer may run without Slurm. On Duke CS, only fitz-* / linux-* compute
    nodes in a live allocation are accepted. Setting SLURM_JOB_ID alone is insufficient:
    scontrol must report RUNNING, the current user, and an allocation containing this host.
    """
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    hostname = socket.gethostname().split(".")[0].lower()
    fqdn = socket.getfqdn().lower()
    is_cluster = _is_cs_cluster(hostname, fqdn)
    job_id = os.environ.get("SLURM_JOB_ID")
    fields = {}
    if is_cluster:
        if not re.fullmatch(r"(?:(?:compsci-cluster-)?fitz-\d+|linux-?\d+)", hostname):
            raise RuntimeError(
                f"Training and sampling are blocked on CS host {hostname}. "
                "Start Jupyter on a fitz-* / linux-* compute node allocated by Slurm."
            )
        if not job_id or not re.fullmatch(r"\d+(?:_\d+)?", job_id):
            raise RuntimeError(
                f"Host {hostname} is a CS compute node, but this process has no valid "
                "SLURM_JOB_ID. Start the notebook/kernel inside an srun allocation."
            )
        output = _run_scontrol("show", "job", "-o", job_id)
        fields = dict(re.findall(r"([A-Za-z][A-Za-z0-9_]*)=(\S+)", output))
        if fields.get("JobState") != "RUNNING":
            raise RuntimeError(f"Slurm job {job_id} is not RUNNING: {fields.get('JobState', 'unknown')}")
        owner = fields.get("UserId", "")
        if owner and not owner.endswith(f"({os.getuid()})"):
            raise RuntimeError(f"Slurm job {job_id} belongs to a different user ({owner})")
        node_list = fields.get("NodeList")
        if not node_list or node_list in ("(null)", "None"):
            raise RuntimeError(f"Slurm job {job_id} does not report an allocated NodeList")
        allocated_hosts = {
            name.split(".")[0].lower()
            for name in _run_scontrol("show", "hostnames", node_list).splitlines()
        }
        if hostname not in allocated_hosts:
            raise RuntimeError(f"Current host {hostname} is not in job {job_id}'s allocation: {node_list}")

    # Delay importing torch until after rejecting an invalid cluster execution context.
    import torch
    cuda_available = torch.cuda.is_available()
    allocated_gpu = True
    if is_cluster:
        allocated_gpu = bool(re.search(r"(?:gres/)?gpu(?::[A-Za-z0-9_]+)?[=:][1-9]\d*", " ".join(
            fields.get(name, "") for name in ("AllocTRES", "TresPerNode", "TresPerTask", "Gres")
        )))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        allocated_gpu = allocated_gpu and visible not in ("", "-1", "NoDevFiles", "none", "void")
    selected = "cuda" if device == "auto" and cuda_available and allocated_gpu else device
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but this Python environment has no available CUDA device")
    if selected == "cuda" and not allocated_gpu:
        raise RuntimeError("CUDA requires a GPU allocated to this Slurm job and valid CUDA_VISIBLE_DEVICES")
    if is_cluster:
        raw_threads = os.environ.get("SLURM_CPUS_PER_TASK") or fields.get("NumCPUs", "1")
        try:
            cpu_threads = max(1, int(raw_threads))
        except ValueError:
            cpu_threads = 1
        # Keep defaults moderate even for allocations that grant a whole machine.
        cpu_threads = min(4, cpu_threads)
    else:
        cpu_threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(cpu_threads)
    return {
        "hostname": hostname,
        "fqdn": fqdn,
        "is_cs_cluster": is_cluster,
        "slurm_job_id": job_id,
        "slurm_partition": fields.get("Partition"),
        "allocated_nodes": fields.get("NodeList"),
        "device": selected,
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.executable,
        "cpu_threads": cpu_threads,
    }


def require_complete_attention(attention_class):
    """Reject the provided placeholders before training or loading a trained model."""
    try:
        source = textwrap.dedent(inspect.getsource(attention_class.forward))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as error:
        raise RuntimeError("Cannot inspect SelfAttention.forward; keep its implementation in model.py") from error
    has_ellipsis = any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))
    has_random_return = any(
        isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "randn_like"
        for node in ast.walk(tree)
    )
    if has_ellipsis or has_random_return:
        raise RuntimeError(
            "SelfAttention.forward still contains homework placeholders. Complete all five "
            "steps and replace the random return with the attention output before training "
            "or loading a checkpoint. The scratch baseline in part (a) remains available."
        )


def load_metadata(data_dir=None):
    """Read the bundled character vocabulary and verify its bijection."""
    directory = Path(data_dir).expanduser().resolve() if data_dir is not None else ROOT / "data"
    with (directory / "meta.pkl").open("rb") as stream:
        metadata = pickle.load(stream)
    if not isinstance(metadata, dict) or not all(key in metadata for key in ("vocab_size", "stoi", "itos")):
        raise ValueError("meta.pkl must contain vocab_size, stoi, and itos")
    vocab_size = metadata["vocab_size"]
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("Metadata vocab_size must be a positive integer")
    stoi, itos = metadata["stoi"], metadata["itos"]
    if len(stoi) != vocab_size or len(itos) != vocab_size:
        raise ValueError("Metadata vocabulary lengths do not match vocab_size")
    if set(itos) != set(range(vocab_size)):
        raise ValueError("Metadata token IDs must be consecutive integers starting at zero")
    if any(not isinstance(char, str) or len(char) != 1 or itos.get(index) != char for char, index in stoi.items()):
        raise ValueError("Metadata stoi/itos must be inverse character mappings")
    return directory, metadata
