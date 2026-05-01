import logging
import subprocess

from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Mapping, Optional, Tuple
import os

logger = logging.getLogger(__name__)


def get_output_or_exit(command: str, blocking: bool = True) -> Optional[str]:
    """
    Runs the bash command. If blocking=True, waits for it to finish,
    logs stdout/stderr line by line, and raises on non-zero exit.
    """
    logger.info(f"Running command: {command}")
    p = subprocess.Popen(
        args=command,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not blocking:
        return
    stdout, stderr = p.communicate()
    for line in stdout.splitlines():
        logger.info(f"Stdout: {line}")
    for line in stderr.splitlines():
        logger.error(f"Stderr: {line}")
    if p.returncode != 0:
        raise ValueError(
            f"Command '{command}' has failed.\nStdout: {stdout}\nStderr: {stderr}"
        )
    return stdout


def create_temporary_slurm_script(
    sbatch_args: List[str],
    py_commands: str,
    conda_env_name: str,
    add_default_init_commands: bool = True,
) -> NamedTemporaryFile:
    """Creates a temporary SLURM script that activates a conda environment.

    Arguments:
    sbatch_args: List of SBATCH arguments.
    py_commands: Commands to run after environment activation.
    conda_env_name: Name of the conda environment to activate.
    add_default_init_commands: Whether to include conda activation and HF_HOME export.

    Returns: Temporary SLURM script. Must be closed/deleted by the caller.
    """
    sbatch_args_str = "\n".join([f"#SBATCH {s}" for s in sbatch_args])
    script_contents = f"#!/bin/bash\n{sbatch_args_str}\n\n"
    if add_default_init_commands:
        script_contents += f"source ~/miniforge3/bin/activate {conda_env_name}\n"
        script_contents += "export HF_HOME=/scratch/$USER/huggingface/job_$SLURM_JOB_ID\n\n"
    script_contents += py_commands
    script_contents += "\n"
    f = NamedTemporaryFile(mode="w", delete=False)
    f.write(script_contents)
    f.seek(0)
    logger.info(f"Contents of dynamically created SLURM script:\n{script_contents}")
    return f


def create_single_gpu_slurm_script(
    py_commands: str,
    conda_env_name: str,
    slurm_output_dir: str = "/scratch/%u/grn/slurm_output",
    email: Optional[str] = None,
    gpu_types: List[str] = ["a100"],
    mem: str = "250GB",
) -> NamedTemporaryFile:
    sbatch_args = [
        "--open-mode=append",
        f"--output={slurm_output_dir}/%x_%j.out",
        f"--error={slurm_output_dir}/%x_%j.err",
        "--export=NONE",
        "--time=47:00:00",
        "--nodes=1",
        "--cpus-per-task=1",
        "--gres=gpu:1",
        f"--constraint={'|'.join(gpu_types)}",
        f"--mem={mem}",
    ]
    if email is not None:
        sbatch_args += [
            "--mail-type=BEGIN,END,FAIL",
            f"--mail-user=%u@nyu.edu",
        ]
    return create_temporary_slurm_script(
        sbatch_args,
        py_commands,
        conda_env_name,
    )


def create_multiple_gpu_slurm_script(
    py_commands: str,
    conda_env_name: str,
    slurm_output_dir: str = "/scratch/%u/grn/slurm_output",
    email: Optional[str] = None,
    gpu_types: List[str] = ["a100"],
    mem: str = "250GB",
    num_gpus: int = 4,
) -> NamedTemporaryFile:
    sbatch_args = [
        "--open-mode=append",
        f"--output={slurm_output_dir}/%x_%j.out",
        f"--error={slurm_output_dir}/%x_%j.err",
        "--export=NONE",
        "--time=47:00:00",
        "--nodes=1",
        f"--ntasks-per-node={num_gpus}",
        f"--cpus-per-task={num_gpus}",
        f"--gres=gpu:{num_gpus}",
        f"--constraint={'|'.join(gpu_types)}",
        f"--mem={mem}",
    ]
    if email is not None:
        sbatch_args += [
            "--mail-type=BEGIN,END,FAIL",
            f"--mail-user=%u@nyu.edu",
        ]
    return create_temporary_slurm_script(
        sbatch_args,
        py_commands,
        conda_env_name,
    )


def create_single_cpu_slurm_script(
    py_commands: str,
    conda_env_name: str,
    slurm_output_dir: str = "/scratch/%u/grn/slurm_output",
    email: Optional[str] = None,
    mem: str = "8GB",
) -> NamedTemporaryFile:
    sbatch_args = [
        "--open-mode=append",
        f"--output={slurm_output_dir}/%x_%j.out",
        f"--error={slurm_output_dir}/%x_%j.err",
        "--export=NONE",
        "--time=47:00:00",
        "--nodes=1",
        "--cpus-per-task=1",
        "--tasks-per-node=1",
        f"--mem={mem}",
    ]
    if email is not None:
        sbatch_args += [
            "--mail-type=BEGIN,END,FAIL",
            f"--mail-user=%u@nyu.edu",
        ]
    return create_temporary_slurm_script(
        sbatch_args,
        py_commands,
        conda_env_name,
    )


def convert_nested_dict_to_option_strs(nested_dict: Dict[str, Any]) -> List[str]:
    """
    Given a nested mapping (like a Hydra config), convert all values into
    option strings of the form "<key 1>.<key 2>...<key n>=value".
    For example:
    {"a": {"b": 1}, "c": 2} -> ["a.b=1", "c=2"]
    """
    outputs = []
    for k, v in nested_dict.items():
        if not isinstance(v, Mapping):
            outputs.append(f"{k}={v}")
        else:
            for o in convert_nested_dict_to_option_strs(v):
                outputs.append(f"{k}.{o}")
    return outputs


def run_all_commands_and_wait_until_all_completed(
    commands: List[str], ignore_failures: bool = False
) -> Tuple[List[str], List[str], List[int]]:
    """
    Runs all the commands in bash shells in parallel processes and
    waits until all child processes have completed. Returns the outputs from all commands.
    """
    all_processes = []
    for cmd in commands:
        p = subprocess.Popen(
            args=cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info(f"Running command: {cmd}")
        all_processes.append(p)
    all_outputs = []
    all_stderrs = []
    returncodes = []
    for p, cmd in zip(all_processes, commands):
        stdout, stderr = p.communicate()
        for line in stdout.splitlines():
            logger.info(f"Stdout: {line}")
        for line in stderr.splitlines():
            logger.error(f"Stderr: {line}")
        if p.returncode != 0 and not ignore_failures:
            raise ValueError(f"Command '{cmd}' has failed.\nStderr: {stderr}")
        all_outputs.append(stdout)
        all_stderrs.append(stderr)
        returncodes.append(p.returncode)
    return all_outputs, all_stderrs, returncodes
