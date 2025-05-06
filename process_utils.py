import logging
import subprocess

from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Mapping, Optional, Tuple
import os

logger = logging.getLogger(__name__)


def get_output_or_exit(command: str, blocking: bool = True) -> Optional[str]:
    """
    Runs the bash command and streams immediate stdout.
    If blocking=True, also waits and returns the output.
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
    for line in p.stdout:
        logger.info(f"Stdout: {line}")
    for line in p.stderr:
        logger.error(f"Stderr: {line}")
    if not blocking:
        return
    p.wait()
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        raise ValueError(
            f"Command '{command}' has failed.\nStdout: {stdout}\nStderr: {stderr}"
        )
    for line in stderr:
        logger.error(line)
    return stdout


def create_temporary_slurm_script(
    sbatch_args: List[str],  # Example: ["--nodelist=sp-0002", "--mem=100GB"]
    py_commands: str,
    conda_env_name: str,
    singularity_overlay_fp: str,
    singularity_img_fp: str = "/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif",
    add_default_init_commands: bool = True,

) -> NamedTemporaryFile:
    """Creates temporary slurm file.

    Arguments:
    sbatch_args: List of SBATCH arguments.
    commands: Commands to include after both sbatch_args and default initial commands.
    add_default_init_commands: Whether to include default initial commands, such as
                               running shared_bashrc and activating the Conda environment.

    Returns: Temporary SLURM script that runs python commands within the singularity container. Must be closed/deleted by the user.
    """
    sbatch_args = "\n".join([f"#SBATCH {s}" for s in sbatch_args])
    script_contents = f"#!/bin/bash\n{sbatch_args}\n\n"
    singularity_bin_path = "/share/apps/singularity/bin/singularity"
    if add_default_init_commands:
        script_contents += f'{singularity_bin_path} exec --nv --overlay {singularity_overlay_fp}:ro {singularity_img_fp} /bin/bash -c "'
        script_contents += "source /ext3/env.sh\n"
        script_contents += f"conda deactivate && conda activate {conda_env_name}\n"
        script_contents += "export HF_HOME=/scratch/$USER/huggingface/job_$SLURM_JOB_ID\n\n"
    script_contents += py_commands
    if add_default_init_commands:
        script_contents += f'\n"'
    f = NamedTemporaryFile(mode="w", delete=False)
    f.write(script_contents)
    f.seek(0)
    logger.info(f"Contents of dynamically created SLURM script:\n{script_contents}")
    return f


def create_single_gpu_slurm_script(
    py_commands: str,
    conda_env_name: str,
    singularity_overlay_fp: str,
    singularity_img_fp: str = "/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif",
    slurm_output_dir: str = "/scratch/%u/grn/slurm_output",
    email: Optional[str] = None,
    gpu_types: List[str] = ["a100"],
    mem: str = "250GB",
) -> NamedTemporaryFile:
    # TODO: convert this to hydra config instead!
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
        singularity_overlay_fp,
        singularity_img_fp,
        add_default_init_commands=True,
    )

def create_multiple_gpu_slurm_script(
    py_commands: str,
    conda_env_name: str,
    singularity_overlay_fp: str,
    singularity_img_fp: str = "/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif",
    slurm_output_dir: str = "/scratch/%u/grn/slurm_output",
    email: Optional[str] = None,
    gpu_types: List[str] = ["a100"],
    mem: str = "250GB",
    num_gpus: int = 4,
) -> NamedTemporaryFile:
    # TODO: convert this to hydra config instead!
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
        singularity_overlay_fp,
        singularity_img_fp,
        add_default_init_commands=True,
    )

def create_single_cpu_slurm_script(
    py_commands: str,
    conda_env_name: str,
    singularity_overlay_fp: str,
    singularity_img_fp: str = "/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif",
    slurm_output_dir: str = "/scratch/%u/grn/slurm_output",
    email: Optional[str] = None,
    mem: str = "8GB"
) -> NamedTemporaryFile:
    # TODO: convert this to hydra config instead!
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
        singularity_overlay_fp,
        singularity_img_fp,
        add_default_init_commands=True,
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
    for p in all_processes:
        stdout_str = ""
        for line in p.stdout:
            logger.info(f"Stdout: {line}")
            stdout_str += f"{line}\n"
        all_outputs.append(stdout_str)

    for p, cmd in zip(all_processes, commands):
        p.wait()
        # If the process failed, return early!
        _, stderr = p.communicate()
        if p.returncode != 0 and not ignore_failures:
            raise ValueError(f"Command '{cmd}' has failed.\nStderr: {stderr}")
        all_stderrs.append(stderr)
        returncodes.append(p.returncode)
    return all_outputs, all_stderrs, returncodes
