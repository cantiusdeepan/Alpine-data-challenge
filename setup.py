import os
import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.communicate()

def setup_environment():
    gpu_available = os.system("nvidia-smi") == 0

    if gpu_available:
        commands = [
            "conda create -n ag python=3.11 -y",
            "conda activate ag",
            "conda install -c conda-forge mamba -y",
            "mamba install -c conda-forge -c pytorch -c nvidia autogluon 'pytorch=*=*cuda*' -y",
            "mamba install -c conda-forge 'ray-tune>=2.10.0,<2.32' 'ray-default>=2.10.0,<2.32' -y"
        ]
    else:
        commands = [
            "conda create -n ag python=3.10 -y",
            "conda activate ag",
            "conda install -c conda-forge mamba -y",
            "mamba install -c conda-forge autogluon -y",
            "mamba install -c conda-forge 'ray-tune>=2.10.0,<2.32' 'ray-default>=2.10.0,<2.32' -y"
        ]

    for command in commands:
        run_command(command)

if __name__ == "__main__":
    setup_environment()
