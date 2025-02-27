# Alpine-data-challenge

## Installation Instructions

### Using setup.py

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd Alpine-data-challenge
    ```

2. Run the setup script:
    ```sh
    python setup.py
    ```

### Using Conda Environment File

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd Alpine-data-challenge
    ```

2. Create the conda environment:
    ```sh
    conda env create -f ag-alpine_valley.yml --name ag
    ```

3. Activate the environment:
    ```sh
    conda activate ag
    mamba install -c conda-forge autogluon "pytorch=*=cuda*"
    mamba install -c conda-forge "ray-tune >=2.10.0,<2.32" "ray-default >=2.10.0,<2.32"  # install ray for faster training
    ```
