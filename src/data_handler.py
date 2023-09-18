"""DeepCNNDoa
Details
----------
    Name: data_handler.py
    Authors: D. H. Shmuel
    Created: 21/07/23
    Edited:

Purpose:
--------
    This scripts handle the creation and processing of synthetic datasets
    based on specified parameters and model types.
    It includes functions for generating datasets, reading data from files,
    computing autocorrelation matrices, and creating covariance tensors.

Attributes:
-----------
    Samples (from src.signal_creation): A class for creating samples used in dataset generation.

    The script defines the following functions:
    * create_dataset: Generates a synthetic dataset based on the specified parameters and model type.
    * read_data: Reads data from a file specified by the given path.
    * create_cov_tensor: Creates a 3D tensor containing the real part,
        imaginary part, and phase component of the covariance matrix.
    * set_dataset_filename: Returns the generic suffix of the datasets filename.

"""

# Imports
import torch
import numpy as np
import copy
import itertools
from tqdm import tqdm
from src.signal_creation import Samples
from pathlib import Path
from src.system_model import SystemModelParams
from src.models import ModelGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataset(
    system_model_params: SystemModelParams,
    samples_size: float,
    model_config: ModelGenerator,
    save_datasets: bool = False,
    datasets_path: Path = None,
    true_doa: list = None,
    phase: str = None,
    doa_test_params=None,
    exp: int = 1,
):
    """
    Generates a synthetic dataset based on the specified parameters and model type.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams
        samples_size (float): The size of the dataset.
        model_type (str): The type of the model.
        save_datasets (bool, optional): Specifies whether to save the dataset. Defaults to False.
        datasets_path (Path, optional): The path for saving the dataset. Defaults to None.
        true_doa (list, optional): Predefined angles. Defaults to None.
        phase (str, optional): The phase of the dataset (test or training phase for CNN model). Defaults to None.

    Returns:
    --------
        tuple: A tuple containing the desired dataset comprised of (X-samples, Y-labels).

    """
    sps_model_dataset = []
    generic_dataset = []
    model_dataset = []
    # Generate doa permutations for CNN model training dataset
    if model_config.model_type.startswith("DeepCNN"):
        if phase.startswith("train"):
            doa_permutations = []
            limit_angle = (model_config.grid_size // 2)
            angles_grid = np.linspace(
                start=-limit_angle, stop=limit_angle, num=model_config.grid_size
            )
            for comb in itertools.combinations(angles_grid, system_model_params.M):
                doa_permutations.append(list(comb))
            # Generate a copy of system_model_params_snr instance
            system_model_params_snr = copy.deepcopy(system_model_params)
            for snr in [-20, -15, -10, -5, 0]:
                system_model_params_snr.set_snr(snr)
                samples_model = Samples(system_model_params_snr)
                for i, doa in tqdm(enumerate(doa_permutations)):
                    # Samples model creation
                    samples_model.set_doa(doa)
                    # Observations matrix creation
                    X, _, A, _ = samples_model.samples_creation(
                        noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                    )
                    X = torch.tensor(X, dtype=torch.complex64)
                    # Calculate true covariance
                    true_covariance = get_true_covariance(
                        A, system_model_params_snr.snr, signal_variance=1, noise_variance=1)
                    X_model = create_cov_tensor(true_covariance)
                    # Rx = torch.cov(X)
                    # X_model = create_cov_tensor(Rx)
                    # Ground-truth creation
                    Y = torch.zeros_like(torch.tensor(angles_grid))
                    for angle in doa:
                        Y[list(angles_grid).index(angle)] = 1
                    model_dataset.append((X_model, Y))
                    generic_dataset.append((X, Y))
        elif phase.startswith("test") and exp == 1:
            samples_model = Samples(system_model_params)
            doa_list = generate_gaped_doa(start=doa_test_params["start"],
                                          end=doa_test_params["end"],
                                          gap=doa_test_params["gap"],
                                          step=doa_test_params["step"])
            for i, doa in tqdm(enumerate(doa_list)):
                # Samples model creation
                samples_model.set_doa(doa)
                # Observations matrix creation
                X = torch.tensor(
                    samples_model.samples_creation(
                        noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                    )[0],
                    dtype=torch.complex64,
                )
                # Generate 3d covariance parameters tensor
                Rx = torch.cov(X)
                X_model = create_cov_tensor(Rx)
                Rx_sps = torch.tensor(spatial_smoothing_covariance(X))
                X_sps = create_cov_tensor(Rx_sps)
                # Ground-truth creation
                Y = torch.tensor(samples_model.doa, dtype=torch.float64)
                generic_dataset.append((X, Y))
                model_dataset.append((X_model, Y))
                sps_model_dataset.append((X_sps, Y))
        elif phase.startswith("test") and exp == 2:
            samples_model = Samples(system_model_params)
            for i in tqdm(range(samples_size)):
                # Samples model creation
                samples_model.set_doa(true_doa)
                # Observations matrix creation
                X = torch.tensor(
                    samples_model.samples_creation(
                        noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                    )[0],
                    dtype=torch.complex64,
                )
                # Generate 3d covariance parameters tensor
                Rx = torch.cov(X)
                X_model = create_cov_tensor(Rx)
                Rx_sps = torch.tensor(spatial_smoothing_covariance(X))
                X_sps = create_cov_tensor(Rx_sps)
                # Ground-truth creation
                Y = torch.tensor(samples_model.doa, dtype=torch.float64)
                generic_dataset.append((X, Y))
                model_dataset.append((X_model, Y))
                sps_model_dataset.append((X_sps, Y))

    if save_datasets:
        model_dataset_filename = (
            f"{model_config.model_type}_DataSet_{system_model_params.signal_type}_"
            + f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_"
            + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
            + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}"
            + ".h5"
        )

        sps_model_dataset_filename = (
            f"sps_{model_config.model_type}_DataSet_{system_model_params.signal_type}_"
            + f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_"
            + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
            + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}"
            + ".h5"
        )
        generic_dataset_filename = (
            f"Generic_DataSet_{system_model_params.signal_type}_"
            + f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_"
            + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
            + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}"
            + ".h5"
        )
        samples_model_filename = (
            f"samples_model_{system_model_params.signal_type}_"
            + f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_"
            + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
            + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}"
            + ".h5"
        )

        torch.save(obj=model_dataset, f=datasets_path /
                   phase / model_dataset_filename)
        torch.save(obj=model_dataset, f=datasets_path /
                   phase / sps_model_dataset_filename)
        torch.save(
            obj=generic_dataset, f=datasets_path / phase / generic_dataset_filename
        )
        if phase.startswith("test"):
            torch.save(
                obj=samples_model, f=datasets_path / phase / samples_model_filename
            )

    return model_dataset, sps_model_dataset, generic_dataset, samples_model


# def read_data(Data_path: str) -> torch.Tensor:
def read_data(path: str):
    """
    Reads data from a file specified by the given path.

    Args:
    -----
        path (str): The path to the data file.

    Returns:
    --------
        torch.Tensor: The loaded data.

    Raises:
    -------
        None

    Examples:
    ---------
        >>> path = "data.pt"
        >>> read_data(path)

    """
    assert isinstance(path, (str, Path))
    data = torch.load(path)
    return data

# def create_cov_tensor(X: torch.Tensor) -> torch.Tensor:


def get_true_covariance(A: torch.Tensor, snr: float, signal_variance: float, noise_variance: float):
    """
    Returns the true covariance calculation, based on system model theory.

    Args:
    -----
        A (torch.Tensor): covariance matrix from size (M, N.)
        snr (float): the snr value.

    Returns:
    --------
        Rx (torch.Tensor): true covariance matrix from size (N, N.)

    Raises:
    -------
        None

    """
    Rs = (10 ** (snr / 20)) * torch.tensor(signal_variance *
                                           torch.eye(A.shape[1]), dtype=torch.complex64)
    Rn = noise_variance * torch.eye(A.shape[0])
    A = torch.tensor(A, dtype=torch.complex64)
    return torch.tensor(A) @ Rs @ torch.t(torch.conj(A)) + Rn


def create_cov_tensor(Rx: torch.Tensor):
    """
    Creates a 3D tensor of size (NxNx3) containing the real part, imaginary part, and phase component of the covariance matrix.

    Args:
    -----
        Rx (torch.Tensor): covariance matrix from size (N, N.)

    Returns:
    --------
        Rx_tensor (torch.Tensor): Tensor containing the auto-correlation matrices, with size (Batch size, N, N, 3).

    Raises:
    -------
        None

    """
    Rx_tensor = torch.stack(
        (torch.real(Rx), torch.imag(Rx), torch.angle(Rx)), 2)
    return Rx_tensor


def generate_gaped_doa(start: float, end: float, gap: float, step: float):
    doa_list = []
    while start <= end:
        doa_list.append([start, start + gap])
        start += step
    return doa_list


def load_datasets(
    system_model_params: SystemModelParams,
    model_type: str,
    samples_size: float,
    datasets_path: Path,
    train_test_ratio: float,
    is_training: bool = False,
):
    """
    Load different datasets based on the specified parameters and phase.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        model_type (str): The type of the model.
        samples_size (float): The size of the overall dataset.
        datasets_path (Path): The path to the datasets.
        train_test_ratio (float): The ration between train and test datasets.
        is_training (bool): Specifies whether to load the training dataset.

    Returns:
    --------
        List: A list containing the loaded datasets.

    """
    datasets = []
    # Define test set size
    test_samples_size = int(train_test_ratio * samples_size)
    # Generate datasets filenames
    model_dataset_filename = f"{model_type}_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
    sps_model_dataset_filename = f"sps_{model_type}_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
    generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
    samples_model_filename = f"samples_model" + set_dataset_filename(
        system_model_params, test_samples_size
    )

    # Whether to load the training dataset
    if is_training:
        # Load training dataset
        try:
            model_trainingset_filename = f"{model_type}_DataSet" + set_dataset_filename(
                system_model_params, samples_size
            )
            train_dataset = read_data(
                datasets_path / "train" / model_trainingset_filename
            )
            datasets.append(train_dataset)
        except:
            raise Exception("load_datasets: Training dataset doesn't exist")
    # Load test dataset
    try:
        test_dataset = read_data(
            datasets_path / "test" / model_dataset_filename)
        datasets.append(test_dataset)
    except:
        raise Exception("load_datasets: Test dataset doesn't exist")

    # Load sps test dataset
    try:
        sps_test_dataset = read_data(
            datasets_path / "test" / sps_model_dataset_filename)
        datasets.append(sps_test_dataset)
    except:
        raise Exception("load_datasets: Test dataset doesn't exist")
    # Load generic test dataset
    try:
        generic_test_dataset = read_data(
            datasets_path / "test" / generic_dataset_filename
        )
        datasets.append(generic_test_dataset)
    except:
        raise Exception("load_datasets: Generic test dataset doesn't exist")
    # Load samples models
    try:
        samples_model = read_data(
            datasets_path / "test" / samples_model_filename)
        datasets.append(samples_model)
    except:
        raise Exception("load_datasets: Samples model dataset doesn't exist")
    return datasets


def set_dataset_filename(system_model_params: SystemModelParams, samples_size: float):
    """Returns the generic suffix of the datasets filename.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        samples_size (float): The size of the overall dataset.

    Returns:
    --------
        str: Suffix dataset filename
    """
    suffix_filename = (
        f"_{system_model_params.signal_type}_"
        + f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_"
        + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
        + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}"
        + ".h5"
    )
    return suffix_filename


def spatial_smoothing_covariance(X: torch.Tensor):
    """
    Calculates the covariance matrix using spatial smoothing technique.

    Args:
    -----
        X (np.ndarray): Input samples matrix.

    Returns:
    --------
        covariance_mat (np.ndarray): Covariance matrix.
    """
    # Define the sub-arrays size
    sub_array_size = int(X.shape[0] / 2) + 1
    # Define the number of sub-arrays
    number_of_sub_arrays = X.shape[0] - sub_array_size + 1
    # Initialize covariance matrix
    covariance_mat = torch.zeros((sub_array_size, sub_array_size)) + 1j * torch.zeros(
        (sub_array_size, sub_array_size)
    )
    for j in range(number_of_sub_arrays):
        # Run over all sub-arrays
        x_sub = X[j: j + sub_array_size, :]
        # Calculate sample covariance matrix for each sub-array
        sub_covariance = torch.cov(x_sub)
        # Aggregate sub-arrays covariances
        covariance_mat += sub_covariance
    # Divide overall matrix by the number of sources
    covariance_mat /= number_of_sub_arrays
    return covariance_mat
