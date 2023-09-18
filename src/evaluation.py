"""
DeepCNNDoa

Details
----------
Name: evaluation.py
Authors: D. H. Shmuel
Created: 21/07/23
Edited: 

Purpose
----------
This module provides functions for evaluating the performance of Deep learning DoA methods
,and conventional subspace methods. 
This scripts also defines function for plotting the methods spectrums.

Functions:
----------
evaluate_dnn_model: Evaluate the DNN model on a given dataset.
evaluate_model_based: Evaluate different model-based algorithms on a given dataset.
add_random_predictions: Add random predictions if the number of predictions
    is less than the number of sources.
evaluate: Wrapper function for model and algorithm evaluations.


"""
# Imports
import torch.nn as nn
from matplotlib import pyplot as plt
from src.utils import device
from src.criterions import RMSELoss, MSPELoss
from src.criterions import RMSPE, MSPE, RMSELoss, RMSE
from src.methods import MUSIC, RootMUSIC, Esprit, MVDR
from src.utils import *
from src.plotting import plot_spectrum


def evaluate_dnn_model(
    model,
    dataset: list,
    criterion: nn.Module,
    model_type: str = "DeppCNN",
    method="angels"
):
    """
    Evaluate the DNN model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        model_type (str, optional): The type of the model. Defaults to "DeepCNN".

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """

    # Initialize values
    overall_loss = 0.0
    test_length = 0
    # Set model to eval mode
    model.eval()
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for data in dataset:
            X, DOA = data
            test_length += DOA.shape[0]
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Get model output
            model_output = model(X)
            if model_type.startswith("DeepCNN"):
                # Deep CNN
                if isinstance(criterion, (RMSELoss, MSPELoss)):
                    # If evaluation performed over test set, loss is RMSPE / MSPE
                    DOA_predictions = model_output
                    # find peaks in the pseudo spectrum of probabilities
                    if method.startswith("peaks"):
                        DOA_predictions = (
                            get_k_peaks(DOA_predictions[0].shape[0], DOA.shape[1],
                                        DOA_predictions[0]) * D2R
                        )
                    else:
                        DOA_predictions = (
                            get_k_angles(DOA_predictions[0].shape[0], DOA.shape[1],
                                         DOA_predictions[0]) * D2R
                        )
                    DOA_predictions = DOA_predictions.view(
                        1, DOA_predictions.shape[0])
                elif isinstance(criterion, nn.BCELoss):
                    # If evaluation performed over validation set, loss is BCE
                    DOA_predictions = model_output
                else:
                    raise Exception(
                        f"evaluate_dnn_model: Loss criterion is not defined for {model_type} model"
                    )
            else:
                raise Exception(
                    f"evaluate_dnn_model: Model type {model_type} is not defined"
                )
            # Compute prediction loss
            # If the amount of predictions is less than the amount of sources
            if model_type.startswith("DeepCNN"):
                eval_loss = criterion(DOA_predictions.float(), DOA.float())
            # add the batch evaluation loss to epoch loss
            overall_loss += eval_loss.item()
        overall_loss = overall_loss / test_length
    return overall_loss


def evaluate_model_based(
    dataset: list,
    system_model,
    criterion: RMSPE,
    plot_spec=False,
    algorithm: str = "music",
    figures: dict = None,
):
    """
    Evaluate different model-based algorithms on a given dataset.

    Args:
        dataset (list): The evaluation dataset.
        system_model (SystemModel): The system model for the algorithms.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        plot_spec (bool): Whether to plot the spectrum for the algorithms. Defaults to False.
        algorithm (str): The algorithm to use (e.g., "music", "mvdr", "esprit", "r-music"). Defaults to "music".
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
        float: The average evaluation loss.

    Raises:
        Exception: If the algorithm is not supported.
    """
    # Initialize parameters for evaluation
    loss_list = []
    for i, data in enumerate(dataset):
        X, doa = data
        X = X[0]
        # Root-MUSIC algorithms
        if "r-music" in algorithm:
            root_music = RootMUSIC(system_model)
            if algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, roots, predictions_all, _, M = root_music.narrowband(
                    X=X, mode="spatial_smoothing"
                )
            else:
                # Conventional
                predictions, roots, predictions_all, _, M = root_music.narrowband(
                    X=X, mode="sample"
                )
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=predictions_all,
                    true_DOA=doa[0] * R2D,
                    roots=roots,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )
        # MUSIC algorithms
        elif "music" in algorithm:
            music = MUSIC(system_model)
            if algorithm.startswith("bb"):
                # Broadband MUSIC
                predictions, spectrum, M = music.broadband(X=X)
            elif algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, spectrum, M = music.narrowband(
                    X=X, mode="spatial_smoothing"
                )
            elif algorithm.startswith("music"):
                # Conventional
                predictions, spectrum, M = music.narrowband(X=X, mode="sample")
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=predictions,
                    true_DOA=doa * R2D,
                    system_model=system_model,
                    spectrum=spectrum,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )

        # ESPRIT algorithms
        elif "esprit" in algorithm:
            esprit = Esprit(system_model)
            if algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, M = esprit.narrowband(
                    X=X, mode="spatial_smoothing")
            else:
                # Conventional
                predictions, M = esprit.narrowband(X=X, mode="sample")
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)

        # MVDR algorithm
        elif algorithm.startswith("mvdr"):
            mvdr = MVDR(system_model)
            # Conventional
            _, spectrum = mvdr.narrowband(X=X, mode="sample")
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=None,
                    true_DOA=doa * R2D,
                    system_model=system_model,
                    spectrum=spectrum,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )
        else:
            raise Exception(
                f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
            )
    return np.mean(loss_list)


def add_random_predictions(M: int, predictions: np.ndarray):
    """
    Add random predictions if the number of predictions is less than the number of sources.

    Args:
        M (int): The number of sources.
        predictions (np.ndarray): The predicted DOA values.

    Returns:
        np.ndarray: The updated predictions with random values.

    """
    # Convert to np.ndarray array
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    while predictions.shape[0] < M:
        # print(f"{algorithm}: cant estimate M sources")
        predictions = np.insert(
            predictions, 0, np.round(
                np.random.rand(1) * 180, decimals=2) - 90.00
        )
    return predictions


def evaluate(
    model_regular: nn.Module,
    model_sps: nn.Module,
    model_type: str,
    model_test_dataset: list,
    generic_test_dataset: list,
    sps_model_test_dataset: list,
    criterion: nn.Module,
    subspace_criterion,
    system_model,
    figures: dict,
    plot_spec: bool = True,
    augmented_methods: list = None,
    subspace_methods: list = None,
):
    """
    Wrapper function for model and algorithm evaluations.

    Parameters:
        model (nn.Module): The DNN model.
        model_type (str): Type of the model.
        model_test_dataset (list): Test dataset for the model.
        generic_test_dataset (list): Test dataset for generic subspace methods.
        criterion (nn.Module): Loss criterion for (DNN) model evaluation.
        subspace_criterion: Loss criterion for subspace method evaluation.
        system_model: instance of SystemModel.
        figures (dict): Dictionary to store figures.
        plot_spec (bool, optional): Whether to plot spectrums. Defaults to True.
        augmented_methods (list, optional): List of augmented methods for evaluation.
            Defaults to None.
        subspace_methods (list, optional): List of subspace methods for evaluation.
            Defaults to None.

    Returns:
        None
    """
    # Set default model-based subspace methods
    if not isinstance(subspace_methods, list):
        subspace_methods = [
            "esprit",
            "music",
            "r-music",
            # "mvdr",
            "sps-r-music",
            "sps-esprit",
            "sps-music"
        ]
    # Evaluate DNN model performances
    model_test_loss = evaluate_dnn_model(
        model=model_sps,
        dataset=sps_model_test_dataset,
        criterion=criterion,
        model_type=model_type,
    )
    print(
        f"Spatial smoothing {model_type} with max probability solver Test loss = {model_test_loss * R2D }")

    # # Evaluate DNN model performances
    # model_test_loss = evaluate_dnn_model(
    #     model=model_sps,
    #     dataset=sps_model_test_dataset,
    #     criterion=criterion,
    #     model_type=model_type,
    #     method="peaks"
    # )
    # print(
    #     f"Spatial smoothing {model_type} with peak finding solver Test loss = {model_test_loss * R2D }")
    # Evaluate DNN model performances
    model_test_loss = evaluate_dnn_model(
        model=model_regular,
        dataset=model_test_dataset,
        criterion=criterion,
        model_type=model_type,
    )
    print(f"{model_type} with with max probability solver Test loss = {model_test_loss * R2D }")

    # model_test_loss = evaluate_dnn_model(
    #     model=model_regular,
    #     dataset=model_test_dataset,
    #     criterion=criterion,
    #     model_type=model_type,
    #     method="peaks"
    # )
    # print(f"{model_type} with with peak finding solver Test loss = {model_test_loss * R2D }")
    # print(f"{model_type} with peaks solver Test loss = {model_test_loss * R2D }")
    # Evaluate classical subspace methods
    for algorithm in subspace_methods:
        loss = evaluate_model_based(
            generic_test_dataset,
            system_model,
            criterion=subspace_criterion,
            plot_spec=plot_spec,
            algorithm=algorithm,
            figures=figures,
        )
        print("{} test loss = {}".format(algorithm.lower(), loss * R2D))
