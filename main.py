"""DeepCNN main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 30/06/23

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets 
        * training: For training the model
        * evaluate_dnn_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.evaluation import evaluate
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator

# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")

if __name__ == "__main__":
    # Initialize paths
    external_data_path = Path.cwd() / "data"
    scenario_data_path = "known_sources"
    datasets_path = external_data_path / "datasets" / scenario_data_path
    simulations_path = external_data_path / "simulations"
    saving_path = external_data_path / "weights"
    # Initialize time and date
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    # Operations commands
    commands = {
        "SAVE_TO_FILE": True,  # Saving results to file or present them over CMD
        "CREATE_DATA": True,  # Creating new dataset
        "LOAD_DATA": False,  # Loading data from exist dataset
        "LOAD_MODEL": True,  # Load specific model for training
        "TRAIN_MODEL": False,  # Applying training operation
        "SAVE_MODEL": False,  # Saving tuned model
        "EVALUATE_MODE": True,  # Evaluating desired algorithms
    }
    # Saving simulation scores to external file
    if commands["SAVE_TO_FILE"]:
        file_path = (
            simulations_path / "results" / "scores" /
            Path(dt_string_for_save + ".txt")
        )
        sys.stdout = open(file_path, "w")
    # Define system model parameters
    # for T in [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    # for T in [2000]:
    # for eta in [0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075]:
    for sigma in [0.075, 0.1, 0.3, 0.4, 0.5, 0.75]:
        # for gap in [1, 2, 4, 6, 10, 14]:
        # for snr in [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]:
        system_model_params = (
            SystemModelParams()
            .set_num_sensors(16)
            .set_num_sources(2)
            .set_num_observations(500)
            .set_snr(-10)
            .set_signal_type("NarrowBand")
            .set_signal_nature("non-coherent")
            # .set_sensors_dev(eta)
            .set_sv_noise(sigma)
        )
        # Generate model configuration
        model_config_regular = (
            ModelGenerator()
            .set_model_type("DeepCNN")
            .set_grid_size(121)
            .set_model(system_model_params.N)
        )
        model_config_sps = (
            ModelGenerator()
            .set_model_type("DeepCNN")
            .set_grid_size(121)
            .set_model(system_model_params.N // 2 + 1)
        )
        # Define samples size
        samples_size = 1000  # Overall dateset size
        train_test_ratio = 1  # training and testing datasets ratio
        # Sets simulation filename
        simulation_filename = get_simulation_filename(
            system_model_params=system_model_params, model_config=model_config_regular
        )
        # Print new simulation intro
        print("------------------------------------")
        print("---------- New Simulation ----------")
        print("------------------------------------")
        print("date and time =", dt_string)
        # Initialize seed
        set_unified_seed()
        # Datasets creation
        if commands["CREATE_DATA"]:
            # Define which datasets to generate
            create_training_data = False  # Flag for creating training data
            create_testing_data = True  # Flag for creating test data
            print("Creating Data...")
            if create_training_data:
                # Generate training dataset
                train_dataset, _, _ = create_dataset(
                    system_model_params=system_model_params,
                    samples_size=samples_size,
                    model_config=model_config_regular,
                    save_datasets=True,
                    datasets_path=datasets_path,
                    true_doa=None,
                    phase="train",
                )
            if create_testing_data:
                # Generate test dataset
                test_dataset, sps_test_dataset, generic_test_dataset, samples_model = create_dataset(
                    system_model_params=system_model_params,
                    samples_size=samples_size,
                    model_config=model_config_regular,
                    save_datasets=True,
                    datasets_path=datasets_path,
                    phase="test",
                    # doa_test_params={"start": -59.5,
                    #                  "end": 57.5,
                    #                  "gap": 2.11,
                    #                  "step": 1},
                    # doa_test_params={"start": -60,
                    #                  "end": 55,
                    #                  "gap": 4.7,
                    #                  "step": 1},
                    doa_test_params={"start": -90,
                                     "end": 85,
                                     "gap": 4.7,
                                     "step": 1},
                    exp=2,
                    true_doa=[10.11, 13.3]
                    # true_doa=[-13.18, -9.58]
                    # true_doa=[-13.18, -13.18 + gap]
                )
        # Datasets loading
        elif commands["LOAD_DATA"]:
            (
                train_dataset,
                test_dataset,
                generic_test_dataset,
                samples_model,
            ) = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config_regular.model_type,
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
                is_training=True,
            )

        # Training stage
        if commands["TRAIN_MODEL"]:
            # Assign the training parameters object
            simulation_parameters = (
                TrainingParams()
                .set_batch_size(256)
                .set_epochs(100)
                .set_model(model_config=model_config_regular)
                .set_optimizer(optimizer="Adam", learning_rate=0.00001, weight_decay=1e-9)
                .set_training_dataset(train_dataset)
                .set_schedular(step_size=100, gamma=0.1)
                .set_criterion()
            )
            if commands["LOAD_MODEL"]:
                simulation_parameters.load_model(
                    loading_path=saving_path / "final_models" / simulation_filename
                )
            # Print training simulation details
            simulation_summary(
                system_model_params=system_model_params,
                model_type=model_config_regular.model_type,
                parameters=simulation_parameters,
                phase="training",
            )
            # Perform simulation training and evaluation stages
            model, loss_train_list, loss_valid_list = train(
                training_parameters=simulation_parameters,
                model_name=simulation_filename,
                saving_path=saving_path,
            )
            # Save model weights
            if commands["SAVE_MODEL"]:
                torch.save(
                    model.state_dict(),
                    saving_path / "final_models" / Path(simulation_filename),
                )
            # Plots saving
            if commands["SAVE_TO_FILE"]:
                plt.savefig(
                    simulations_path
                    / "results"
                    / "plots"
                    / Path(dt_string_for_save + r".png")
                )
            else:
                pass
                # plt.show()

        # Evaluation stage
        if commands["EVALUATE_MODE"]:
            # Initialize figures dict for plotting
            figures = initialize_figures()
            # Define loss measure for evaluation
            criterion, subspace_criterion = set_criterions("rmse")
            # Load datasets for evaluation
            if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
                test_dataset, sps_test_dataset, generic_test_dataset, samples_model = load_datasets(
                    system_model_params=system_model_params,
                    model_type=model_config_regular.model_type,
                    samples_size=samples_size,
                    datasets_path=datasets_path,
                    train_test_ratio=train_test_ratio,
                )

            # Generate DataLoader objects
            model_test_dataset = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False, drop_last=False
            )
            sps_model_test_dataset = torch.utils.data.DataLoader(
                sps_test_dataset, batch_size=1, shuffle=False, drop_last=False
            )
            generic_test_dataset = torch.utils.data.DataLoader(
                generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
            )
            # Load pre-trained model
            if not commands["TRAIN_MODEL"]:
                # Define an evaluation parameters instance
                simulation_parameters_regular = (
                    TrainingParams()
                    .set_model(model_config=model_config_regular)
                    .load_model(
                        loading_path=saving_path / "final_models" /
                        Path(simulation_filename + "_regular")
                    )
                )
                simulation_parameters_sps = (
                    TrainingParams()
                    .set_model(model_config=model_config_sps)
                    .load_model(
                        loading_path=saving_path / "final_models" /
                        Path(f'DeepCNN_grid_size={model_config_sps.grid_size}' + "_sps")
                    )
                )

                model_regular = simulation_parameters_regular.model
                model_sps = simulation_parameters_sps.model
            # print simulation summary details
            simulation_summary(
                system_model_params=system_model_params,
                model_type=model_config_regular.model_type,
                phase="evaluation",
                parameters=simulation_parameters_regular,
            )
            # Evaluate DNN models, augmented and subspace methods
            evaluate(
                model_regular=model_regular,
                model_sps=model_sps,
                model_type=model_config_regular.model_type,
                model_test_dataset=model_test_dataset,
                sps_model_test_dataset=sps_model_test_dataset,
                generic_test_dataset=generic_test_dataset,
                criterion=criterion,
                subspace_criterion=subspace_criterion,
                system_model=samples_model,
                figures=figures,
                plot_spec=False,
            )
        # plt.show()
        print("end")
