# Location for utility functions that are used at various stages during the model training and validation process

import torch
import numpy as np
import time
import os
import Dataset_Generator


# load the full dataset into RAM
class Full_Load_Dataset(torch.utils.data.Dataset):

    def __init__(self, input_data, output_data, common_IDs):
        super(Full_Load_Dataset, self).__init__()
        self.input_data = np.array(input_data, dtype=np.float32)  # [batch, seq, features]
        self.output_data = np.array(output_data, dtype=np.float32)  # target values
        self.common_IDs = common_IDs  # just to track contacts

    # when called, return the data at the index. Taking from CPU memory and moving to GPU memory if available
    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        input_data = torch.from_numpy(self.input_data[index, :, :]).requires_grad_(True)
        output_data = torch.from_numpy(self.output_data[index, :, :]).requires_grad_(False)

        # move data to the GPU if available
        if torch.cuda.is_available():
            input_data = input_data.cuda()
            output_data = output_data.cuda()

        return input_data, output_data

    # number of samples in the dataset
    def __len__(self):
        return self.input_data.shape[0]


# this function will be used when the all data can be loaded into RAM (no lazy loading)
def Run_Train_Val(training_data, val_data, model, optimiser, loss_function, num_epochs=10,
                  batch_size=32, variable_names=None, scheduler=None, calculate_feature_importance=False,
                  num_permutations=10, units="N"):

    """
    Function to run the training and validation of a model. This function is only for when all the data can be loaded
    Parameters
    ----------
    training_data: needs to be a torch.utils.data.Dataset object (combine Full_Load_Dataset class and Dataset_Generator)
    val_data: needs to be a torch.utils.data.Dataset object
    model: pytorch model
    optimiser: predefined pytorch optimiser
    loss_function: predefined pytorch loss function
    num_epochs: number of epochs to train for (int)
    batch_size: batch size for training (int)
    variable_names: list of strings containing the names of the variables in the dataset
    scheduler: predefined pytorch scheduler [not yet supported]
    calculate_feature_importance: boolean to indicate if feature importance should be calculated
    num_permutations: number of permutations to use when calculating feature importance
    units: string to indicate the units of the output variable

    Returns: dictionary containing the training and validation loss for each epoch and the trained model
    -------
    """

    # create a RMSE loss to evaluate performance on incase the loss function is not RMSE
    RMSE_loss_func = torch.nn.MSELoss(reduction='none')

    if scheduler is not None:
        raise ValueError("Schedulers not yet supported")

    # initiate data loaders - again this is only for when all data can be loaded into RAM (need to use sampler if not)
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # print the number of samples in the training dataloader
    print(f"Number of samples in training dataloader: {len(train_loader.dataset)}")

    total_batches = len(train_loader)  # needed for the epoch progress bar

    # if using GPU then move everything that needs to be on the GPU to the GPU
    if torch.cuda.is_available():
        model = model.cuda()

    # initialise the loss and epoch lists
    train_loss = []
    val_loss = []

    training_start = time.time()  # start the timer for the training

    # training loop
    for epoch in range(num_epochs):
        print(f"\n----- Epoch: {epoch+1} of {num_epochs} -----")

        train_loss_list = []
        val_loss_list = []

        epoch_start = time.time()  # start the timer for the epoch

        for i, (inputs, labels) in enumerate(train_loader):

            force_predicted = model(inputs)  # forward pass to get outputs

            loss_per_sample = loss_function(force_predicted, labels).mean(dim=1)  # calculate the loss for each sample
            loss = loss_per_sample.mean()  # get the average loss across the whole batch
            loss.backward()  # backward pass to calculate the gradients

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip the gradients to avoid exploding

            optimiser.step()  # update the model weights
            optimiser.zero_grad()  # clear the gradients before the next batch

            progress = ((i + 1) / total_batches) * 100   # Epoch progress bar
            print(f"\rEpoch {epoch + 1}/{num_epochs} [{progress:.3f}%] current train loss = {np.sqrt(loss.detach().cpu().numpy()):.1f} N", end='')

        # calculate the training loss after all the training within this epoch has been completed
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(train_loader):
                force_predicted = model(inputs)
                loss_per_sample = RMSE_loss_func(force_predicted, labels).mean(dim=1)
                train_loss_list.append(loss_per_sample.detach().cpu().numpy())

        # training loss for this epoch
        train_loss_list = np.concatenate(train_loss_list)
        train_loss.append(np.mean(train_loss_list))

        # calculate the validation loss after all the training within this epoch has been completed
        estimated_output = []
        ground_truth_output = []
        RMSE_loss_list = []

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                force_predicted = model(inputs)
                loss_per_sample = RMSE_loss_func(force_predicted, labels).mean(dim=1)
                val_loss_list.append(loss_per_sample.detach().cpu().numpy())

                estimated_output.append(force_predicted.detach().cpu().numpy())
                ground_truth_output.append(labels.detach().cpu().numpy())
                RMSE_loss_list.append(np.sqrt(loss_per_sample.detach().cpu().numpy()))

        # validation loss for this epoch
        val_loss_list = np.concatenate(val_loss_list)
        val_loss.append(np.mean(val_loss_list))

        epoch_end = time.time()  # end the timer for the epoch
        epoch_time = epoch_end - epoch_start  # calculate the time taken for the epoch

        # print the average loss for each epoch
        print(f"--> Epoch {epoch + 1}/{num_epochs} Training Loss: {np.sqrt(train_loss[epoch]):.2f} {units}, Validation Loss: {np.sqrt(val_loss[epoch]):.2f} {units}")
        print(f"--> Epoch {epoch + 1}/{num_epochs} took {epoch_time/60:.1f} minutes")

        model.train()  # convert the model back to training mode for the next epoch

    # if using GPU then move the model back to the CPU
    if torch.cuda.is_available():
        model = model.cpu()

    # stack the lists into arrays
    estimated_output = np.vstack(estimated_output)
    estimated_output = np.squeeze(estimated_output, axis=2)

    ground_truth_output = np.vstack(ground_truth_output)
    ground_truth_output = np.squeeze(ground_truth_output, axis=2)

    RMSE_loss = np.concatenate(RMSE_loss_list)

    # create a dictionary of the necessary data to return from the function
    training_output_dict = {"train_loss": train_loss,
                            "val_loss": val_loss,
                            "model": model, # can save the model parameters to a file
                            "estimated_output": estimated_output,
                            "ground_truth_output": ground_truth_output,
                            "RMSE_loss": RMSE_loss,
                            "input_variable_names": variable_names}

    if calculate_feature_importance:
        print("\nCalculating feature importance...")
        feature_contributions_dict = Feature_Importance(training_output_dict, val_loader, num_permutations=num_permutations)
        training_output_dict["feature_contributions"] = feature_contributions_dict

    training_end = time.time()  # end the timer for the training
    print(f"\nFull training of {num_epochs} epochs took {(training_end-training_start)/60:.1f} minutes")

    return training_output_dict


# custom implementation of the captum feature permutation algorithm (built in doesn't work for time series)
def Feature_Importance(training_output_dict, val_dataloader, num_permutations=10):

    # get the trained model
    model = training_output_dict["model"]
    variable_names = training_output_dict["input_variable_names"]
    baseline_RMSE_loss = training_output_dict["RMSE_loss"]
    MSE_loss_func = torch.nn.MSELoss(reduction='none')

    contributions_dict = {}

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    with torch.no_grad():

        # highest level will iterate through the variables
        for variable_count, input_variable in enumerate(variable_names):

            all_permutations_PFI = []

            # within each variable we will iterate through the number of permutations
            for permutation in range(num_permutations):

                print(f"\rCalculating feature importance for {input_variable} [{variable_count+1}/{len(variable_names)}] - permutation: {permutation+1}/{num_permutations}", end='\r')

                all_RMSE = []

                # finally will iterate through the batches
                for i, (inputs, labels) in enumerate(val_dataloader):

                    # permute the current feature with random values
                    permuted_inputs = inputs.clone()  # shape: [batch, seq_len, num_features]
                    permuted_inputs[:, :, variable_count] = torch.rand(permuted_inputs.shape[0], permuted_inputs.shape[1])

                    # get the output of the model with the permuted feature
                    permuted_output = model(permuted_inputs).detach()  # shape: [batch, seq, 1]

                    # calculate the RMSE between the labels and the permuted output
                    loss_per_sample = MSE_loss_func(permuted_output, labels).mean(dim=1)
                    RMSE_per_sample = np.sqrt(loss_per_sample.detach().cpu().numpy())

                    # add the data from this batch to the list
                    all_RMSE.append(RMSE_per_sample)

                # stack the list into an array
                all_RMSE = np.concatenate(all_RMSE)

                # calculate the PFI for this permutation (ratio between permuted error and baseline error) and add to the list
                all_permutations_PFI.append(all_RMSE / baseline_RMSE_loss)

            # stack the lists into an array and calculate the PFI for each sample across the permutations
            all_permutations_PFI = np.stack(all_permutations_PFI, axis=1)
            all_permutations_PFI = np.squeeze(all_permutations_PFI, axis=2)
            mean_PFI = np.mean(all_permutations_PFI, axis=1)  # calculate the average along the permutations axis
            # store the mean PFI across all samples and the SD of the PFI across all samples
            overall_mean_PFI = np.mean(mean_PFI)
            overall_SD_PFI = np.std(mean_PFI)

            # add the data to the dictionary
            contributions_dict[f"{input_variable}_mean"] = overall_mean_PFI
            contributions_dict[f"{input_variable}_SD"] = overall_SD_PFI

            print(f"{input_variable} - mean PFI: {overall_mean_PFI:.3f}, SD PFI: {overall_SD_PFI:.3f}")

    # move the model back to the CPU if it was on the GPU
    if torch.cuda.is_available():
        model = model.cpu()

    return contributions_dict


# function to run the feature importance evaluation after the training has already been completed
def post_train_feature_importance(LOSO_results_path, num_permutations=10):

    import ML_Models

    # get the name of every path in the results folder
    results_files = os.listdir(LOSO_results_path)
    all_participants = [file[5:9] for file in results_files]

    participant_start_time = time.time()

    for count, participant in enumerate(all_participants):

        print(f"======== RUNNING FEATURE IMPORTANCE FOR {participant} [{count+1}/{len(all_participants)}] ========")

        data_path = os.path.join(LOSO_results_path, results_files[count])
        LOSO_results = np.load(data_path, allow_pickle=True).item()

        train_participants = all_participants.copy()
        train_participants.remove(participant)
        val_participants = [participant]

        # get the training and validation data
        train_generator = Dataset_Generator.DatasetGenerator(participants=train_participants,
                                                             input_variables=LOSO_results['input_variables'],
                                                             output_variables=LOSO_results['output_variable'],
                                                             side=LOSO_results['side'])


        _, _, _ = train_generator.collect_dataset()
        train_input_data = train_generator.all_input_data
        train_output_data = train_generator.all_output_data
        train_contact_IDs = train_generator.all_contact_IDs

        # collect the validation data
        val_generator = Dataset_Generator.DatasetGenerator(participants=val_participants,
                                                           input_variables=LOSO_results['input_variables'],
                                                           output_variables=LOSO_results['output_variable'],
                                                           side=LOSO_results['side'])


        val_input_data, val_output_data, val_contact_IDs = val_generator.collect_dataset()

        # scale the datasets to Zscores
        dataset_processor = Dataset_Generator.DatasetProcessor(LOSO_results['input_variables'], train_input_data, val_input_data)
        scaled_training_input_data, scaled_val_input_data, _, stats_list = dataset_processor.convert_to_zscores(group_FSR=False)

        # convert the data to a pytorch dataset then a dataloader
        val_dataset = Full_Load_Dataset(scaled_val_input_data, val_output_data, val_contact_IDs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)

        # pass the data to the feature importance function
        feature_contributions_dict = Feature_Importance(LOSO_results, val_loader, num_permutations=num_permutations)

        # add the feature contributions to the LOSO results dictionary
        LOSO_results["feature_contributions"] = feature_contributions_dict

        # save the updated LOSO results dictionary
        np.save(data_path, LOSO_results)

        participant_end_time = time.time()
        participant_time = participant_end_time - participant_start_time
        print(f"======== FINISHED FEATURE IMPORTANCE FOR {participant} [{count+1}/{len(all_participants)}]     Total time for this participant = {participant_time:.0f} ========")

