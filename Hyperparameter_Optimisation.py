import optuna
import Training_Utility_Functions as Training_Utils
import os
import Dataset_Generator
import torch
import numpy as np
import logging
import ML_Models


logging_file_name = "optuna_logs.txt"  # file name for where the hyperparameter optimisation output logs will be saved
base_path = "/app"  # this is the base path for this repo's location (for example within the docker container)

## SET PARAMATERS OF THE MODEL TO BE OPTIMISED ##
input_variables = ["FSR_1", "FSR_2", "FSR_3", "FSR_4", "FSR_5", "FSR_6", "FSR_7", "FSR_8", "FSR_9", "FSR_10", "FSR_11",
                   "FSR_12", "FSR_13", "FSR_14", "FSR_15", "FSR_16", "Nurvv_CoPx", "Nurvv_CoPy", "GCT", "mass", "speed",
                   "incline", "insole_length", "Acc_x", "Acc_y", "Acc_z"]

target_variable = ["vGRF"]
side = "left"
##################################################

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this will be the objective for a single fold of the cross fold validation
def objective(trial, train_loader, val_loader, input_variables):

    # choose the parameters to optimise
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 1, 15)

    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128, 256, 512])
    input_dropout = trial.suggest_float("input_dropout", 0.1, 0.5)
    linear_dropout = trial.suggest_float("linear_dropout", 0.1, 0.5)
    linear3_size = trial.suggest_categorical("linear3_size", [32, 64, 128, 256, 512])
    linear4_size = trial.suggest_categorical("linear4_size", [32, 64, 128, 256, 512])
    activation_function = trial.suggest_categorical("activation_function", ["leaky_relu", "relu", "tanh", "sigmoid"])

    # define the model and pass the parameters to the model
    model = ML_Models.LSTM_Dropout(len(input_variables), hidden_size=lstm_hidden_size, input_dropout=input_dropout,
                                      linear_dropout=linear_dropout, linear2_in=linear3_size, linear3_in=linear4_size,
                                      activation_function=activation_function)

    model = model.to(device)

    # define the optimiser and pass the parameters to the optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss(reduction='none')

    # initialise the loss and epoch lists
    val_loss = []

    # training loop for epoch in range(epochs)
    for epoch in range(epochs):

        train_loss_list = []
        val_loss_list = []

        model.train()
        print("\n")

        for i, (inputs, labels) in enumerate(train_loader):
            force_predicted = model(inputs)  # forward pass to get outputs

            loss_per_sample = loss_function(force_predicted, labels).mean(dim=1)  # calculate the loss for each sample
            loss = loss_per_sample.mean()  # get the average loss across the whole batch
            loss.backward()  # backward pass to calculate the gradients

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip the gradients to avoid them exploding or vanishing

            optimiser.step()  # update the model weights
            optimiser.zero_grad()  # clear the gradients before the next batch

            print(f"Epoch: {epoch + 1}/{epochs} | Batch: {i + 1}/{len(train_loader)} | Loss: {np.sqrt(loss.item()):.2f} N", end="\r")

        # calculate the validation loss after all the training within this epoch has been completed
        estimated_output = []
        ground_truth_output = []
        RMSE_loss_list = []

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                force_predicted = model(inputs)
                loss_per_sample = loss_function(force_predicted, labels).mean(dim=1)
                val_loss_list.append(loss_per_sample.detach().cpu().numpy())

                estimated_output.append(force_predicted.detach().cpu().numpy())
                ground_truth_output.append(labels.detach().cpu().numpy())
                RMSE_loss_list.append(np.sqrt(loss_per_sample.detach().cpu().numpy()))

        # validation loss for this epoch
        val_loss_list = np.concatenate(val_loss_list)
        val_loss.append(np.mean(val_loss_list))

        # calculate the val loss at the end of each epoch
        current_val_loss = np.mean(val_loss_list)
    print("\n\n")

    # return the current val loss
    return current_val_loss

def objective_cv(trial):

    # Define the participants in each fold and place all folds into a single list
    fold1 = ["P006", "P007", "P019", "P011", "P014"]
    fold2 = ["P016", "P035", "P029", "P022", "P039"]
    fold3 = ["P012", "P002", "P043", "P047", "P050"]
    fold4 = ["P008", "P038", "P041", "P030", "P024"]
    fold5 = ["P004", "P031", "P020", "P025", "P027"]

    all_folds = [fold1, fold2, fold3, fold4, fold5]

    fold_losses = []

    for fold_count, fold in enumerate(all_folds):

        val_participants = fold
        training_participants = all_folds.copy()

        # remove the current fold from the training participants and condense the list of lists into a single list
        training_participants.remove(fold)
        training_participants = [item for sublist in training_participants for item in sublist]

        # collect the training data
        train_generator = Dataset_Generator.DatasetGenerator(participants=training_participants,
                                                             input_variables=input_variables,
                                                             output_variables=target_variable,
                                                             side=side,
                                                             base_path=base_path)

        train_input_data, train_output_data, train_contact_IDs = train_generator.collect_dataset()

        # collect the validation data
        val_generator = Dataset_Generator.DatasetGenerator(participants=val_participants,
                                                           input_variables=input_variables,
                                                           output_variables=target_variable,
                                                           side=side,
                                                           base_path=base_path)

        val_input_data, val_output_data, val_contact_IDs = val_generator.collect_dataset()

        # scale the data to Zscores
        dataset_processor = Dataset_Generator.DatasetProcessor(input_variables, train_input_data, val_input_data)
        scaled_training_input_data, scaled_val_input_data, _, stats_list = dataset_processor.convert_to_zscores(group_FSR=False)

        # convert the data to pytorch datasets
        train_dataset = Training_Utils.Full_Load_Dataset(scaled_training_input_data, train_output_data, train_contact_IDs)
        val_dataset = Training_Utils.Full_Load_Dataset(scaled_val_input_data, val_output_data, val_contact_IDs)

        # initialise the dataloaders
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

        # run the objective function for a single fold
        fold_loss = objective(trial, train_loader, val_loader, input_variables)

        # add the loss for this fold to the list of losses
        fold_losses.append(fold_loss)

    # print the mean loss for all folds and the hyperparameters used
    print(f"\n\n--> Mean loss across all folds: {np.sqrt(np.mean(fold_losses)):.2f} N\n")

    return np.mean(fold_losses)


# Set up logging
logging_file_path = os.path.join(base_path, logging_file_name)
logging.basicConfig(filename=logging_file_path, level=logging.INFO)


def callback(study, trial):
    logging.info(f"\nTrial {trial.number} finished with value: {np.sqrt(trial.value):.3f} N")
    logging.info(f"Params: {trial.params}\n")


study = optuna.create_study(direction="minimize")
study.optimize(objective_cv, n_trials=250, timeout=150000, callbacks=[callback])


print("Best trial:")
trial = study.best_trial

print(f"Loss: {np.sqrt(trial.value)} N")

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))



