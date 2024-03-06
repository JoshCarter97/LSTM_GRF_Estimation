
import torch
import torch.nn as nn
import numpy as np
import Training_Utility_Functions as Training_Utils
import os
import Dataset_Generator
import ML_Models


base_path = "/app"  # this is the base path for this repo's location (for example within the docker container)


# read in the participant details from the npy file
participant_details = np.load(os.path.join(base_path, "Dataset", "Info_Files", "participant_characteristics.npy"), allow_pickle=True).item()


participants = ["P002", "P003", "P004", "P006", "P007", "P008", "P009", "P010", "P011", "P012", "P013", "P014", "P015",
                "P016", "P017", "P018", "P019", "P020", "P021", "P022", "P023", "P024", "P025", "P026", "P027", "P028",
                "P029", "P030", "P031", "P032", "P033", "P034", "P035", "P036", "P037", "P038", "P039", "P040", "P041",
                "P042", "P043", "P044", "P045", "P046", "P047", "P048", "P049", "P050"]


# loop through and create a list of participants in train and val set
for participant in participants:

    ### VARIABLES TO ADJUST ###
    input_variables = ["FSR_1", "FSR_2", "FSR_3", "FSR_4", "FSR_5", "FSR_6", "FSR_7", "FSR_8", "FSR_9", "FSR_10",
                       "FSR_11", "FSR_12", "FSR_13", "FSR_14", "FSR_15", "FSR_16", "Nurvv_CoPx", "Nurvv_CoPy",
                       "GCT", "mass", "speed", "incline", "insole_length", "Acc_x", "Acc_y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z"]

    target_variable = ["vGRF"]
    side = "left"

    trial_dict_path = os.path.join(base_path, "Dataset", "Info_Files", "processed_trials.npy")
    trial_dict = np.load(trial_dict_path, allow_pickle=True).item()

    EPOCHS = 3
    BATCH_SIZE = 16
    lr = 0.0001
    model = ML_Models.LSTM_Dropout(len(input_variables), hidden_size=256, input_dropout=0.2, linear_dropout=0.3,
                                      linear2_in=256, linear3_in=128, activation_function='leaky_relu')

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss(reduction='none')

    debug = False

    ###############################

    print(f"\n ============ Validating on participant {participant} ============")

    # current participant will be the validation participant (left out of training)
    val_participant = [participant]

    # The other 49 will be in the training set
    train_participants = participants.copy()
    train_participants.remove(participant)

    # just to speed up a full run through to make sure everything is working
    if debug:
        train_participants = train_participants[0:2]
        EPOCHS = 1

    # collect the training data
    train_generator = Dataset_Generator.DatasetGenerator(participants=train_participants,
                                                         input_variables=input_variables,
                                                         output_variables=target_variable,
                                                         side=side)

    train_generator.file_extensions = {"force_extension": "_segmented_f_norm_15Hz.npy",
                                       "FSR_extension": "_segmented_FSR_norm.npy",
                                       "NURVV_IMU_extension": "_segmented_Nurvv_IMU_norm.npy"}

    _, _, _ = train_generator.collect_dataset()
    train_input_data = train_generator.all_input_data
    train_output_data = train_generator.all_output_data
    train_contact_IDs = train_generator.all_contact_IDs

    # collect the validation data
    val_generator = Dataset_Generator.DatasetGenerator(participants=val_participant,
                                                       input_variables=input_variables,
                                                       output_variables=target_variable,
                                                       side=side)

    val_generator.file_extensions = {"force_extension": "_segmented_f_norm_15Hz.npy",
                                     "FSR_extension": "_segmented_FSR_norm.npy",
                                     "delsys_IMU_extension": "_segmented_IMU_norm.npy",
                                     "NURVV_IMU_extension": "_segmented_Nurvv_IMU_norm.npy",
                                     "moment_extension": "_moments_norm.npy"}

    val_input_data, val_output_data, val_contact_IDs = val_generator.collect_dataset()

    # scale the datasets to Zscores
    dataset_processor = Dataset_Generator.DatasetProcessor(input_variables, train_input_data, val_input_data)
    train_output_data, val_output_data = dataset_processor.change_sequence_length(101, train_output_data, val_output_data)  #### TESTING THIS ####
    scaled_training_input_data, scaled_val_input_data, _, stats_list = dataset_processor.convert_to_zscores(group_FSR=False)

    # convert the data to pytorch datasets
    train_dataset = Training_Utils.Full_Load_Dataset(scaled_training_input_data, train_output_data, train_contact_IDs)
    val_dataset = Training_Utils.Full_Load_Dataset(scaled_val_input_data, val_output_data, val_contact_IDs)

    training_output_dict = Training_Utils.Run_Train_Val(train_dataset, val_dataset, model, optimiser, loss_func,
                                                        num_epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                        variable_names=input_variables, calculate_feature_importance=False,
                                                        num_permutations=100, units="Nm")

    # add the input variables, output variable, and scaling stats to the dictionary
    training_output_dict["input_variables"] = input_variables
    training_output_dict["output_variable"] = target_variable
    training_output_dict["stats_list"] = stats_list
    training_output_dict["side"] = side
    training_output_dict["val_contact_IDs"] = val_contact_IDs  # to be used to evaulate the results in more detail (order should be correct as no shuffle)

    # save the outputs of the training
    save_directory = os.path.join(base_path, f"LOSO_Results_{target_variable[0]}")
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)

    save_path = os.path.join(save_directory, f"LOSO_{participant}inVal_training_outputs.npy")
    np.save(save_path, training_output_dict)



