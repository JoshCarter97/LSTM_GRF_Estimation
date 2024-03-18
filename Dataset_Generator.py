import numpy as np
import os
import copy

class DatasetGenerator:

    def __init__(self, participants, input_variables, output_variables, base_path="/app", side="left"):
        self.participants = participants
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.side = side
        self.base_path = base_path
        self.trial_dict_basepath = os.path.join(base_path, "Dataset", "Info_Files")
        self.participant_characteristics_path = os.path.join(base_path, "Dataset", "Info_Files", "participant_characteristics.npy")
        self.sequence_length = 400
        self.file_extensions = {"force_extension": "_segmented_f_norm_15Hz.npy",
                                "FSR_extension": "_segmented_FSR_norm.npy",
                                "NURVV_IMU_extension": "_segmented_Nurvv_IMU_norm_10Hz.npy"}

        self.trial_dict_names = {"force_extension": "processed_trials.npy",
                                 "FSR_extension": "processed_trials.npy",
                                 "NURVV_IMU_extension": "Nurvv_processed_trials.npy"}


        # the list of possible variables that can be selected by the user and the extension they are saved in
        self.variable_locations = {"vGRF": "force_extension", "apGRF": "force_extension", "mlGRF": "force_extension",
                                   "CoPx": "force_extension", "CoPy": "force_extension", "GCT": "force_extension",
                                   "insole_length": "participant_characteristics", "mass": "participant_characteristics",
                                   "height": "participant_characteristics", "age": "participant_characteristics",
                                   "speed": "trial_name", "incline": "trial_name", "FSR_1": "FSR_extension",
                                   "FSR_2": "FSR_extension", "FSR_3": "FSR_extension", "FSR_4": "FSR_extension",
                                   "FSR_5": "FSR_extension", "FSR_6": "FSR_extension", "FSR_7": "FSR_extension",
                                   "FSR_8": "FSR_extension", "FSR_9": "FSR_extension", "FSR_10": "FSR_extension",
                                   "FSR_11": "FSR_extension", "FSR_12": "FSR_extension", "FSR_13": "FSR_extension",
                                   "FSR_14": "FSR_extension", "FSR_15": "FSR_extension", "FSR_16": "FSR_extension",
                                   "FSRsSum": "FSR_extension", "Nurvv_CoPx": "calculated", "Nurvv_CoPy": "calculated",
                                   "Acc_x": "NURVV_IMU_extension", "Acc_y": "NURVV_IMU_extension",
                                   "Acc_z": "NURVV_IMU_extension", "Gyro_x": "NURVV_IMU_extension",
                                   "Gyro_y": "NURVV_IMU_extension", "Gyro_z": "NURVV_IMU_extension"}

        self.discrete_variables = ["GCT", "insole_length", "mass", "height", "age", "speed", "incline"]
        self.calculated_variables = ["Nurvv_CoPx", "Nurvv_CoPy"]

        if self.__class__.__name__ == "DatasetGenerator":
            # check that all variables are valid
            for variable in input_variables:
                if variable not in self.variable_locations.keys():
                    raise ValueError(f"Chosen Variable: {variable} is not a valid variable. "
                                     f"\n\nPlease choose from: {self.variable_locations.keys()}\n")
            for variable in output_variables:
                if variable not in self.variable_locations.keys():
                    raise ValueError(f"Chosen Variable: {variable} is not a valid variable. "
                                     f"\n\nPlease choose from: {self.variable_locations.keys()}\n")

        # if the side intput is not left, right or both, raise an error
        if side not in ["left", "right", "both"]:
            raise ValueError(f"Input to side: {side} is not valid. Please choose from: ['left', 'right', 'both']")

    def _get_common_trials(self):

        # based on the file extensions that have been covered by the participants and trials, get the list of
        # file extensions that will be needed to load the data
        file_categories = []
        for variable in self.input_variables:
            if self.variable_locations[variable] not in file_categories:
                if self.variable_locations[variable] in self.file_extensions.keys():
                    file_categories.append(self.variable_locations[variable])

        for variable in self.output_variables:
            if self.variable_locations[variable] not in file_categories:
                if self.variable_locations[variable] in self.file_extensions.keys():
                    file_categories.append(self.variable_locations[variable])

        trial_dicts_tocheck = np.unique(np.array([self.trial_dict_names[file] for file in file_categories]))

        if len(trial_dicts_tocheck) == 1:
            # There is only one trial dict to check, so simply set self.trial_dict and return.
            self.trial_dict = np.load(os.path.join(self.trial_dict_basepath,
                                                   trial_dicts_tocheck[0]), allow_pickle=True).item()
            return

        trial_dicts = [np.load(os.path.join(self.trial_dict_basepath, trial_dict), allow_pickle=True).item() for trial_dict in trial_dicts_tocheck]

        common_trials_dict = {}

        for participant in self.participants:
            print(f"Finding common trials for participant: {participant}...", end="\r")

            current_participant_trials = [set(trial_dict[participant]) for trial_dict in trial_dicts]

            # Use the intersection of sets directly
            common_trials = set.intersection(*current_participant_trials)

            common_trials_dict[participant] = np.array(list(common_trials))

        self.trial_dict = common_trials_dict

    def _get_participant_characteristics(self):

        self.characteristics_dict = np.load(self.participant_characteristics_path, allow_pickle=True).item()

    def _get_discrete_variable(self, variable_name, participant, trial):

        # going to use the force data to find out the number of samples to create (assuming that has most IDs)
        file_path = os.path.join(self.base_path, "Dataset", participant, f"{participant}_{trial}_1{self.file_extensions['force_extension']}")
        force_data = np.load(file_path, allow_pickle=True).item()

        # will already have a value for each sample here
        if variable_name == "GCT":

            if self.side == "both":
                contact_IDs = np.vstack((force_data["left"]["contact_IDs"].reshape(force_data["left"]["contact_IDs"].shape[0], 1),
                                         force_data["right"]["contact_IDs"].reshape(force_data["right"]["contact_IDs"].shape[0], 1)))

                L_GCTs = np.array(force_data["left"]["GCT"])
                R_GCTs = np.array(force_data["right"]["GCT"])

                single_variable_data = np.vstack((L_GCTs.reshape(L_GCTs.shape[0], 1), R_GCTs.reshape(R_GCTs.shape[0], 1)))

            else:
                contact_IDs = force_data[self.side]["contact_IDs"]
                single_variable_data = np.array(force_data[self.side]["GCT"])

            # duplicate the single variable data to the number of samples in the time series
            data_array = np.tile(single_variable_data.reshape(contact_IDs.shape[0], 1), (1, self.sequence_length))

        # as these values are the same for each sample, we can just duplicate them to the number of samples
        else:
            if variable_name in ["insole_length", "mass", "height", "age"]:
                self._get_participant_characteristics()
                discrete_value = self.characteristics_dict[participant][variable_name]

            # trial name and participant characteristics
            elif variable_name == "speed":
                self._get_participant_characteristics()
                norm_speed = self.characteristics_dict[participant]["norm_speed"]

                if "slow" in trial: discrete_value = 0.9 * norm_speed
                elif "fast" in trial: discrete_value = 1.1 * norm_speed
                elif "norm" in trial: discrete_value = norm_speed
                elif "fixed" in trial: discrete_value = 12.0
                else:
                    raise ValueError(f"Trial: {trial} does not have a valid speed. "
                                     f"\n\nPlease choose from: ['slow', 'fast', 'norm', 'fixed']")

            # trial name
            elif variable_name == "incline":

                if "flat" in trial: discrete_value = 1.0
                elif "up" in trial: discrete_value = 6.0
                elif "down" in trial: discrete_value = -4.0
                else:
                    raise ValueError(f"Trial: {trial} does not have a valid incline. "
                                     f"\n\nPlease choose from: ['flat', 'up', 'down']")

            # duplicate the discrete value to number of samples
            if self.side == "both":
                L_contact_IDs = force_data["left"]["contact_IDs"].reshape(force_data["left"]["contact_IDs"].shape[0], 1)
                R_contact_IDs = force_data["right"]["contact_IDs"].reshape(force_data["right"]["contact_IDs"].shape[0], 1)

                # duplicate the discrete value to number of samples
                L_discrete_value = np.full((L_contact_IDs.shape[0], 1), discrete_value)
                R_discrete_value = np.full((R_contact_IDs.shape[0], 1), discrete_value)

                # contact IDs are ready for export
                contact_IDs = np.vstack((L_contact_IDs, R_contact_IDs))

                # variable data still only has one time point, so need to duplicate it to number of samples
                single_variable_data = np.vstack((L_discrete_value, R_discrete_value))

            else:
                contact_IDs = force_data[self.side]["contact_IDs"]
                single_variable_data = np.full((force_data[self.side]["contact_IDs"].shape[0], 1), discrete_value)

            # duplicate the single variable data to the number of samples in the time series
            data_array = np.tile(single_variable_data.reshape(contact_IDs.shape[0], 1), (1, self.sequence_length))

        return contact_IDs, data_array

    def _get_variable(self, variable_name, participant, trial):

        if variable_name in self.discrete_variables:
            all_contact_IDs, all_variable_data = self._get_discrete_variable(variable_name, participant, trial)
            return all_contact_IDs, all_variable_data

        # get the extension of the file
        try:
            extension_group = self.variable_locations[variable_name]
            extension = self.file_extensions[extension_group]
        except KeyError:
            raise KeyError(f"Variable: {variable_name} is not a valid variable. Please choose from: "
                           f"{self.variable_locations.keys()}")

        file_path = os.path.join(self.base_path, "Dataset", participant, f"{participant}_{trial}_1{extension}")
        data = np.load(file_path, allow_pickle=True).item()

        # if we are getting nurvv data then we don't need to deal with the first layer of the dict being sides
        if "NURVV" in extension_group:
            if self.side == "both":
                raise ValueError("Requesting a NURVV variable whilst both sides is selected (NURVV only left)")

            contact_IDs = data["contact_IDs"]  # get the contact IDs
            data_array = data[variable_name]  # get the variable data

        # otherwise it will be force, or FSR - where we have the side layer of dict first
        else:

            if self.side == "both":
                contact_IDs_L = data["left"]["contact_IDs"].reshape(data["left"]["contact_IDs"].shape[0], 1)
                variable_data_L = data["left"][variable_name]
                contact_IDs_R = data["right"]["contact_IDs"].reshape(data["right"]["contact_IDs"].shape[0], 1)
                variable_data_R = data["right"][variable_name]

                contact_IDs = np.vstack((contact_IDs_L, contact_IDs_R))
                data_array = np.vstack((variable_data_L, variable_data_R))

            else:
                contact_IDs = data[self.side]["contact_IDs"]
                data_array = data[self.side][variable_name]

            if contact_IDs.shape[0] != data_array.shape[0]:
                print(f"\n!WARNING! Variable: {variable_name} has shape: {data_array.shape}, with contact IDs shape: {contact_IDs.shape}")

        return contact_IDs, data_array

    # method will be called when an input metric is being calculated from multiple other variables
    def _get_calculated_variable(self, variable_name, participant, trial):

        nurvv_cop_names = ["Nurvv_CoPx", "Nurvv_CoPy"]
        FSR_channel_names = [f"FSR_{i}" for i in range(1, 17)]
        characteristics_data = np.load(self.participant_characteristics_path, allow_pickle=True).item()
        insole_size = characteristics_data[participant]["Insole_size"]
        save_flag = False  # flag to identify if this is the first calculation and whether to save the data

        if variable_name in nurvv_cop_names:

            CoP_axis = variable_name[-1]  # get the axis from the variable name

            # we need to use the FSR data to calculate the CoP
            extension = self.file_extensions["FSR_extension"]
            file_path = os.path.join(self.base_path, "Dataset", participant, f"{participant}_{trial}_1{extension}")
            FSR_data = np.load(file_path, allow_pickle=True).item()

            # get the contact IDs and FSR data
            if self.side == "both":
                contact_IDs_L = FSR_data["left"]["contact_IDs"].reshape(FSR_data["left"]["contact_IDs"].shape[0], 1)
                contact_IDs_R = FSR_data["right"]["contact_IDs"].reshape(FSR_data["right"]["contact_IDs"].shape[0], 1)
                contact_IDs = np.vstack((contact_IDs_L, contact_IDs_R))

                # check if the data has already been calculated and just use that if so
                try:
                    CoP_data_L = FSR_data["left"][variable_name]
                    CoP_data_R = FSR_data["right"][variable_name]

                # otherwise we need to calculate it for the first time around
                except KeyError:

                    save_flag = True

                    FSR_data_L = np.stack([FSR_data["left"][channel] for channel in FSR_channel_names], axis=2)
                    FSR_data_R = np.stack([FSR_data["right"][channel] for channel in FSR_channel_names], axis=2)

                    # calculate the CoP for each side separately and then stack the outputs together
                    current_cop_calculator = CoP_Calculator(insole_size, "left")
                    CoP_data_L = current_cop_calculator.calculate_CoP_1D(FSR_data_L, CoP_axis)
                    FSR_data["left"][variable_name] = CoP_data_L  # save this data to the file for future use

                    current_cop_calculator = CoP_Calculator(insole_size, "right")
                    CoP_data_R = current_cop_calculator.calculate_CoP_1D(FSR_data_R, CoP_axis)
                    FSR_data["right"][variable_name] = CoP_data_R  # save this data to the file for future use

                CoP_data = np.vstack((CoP_data_L, CoP_data_R))

            # just taking data from one side, do not need to worry about stacking data
            else:
                contact_IDs = FSR_data[self.side]["contact_IDs"]

                try:
                    CoP_data = FSR_data[self.side][variable_name]

                except KeyError:
                    save_flag = True

                    # stack the FSR data into a 3D array [samples, seq_len, channels]
                    current_FSR_data = np.stack([FSR_data[self.side][channel] for channel in FSR_channel_names], axis=2)
                    current_cop_calculator = CoP_Calculator(insole_size, self.side)
                    CoP_data = current_cop_calculator.calculate_CoP_1D(current_FSR_data, CoP_axis)
                    FSR_data[self.side][variable_name] = CoP_data  # save this data to the file for future use

        if save_flag:
            # save the calculated data back to the file for future use
            np.save(file_path, FSR_data)
            print(f"have calculated {variable_name} for participant {participant} and trial {trial}, saving it back to the same file -> {file_path}")

        return contact_IDs, CoP_data

    # Intention for this method is to collect data from all chosen variables for a single trial, structuring it in this
    # way will speed up the removal of non-common IDs
    def _collect_full_trial(self, participant, trial):

        ## INPUT ##
        all_data = []  # store the data for each input variable for this participant and trial
        all_contact_IDs = []  # store the contact IDs for each input variable for this participant and trial

        for variable_count, variable in enumerate(self.input_variables):

            # get the extension of the file
            try:
                extension_group = self.variable_locations[variable]
            except KeyError:
                raise KeyError(f"Variable: {variable} is not a valid variable. Please choose from: "
                               f"{self.variable_locations.keys()}")

            # find out if it is a discrete variable
            if variable in self.discrete_variables:
                cur_contact_IDs, cur_data_array = self._get_discrete_variable(variable, participant, trial)
                all_contact_IDs.append(cur_contact_IDs)
                all_data.append(cur_data_array)

            elif variable in self.calculated_variables:
                cur_contact_IDs, cur_data_array = self._get_calculated_variable(variable, participant, trial)
                all_contact_IDs.append(cur_contact_IDs)
                all_data.append(cur_data_array)

            else:
                cur_contact_IDs, cur_data_array = self._get_variable(variable, participant, trial)
                all_contact_IDs.append(cur_contact_IDs)
                all_data.append(cur_data_array)


        # convert the contact IDs to a column vector
        all_contact_IDs = [contact_IDs.reshape(contact_IDs.shape[0], 1) for contact_IDs in all_contact_IDs]

        ## OUTPUT ##
        output_contact_IDs, output_variable_data = self._get_variable(self.output_variables[0], participant, trial)
        output_contact_IDs = output_contact_IDs.reshape(output_contact_IDs.shape[0], 1)

        ## DELETE NON COMMON CONTACT IDS ##
        # input

        # need a list with input and output contact IDs to find the common subset of contact IDs
        all_contact_IDs.append(output_contact_IDs)

        common_contact_IDs = all_contact_IDs[0] # need to initiate the common contact IDs search with the first set

        # find the common contact IDs between all data sources
        for contact_IDs in all_contact_IDs[1:]:
            common_contact_IDs = np.intersect1d(common_contact_IDs, contact_IDs)

        # loop through each variables data deleting the non common contact IDs
        for variable_count, variable in enumerate(self.input_variables):

            # find the contacts that need to be removed from the current array (non common)
            non_common_contact_IDs = np.setdiff1d(all_contact_IDs[variable_count], common_contact_IDs)

            # Find the indices of elements to delete
            delete_indices = np.where(np.isin(all_contact_IDs[variable_count], non_common_contact_IDs))[0]

            # Delete the rows using the delete_indices
            all_data[variable_count] = np.delete(all_data[variable_count], delete_indices, axis=0)
            all_contact_IDs[variable_count] = np.delete(all_contact_IDs[variable_count], delete_indices, axis=0)

        # output
        non_common_contact_IDs = np.setdiff1d(output_contact_IDs, common_contact_IDs)

        # Find the indices of elements to delete
        delete_indices = np.where(np.isin(output_contact_IDs, non_common_contact_IDs))[0]

        # Delete the rows using the delete_indices
        output_variable_data = np.delete(output_variable_data, delete_indices, axis=0)
        all_contact_IDs[-1] = np.delete(all_contact_IDs[-1], delete_indices, axis=0)  # the last item of all contact ids is those from the output variable

        # add a new axis so that the output has the same number of dimensions as the input [samples, seq_len, 1]
        output_variable_data = output_variable_data[:, :, np.newaxis]

        # stack data from each variable along a third axis after non-common arrays have been removed
        all_data = np.stack(all_data, axis=2)

        # check that each array within all_contact_IDs are the same
        for i in range(1, len(all_contact_IDs)):
            if not np.array_equal(all_contact_IDs[i], all_contact_IDs[i-1]):
                raise ValueError("Not all contact IDs are the same for each variable. This should not be possible at this stage.")

            # input data
        return all_data, output_variable_data, all_contact_IDs[0]

    def collect_dataset(self):

        self._get_participant_characteristics()

        print(f"Collecting data for participants: {self.participants}")

        # each element in the list will be data from all variables for a single trial, to be stacked in axis=0
        input_data_list = []
        output_data_list = []
        contact_IDs_list = []

        # get the common trials between all participants
        self._get_common_trials()

        for participant in self.participants:
            trials = self.trial_dict[participant]

            # check if the first element of trials is of np array type, if yes then take that array out
            if trials[0].__class__.__name__ == "ndarray":
                trials = trials[0]

            for trial in trials:

                print(f"Collecting data for participant: {participant} and trial: {trial}...", end="\r")
                input_data, output_data, contact_IDs = self._collect_full_trial(participant, trial)

                input_data_list.append(input_data)
                output_data_list.append(output_data)
                contact_IDs_list.append(contact_IDs)

        # stack the data (already been screened for common IDs)
        all_input_data = np.vstack(input_data_list)
        all_output_data = np.vstack(output_data_list)
        all_contact_IDs = np.vstack(contact_IDs_list)

        # save all the data to this object for use by future methods
        self.all_input_data = all_input_data
        self.all_output_data = all_output_data
        self.all_contact_IDs = all_contact_IDs

        return all_input_data, all_output_data, all_contact_IDs



class DatasetProcessor:

    def __init__(self, input_variables, training_input, validation_input, test_input=None):
        self.input_variables = input_variables
        self.training_input = training_input
        self.validation_input = validation_input
        self.test_input = test_input

        # the list of possible variables that can be selected by the user and the extension they are saved in
        self.variable_locations = {"vGRF": "force_extension", "apGRF": "force_extension", "mlGRF": "force_extension",
                                   "CoPx": "force_extension", "CoPy": "force_extension", "GCT": "force_extension",
                                   "insole_length": "participant_characteristics", "mass": "participant_characteristics",
                                   "height": "participant_characteristics", "age": "participant_characteristics",
                                   "speed": "trial_name", "incline": "trial_name", "FSR_1": "FSR_extension",
                                   "FSR_2": "FSR_extension", "FSR_3": "FSR_extension", "FSR_4": "FSR_extension",
                                   "FSR_5": "FSR_extension", "FSR_6": "FSR_extension", "FSR_7": "FSR_extension",
                                   "FSR_8": "FSR_extension", "FSR_9": "FSR_extension", "FSR_10": "FSR_extension",
                                   "FSR_11": "FSR_extension", "FSR_12": "FSR_extension", "FSR_13": "FSR_extension",
                                   "FSR_14": "FSR_extension", "FSR_15": "FSR_extension", "FSR_16": "FSR_extension",
                                   "FSRsSum": "FSR_extension",
                                   "Acc_x": "NURVV_IMU_extension", "Acc_y": "NURVV_IMU_extension",
                                   "Acc_z": "NURVV_IMU_extension", "Gyro_x": "NURVV_IMU_extension",
                                   "Gyro_y": "NURVV_IMU_extension", "Gyro_z": "NURVV_IMU_extension"}

        self.discrete_variables = ["insole_length", "mass", "height", "age", "speed", "incline"]

    # helper method that does the actual conversion of the ADC values to load
    def _FSR_to_load_conversion(self, data_array):

        # convert the ADC values to voltage
        data_array = (data_array / 65536) * 2.8

        # convert the voltage to load
        data_array = np.exp(data_array - 0.155) / 0.463

        return data_array

    # helper function for the 'change_sequence_length' method
    def _resample_3D_array(self, new_sequence_length, array):

        # create x arrays for the original and new signals
        orig_x = np.linspace(0, array.shape[1] - 1, array.shape[1])
        new_x = np.linspace(0, array.shape[1] - 1, new_sequence_length)

        # create an empty array to store the resampled data
        resampled_array = np.empty((array.shape[0], new_sequence_length, array.shape[2]))

        # loop through each sample and feature and resample the data
        for feature_num in range(array.shape[2]):
            for sample_num in range(array.shape[0]):
                print(f"Resampling sample: {sample_num+1} of {array.shape[0]}, feature: {feature_num+1} of {array.shape[2]}", end="\r")
                resampled_array[sample_num, :, feature_num] = np.interp(new_x, orig_x, array[sample_num, :, feature_num])

        return resampled_array


    # use the np.interp function to resample the data to the new sequence length
    def change_sequence_length(self, new_sequence_length, training_output, validation_output, test_output=None):

        # check that the new sequence length is an integer
        new_sequence_length = int(new_sequence_length)

        # normalise all train and val input and output data
        self.training_input = self._resample_3D_array(new_sequence_length, self.training_input)
        self.validation_input = self._resample_3D_array(new_sequence_length, self.validation_input)

        training_output = self._resample_3D_array(new_sequence_length, training_output)
        validation_output = self._resample_3D_array(new_sequence_length, validation_output)

        # do the same for the test data if it exists
        if self.test_input is not None and test_output is not None:
            self.test_input = self._resample_3D_array(new_sequence_length, self.test_input)
            test_output = self._resample_3D_array(new_sequence_length, test_output)

        if test_output is not None:
            return training_output, validation_output, test_output
        else:
            return training_output, validation_output


    # The intention of this method is to convert the raw FSR output values ADC into 'Load' values via conversion to Voltage
    # the equations being used for the conversion are taken from benchtop testing completed by NURVV
    def convert_FSR_to_Load(self):

        # if the input variables do not contain any names with FSR in then raise an error
        if not any("FSR" in variable for variable in self.input_variables):
            raise ValueError("Chosen input variables does not contain any FSR data, so can not convert to load")

        # get the indices of the FSR variables
        FSR_indices = [i for i, variable in enumerate(self.input_variables) if "FSR" in variable]

        for channel in FSR_indices:

            self.training_input[:, :, channel] = self._FSR_to_load_conversion(self.training_input[:, :, channel])
            self.validation_input[:, :, channel] = self._FSR_to_load_conversion(self.validation_input[:, :, channel])

            if self.test_input is not None:
                self.test_input[:, :, channel] = self._FSR_to_load_conversion(self.test_input[:, :, channel])

        # make a string with all the FSR variables in
        FSR_channels = []
        for channel in FSR_indices: FSR_channels.append(self.input_variables[channel])

        print(f"FSR data has been converted to load for the following channels in the input data: {FSR_channels}")


    # method that deals with data in a single array in isolation [samples, seq_len] and time series data only
    # this implementation ignores time, and averages value across the time series and across samples
    def _zscore_time_series(self, training_array, validation_array, report_stats=False, test_array=None):

        # calculate the mean and SD of the training data
        training_mean = np.mean(training_array)
        training_SD = np.std(training_array)

        # subtract the mean and divide by the SD for each array
        training_array = (training_array - training_mean) / training_SD
        validation_array = (validation_array - training_mean) / training_SD

        if test_array is not None:
            test_array = (test_array - training_mean) / training_SD

        if report_stats:
            return training_array, validation_array, test_array, [training_mean, training_SD]
        else:
            return training_array, validation_array, test_array

    # method for converting discrete variables to zscores, get each unique value from the first time point
    def _zscore_discrete(self, training_array, validation_array, report_stats=False, test_array=None):

        # get the unique values from the first time point
        unique_values = np.unique(training_array[:, 0])

        # calculate the mean and SD of the training data
        training_mean = np.mean(unique_values)
        training_SD = np.std(unique_values)

        # this can happen if using very few participants and the unique values are all the same (i.e. insole size)
        if training_SD == 0.0: training_SD = 0.0000000001

        # subtract the mean and divide by the SD for each array
        training_array = (training_array - training_mean) / training_SD
        validation_array = (validation_array - training_mean) / training_SD

        if test_array is not None:
            test_array = (test_array - training_mean) / training_SD

        if report_stats:
            return training_array, validation_array, test_array, [training_mean, training_SD]
        else:
            return training_array, validation_array, test_array

    def _group_zscores(self, training_array_section, validation_array_section, report_stats=False, test_array_section=None):

        # calculate the mean and SD of the whole training data (subsection)
        training_mean = np.mean(training_array_section)
        training_SD = np.std(training_array_section)

        # subtract the mean and divide by the SD for each array
        training_array_section = (training_array_section - training_mean) / training_SD
        validation_array_section = (validation_array_section - training_mean) / training_SD

        if test_array_section is not None:
            test_array_section = (test_array_section - training_mean) / training_SD

        if report_stats:
            return training_array_section, validation_array_section, test_array_section, [training_mean, training_SD]
        else:
            return training_array_section, validation_array_section, test_array_section

    def convert_to_zscores(self, group_FSR=False):

        scaled_training_data = []
        scaled_validation_data = []
        scaled_test_data = []
        stats_list = []

        # make a copy of the input variables list to work with as will need to remove variables if being dealth with differently
        completed_variables_indexes = []

        # !making the assumption that the FSR data will be grouped next to each other in the input variables list
        if group_FSR:

            print("Converting FSR data to zscores as a group")

            # identify the channels in the data that contain FSR data
            FSR_indices = [i for i, variable in enumerate(self.input_variables) if "FSR" in variable]
            for index in FSR_indices: completed_variables_indexes.append(index)

            # take out all the data from the FSR channels
            training_FSR_data = self.training_input[:, :, FSR_indices]
            validation_FSR_data = self.validation_input[:, :, FSR_indices]
            if self.test_input is not None: test_FSR_data = self.test_input[:, :, FSR_indices]
            else: test_FSR_data = None

            # convert to z scores as a group using the helper method
            scaled_training_FSR_data, scaled_validation_FSR_data, scaled_test_FSR_data, stats = self._group_zscores(training_FSR_data, validation_FSR_data, report_stats=True, test_array_section=test_FSR_data)

            # add the scaled data to the list ready to be stacked together
            scaled_training_data.append(scaled_training_FSR_data)
            scaled_validation_data.append(scaled_validation_FSR_data)
            if self.test_input is not None: scaled_test_data.append(scaled_test_FSR_data)
            stats_list.append(stats)

        for variable_count, variable in enumerate(self.input_variables):

            # if the variable has already been dealt with then skip it
            if variable_count in completed_variables_indexes: continue

            print(f"Converting variable: {variable} to zscores...", end="\r")

            # if the variable is a discrete variable then use the discrete zscore method
            if variable in self.discrete_variables:
                training_array = self.training_input[:, :, variable_count]
                validation_array = self.validation_input[:, :, variable_count]

                if self.test_input is not None: test_array = self.test_input[:, :, variable_count]
                else: test_array = None

                training_array, validation_array, test_array, stats = self._zscore_discrete(training_array, validation_array, report_stats=True, test_array=test_array)

            # otherwise use the time series zscore method
            else:
                training_array = self.training_input[:, :, variable_count]
                validation_array = self.validation_input[:, :, variable_count]

                if self.test_input is not None: test_array = self.test_input[:, :, variable_count]
                else: test_array = None

                training_array, validation_array, test_array, stats = self._zscore_time_series(training_array, validation_array, report_stats=True, test_array=test_array)

            # add scaled data to the list ready to be stacked together (need to create a new axis for the stacking)
            scaled_training_data.append(training_array[:, :, np.newaxis])
            scaled_validation_data.append(validation_array[:, :, np.newaxis])
            if self.test_input is not None: scaled_test_data.append(test_array[:, :, np.newaxis])
            stats_list.append(stats)

        # stack the scaled data together along the last axis
        scaled_training_data = np.concatenate(scaled_training_data, axis=2)
        scaled_validation_data = np.concatenate(scaled_validation_data, axis=2)
        if self.test_input is not None: scaled_test_data = np.stack(scaled_test_data, axis=2)

        if scaled_test_data is None:
            return scaled_training_data, scaled_validation_data, stats_list
        else:
            return scaled_training_data, scaled_validation_data, scaled_test_data, stats_list



# All the code needed to be able to calculate CoP position from the NURVV insole
sensor_locations_dict = {"XS": {"left": {"Horizontal": [-10.00, 10.00, 10.65, 10.65, 10.65, 21.34, 19.00, 8.11, 8.58, -7.35, -8.82, -16.29, -20.75, -10.65, -10.65, -10.00],
                                         "Vertical":   [21.00,  21.00, 51.00, 81.00, 111.0, 147.0, 185.0, 158.0, 200.0, 191.0, 156.0,  176.0, 143.5, 111.00, 81.00,  51.00]},
                                "right": {"Horizontal": [10.00, -10.00, -10.65, -10.65, -10.65, -21.34, -19.00, -8.11, -8.58, 7.35, 8.82, 16.29, 20.75, 10.65, 10.65, 10.00],
                                          "Vertical":  [21.00,  21.00, 51.00, 81.00, 111.0, 147.0, 185.0, 158.0, 200.0, 191.0, 156.0,  176.0, 143.5, 111.00, 81.00,  51.00]}},

                         "S": {"left": {"Horizontal": [-10.00, 10.00, 10.00, 10.00, 10.00, 23.00, 19.00, 8.00, 8.00, -7.00, -11.00, -17.00, -25.00, -10.00, -10.00, -10.00],
                                        "Vertical":   [21.00, 21.00, 51.00, 86.00, 121.00, 156.00, 204.00, 167.00, 220.00, 211.00, 165.00, 196.50, 152.5, 121.00, 86.00, 51.00]},
                               "right": {"Horizontal": [10.00, -10.00, -10.00, -10.00, -10.00, -23.00, -19.00, -8.00, -8.00, 7.00, 11.00, 17.00, 25.00, 10.00, 10.00, 10.00],
                                         "Vertical": [[21.00, 21.00, 51.00, 86.00, 121.00, 156.00, 204.00, 167.00, 220.00, 211.00, 165.00, 196.50, 152.5, 121.00, 86.00, 51.00]]}},

                         "M": {"left": {"Horizontal": [-12.00, 12.00, 12.00, 12.00, 12.00, 29.50, 24.00, 11.50, 11.50, -2.50, -6.50, -12.50, -24.50, -12.00, -12.00, -12.00],
                                        "Vertical": [20.00, 20.00, 55.00, 94.00, 129.00, 174.00, 216.00, 185.00, 231.00, 222.00, 183.00, 207.00, 171.00, 129.00, 94.00, 55.00]},
                               "right": {"Horizontal": [12.00, -12.00, -12.00, -12.00, -12.00, -29.50, -24.00, -11.50, -11.50, 2.50, 6.50, 12.50, 24.50, 12.00, 12.00, 12.00],
                                         "Vertical": [20.00, 20.00, 55.00, 94.00, 129.00, 174.00, 216.00, 185.00, 231.00, 222.00, 183.00, 207.00, 171.00, 129.00, 94.00, 55.00]}},

                         "L": {"left": {"Horizontal": [-15.00, 15.00, 15.00, 13.50, 11.00, 28.75, 30.75, 11.25, 15.75, -1.25, -9.25, -14.25, -23.25, -14.00, -16.50, -15.00],
                                        "Vertical": [30.00, 30.00, 65.00, 100.00, 135.00, 180.00, 236.00, 190.00, 248.00, 238.00, 188.00, 223.00, 172.00, 135.00, 100.00, 65.00]},
                               "right": {"Horizontal": [15.00, -15.00, -15.00, -13.50, -11.00, -28.75, -30.75, -11.25, -15.75, 1.25, 9.25, 14.25, 23.25, 14.00, 16.50, 15.00],
                                         "Vertical": [30.00, 30.00, 65.00, 100.00, 135.00, 180.00, 236.00, 190.00, 248.00, 238.00, 188.00, 223.00, 172.00, 135.00, 100.00, 65.00]}},

                         "XL": {"left": {"Horizontal": [-15.00, 15.00, 15.00, 15.00, 15.00, 31.00, 29.00, 12.00, 14.00, -2.00, -7.00, -16.00, -26.00, -15.00, -15.00, -15.00],
                                         "Vertical": [26.00, 26.00, 66.00, 106.00, 146.00, 196.00, 252.00, 206.00, 264.00, 254.00, 204.00, 239.00, 190.00, 146.00, 106.00, 66.00]},
                                "right": {"Horizontal": [15.00, -15.00, -15.00, -15.00, -15.00, -31.00, -29.00, -12.00, -14.00, 2.00, 7.00, 16.00, 26.00, 15.00, 15.00, 15.0],
                                          "Vertical": [26.00, 26.00, 66.00, 106.00, 146.00, 196.00, 252.00, 206.00, 264.00, 254.00, 204.00, 239.00, 190.00, 146.00, 106.00, 66.00]}},

                         "XXL": {"left": {"Horizontal": [-15.00, 15.00, 15.00, 15.00, 15.00, 32.00, 26.00, 12.00, 10.00, -5.00, -10.00, -20.00, -30.00, -15.00, -15.00, -15.00],
                                          "Vertical": [26.00, 26.00, 66.00, 106.00, 146.00, 206.00, 267.00, 216.00, 281.00, 269.00, 204.00, 254.00, 198.00, 146.00, 106.00, 66.00]},
                                 "right": {"Horizontal": [15.00, -15.00, -15.00, -15.00, -15.00, -32.00, -26.00, -12.00, -10.00, 5.00, 10.00, 20.00, 30.00, 15.00, 15.00, 15.00],
                                           "Vertical": [26.00, 26.00, 66.00, 106.00, 146.00, 206.00, 267.00, 216.00, 281.00, 269.00, 204.00, 254.00, 198.00, 146.00, 106.00, 66.00]}}}


class CoP_Calculator():
    def __init__(self, insole_size, insole_side):
        self.insole_size = insole_size
        self.insole_side = insole_side

        # get the horizontal and vertical distances of each sensor from the origin position
        self.X_locs = sensor_locations_dict[self.insole_size][self.insole_side]["Horizontal"]
        self.Y_locs = sensor_locations_dict[self.insole_size][self.insole_side]["Vertical"]

    def Organise_FSR_data(self, FSR_data_dict):

        # Assumption is that within the data dictionary FSR data for each channel is stored as a numpy array
        # (rows = foot contact, cols = time [normalised]) stored in the dict as "FSR_1" etc. and "FSRsSum"

        # calculate how many contacts are in the input dataset and create an output dictionary for each one
        num_contacts = FSR_data_dict["FSR_1"].shape[0]

        # calculate the normalised length of each contact
        contact_length = FSR_data_dict["FSR_1"].shape[1]
        print(f"organising the data for {num_contacts} contacts, with {contact_length} data points each to allow CoP calculation...")

        # create an input dictionary that organises the data into the correct format for the CoP calculation
        FSR_input_dict = {}

        for i in range(num_contacts):
            contact_data = np.zeros((contact_length, 17))

            for x in range(16):
                contact_data[:, x] = FSR_data_dict[f"FSR_{x+1}"][i, :]

            contact_data[:, 16] = FSR_data_dict["FSRsSum"][i, :]

            FSR_input_dict[f"Contact_{i}"] = contact_data

        self.FSR_input_dict = FSR_input_dict

        # initialise an output dictionary to store calculated Centre of Pressure values
        self.CoP_output_dict = {}

    def Calculate_CoP(self):

        for contact in self.FSR_input_dict.keys():

                # get the FSR data for the current contact
                FSR_data = self.FSR_input_dict[contact]

                # calculate the total pressure for the current contact
                total_pressure = FSR_data[:, 16]

                # initialise a 2D numpy array with seq_len rows and 2 columns to store the CoP values
                CoP_output = np.zeros((FSR_data.shape[0], 2))

                # calculate the CoP for each time point
                for i in range(FSR_data.shape[0]):
                    # calculate the horizontal and vertical CoP for the current time point
                    CoP_output[i, 0] = np.sum(FSR_data[i, :16] * self.X_locs) / total_pressure[i]  # X CoP
                    CoP_output[i, 1] = np.sum(FSR_data[i, :16] * self.Y_locs) / total_pressure[i]  # Y CoP

                    # store the CoP values in the output dictionary saved as a 2D numpy array where column 1 is CoP X and column 2 is CoP Y
                    self.CoP_output_dict[contact] = CoP_output

        return self.CoP_output_dict

    # intention is this method only calculates either X or Y and returns it (needed for ML utils class - DatasetGenerator)
    def calculate_CoP_1D(self, FSR_array, axis):

        # FSR_array input will be a 3D array with shape [num_contacts, seq_len, 16]
        all_CoP_outputs_list = []

        for contact in range(FSR_array.shape[0]):
            # get the FSR data for the current contact
            FSR_data = FSR_array[contact, :, :]

            # calculate the total pressure for the current contact
            total_pressure = np.sum(FSR_data[:, :16], axis=1)

            CoP_output = np.zeros(FSR_data.shape[0])

            # calculate the CoP for each time point
            for i in range(FSR_data.shape[0]):
                # calculate the horizontal and vertical CoP for the current time point
                if axis == "x":
                    CoP_output[i] = np.sum(FSR_data[i, :16] * self.X_locs) / total_pressure[i]
                elif axis == "y":
                    CoP_output[i] = np.sum(FSR_data[i, :16] * self.Y_locs) / total_pressure[i]

            all_CoP_outputs_list.append(CoP_output)

        # stack the list of CoP outputs into a 2D numpy array
        all_CoP_outputs_array = np.stack(all_CoP_outputs_list, axis=0)

        return all_CoP_outputs_array


    # Takes the starting position of the CoP away from all the CoP positions for each contact (will allow comparison between different insole sizes)
    def Calculate_relative_CoP(self):

        # check if Calculate_CoP has been run yet
        if not hasattr(self, "CoP_output_dict"):
            print("Calculate_CoP must be run first")
            return

        self.relative_CoP_output_dict = copy.deepcopy(self.CoP_output_dict)

        for contact in self.CoP_output_dict.keys():
            # get the starting CoP position
            start_CoP_X = self.CoP_output_dict[contact][0, 0]
            start_CoP_Y = self.CoP_output_dict[contact][0, 1]

            # take the starting CoP position from all the CoP positions for the current contact
            self.relative_CoP_output_dict[contact][:, 0] = self.CoP_output_dict[contact][:, 0] - start_CoP_X
            self.relative_CoP_output_dict[contact][:, 1] = self.CoP_output_dict[contact][:, 1] - start_CoP_Y

        return self.relative_CoP_output_dict



