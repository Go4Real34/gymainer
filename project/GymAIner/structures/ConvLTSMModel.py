from keras.models import Sequential
from keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping

from keras.utils import plot_model

from .DatasetHandler import DatasetHandler
from .ModelSettings import ModelSettings

class ConvLTSMModel:
    def __init__(self, settings):
        self.settings = settings
        
        self.all_settings = ModelSettings(dataset_path=self.settings["model"]["dataset_path"], 
                                          sequence_length=self.settings["model"]["sequence_length"], 
                                          validation_ratio=self.settings["model"]["validation_ratio"], 
                                          test_ratio=self.settings["model"]["test_ratio"], 
                                          filter_counts=self.settings["model"]["filter_counts"], 
                                          kernel_sizes=self.settings["model"]["kernel_sizes"], 
                                          activations=self.settings["model"]["activations"], 
                                          data_formats=self.settings["model"]["data_formats"], 
                                          recurrent_dropouts=self.settings["model"]["recurrent_dropouts"], 
                                          return_sequences=self.settings["model"]["return_sequences"], 
                                          pool_sizes=self.settings["model"]["pool_sizes"], 
                                          paddings=self.settings["model"]["paddings"], 
                                          dropouts=self.settings["model"]["dropouts"], 
                                          label_count=self.settings["model"]["label_count"], 
                                          
                                          early_stopping_callback_count=self.settings["early_stopping"]["count"], 
                                          early_stopping_callback_monitors=self.settings["early_stopping"]["monitors"], 
                                          early_stopping_callback_patiences=self.settings["early_stopping"]["patiences"], 
                                          early_stopping_callback_modes=self.settings["early_stopping"]["modes"], 
                                          early_stopping_callback_restore_best_weights=self.settings["early_stopping"]["restore_best_weights"])
        
        self.dataset_handler = DatasetHandler(self.all_settings.get_dataset_path(), 
                                              self.all_settings.get_sequence_length(), 
                                              self.all_settings.get_test_ratio())
        
        self.model = self.create_model()
        self.early_stopping_callbacks = self.create_early_stopping_callbacks()
        
        return
    
    def create_model(self):
        model = Sequential()
        
        model.add(ConvLSTM2D(filters=self.all_settings.get_filter_count(0), 
                             kernel_size=self.all_settings.get_kernel_size(0), 
                             activation=self.all_settings.get_activation(0), 
                             data_format=self.all_settings.get_data_format(0), 
                             recurrent_dropout=self.all_settings.get_recurrent_dropout(0), 
                             return_sequences=self.all_settings.get_return_sequence(0), 
                             input_shape = (self.dataset_handler.SEQUENCE_LENGTH, 
                                            self.dataset_handler.RESIZE_WIDTH, 
                                            self.dataset_handler.RESIZE_HEIGHT, 
                                            self.dataset_handler.COLOR_CHANNELS)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_pool_size(0), 
                               padding=self.all_settings.get_padding(0), 
                               data_format=self.all_settings.get_data_format(1)))
        
        model.add(TimeDistributed(Dropout(self.all_settings.get_dropout(0))))
        
        model.add(ConvLSTM2D(filters=self.all_settings.get_filter_count(1), 
                             kernel_size=self.all_settings.get_kernel_size(1), 
                             activation=self.all_settings.get_activation(1), 
                             data_format=self.all_settings.get_data_format(2), 
                             recurrent_dropout=self.all_settings.get_recurrent_dropout(1), 
                             return_sequences=self.all_settings.get_return_sequences(1)))

        model.add(MaxPooling3D(pool_size=self.all_settings.get_pool_size(1), 
                               padding=self.all_settings.get_padding(1), 
                               data_format=self.all_settings.get_data_format(3)))
        
        model.add(TimeDistributed(Dropout(self.all_settings.get_dropout(1))))
        
        model.add(ConvLSTM2D(filters=self.all_settings.get_filter_count(2), 
                             kernel_size=self.all_settings.get_kernel_size(2), 
                             activation=self.all_settings.get_activation(2), 
                             data_format=self.all_settings.get_data_format(4), 
                             recurrent_dropout=self.all_settings.get_recurrent_dropout(2), 
                             return_sequences=self.all_settings.get_return_sequences(2)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_pool_size(2), 
                               padding=self.all_settings.get_padding(2), 
                               data_format=self.all_settings.get_data_format(5)))
        
        model.add(TimeDistributed(Dropout(self.all_settings.get_dropout(2))))
        
        model.add(ConvLSTM2D(filters=self.all_settings.get_filter_count(3), 
                             kernel_size=self.all_settings.get_kernel_size(3), 
                             activation=self.all_settings.get_activation(3), 
                             data_format=self.all_settings.get_data_format(6), 
                             recurrent_dropout=self.all_settings.get_recurrent_dropout(3), 
                             return_sequences=self.all_settings.get_return_sequences(3)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_pool_size(3), 
                               padding=self.all_settings.get_padding(3), 
                               data_format=self.all_settings.get_data_format(7)))
        
        model.add(Flatten())
        
        model.add(Dense(units=self.all_settings.get_label_count(), 
                        activation=self.all_settings.get_activation(4)))

        model.summary()
        plot_model(model, show_shapes=True, show_layer_names=True)
        
        return model
    
    def create_early_stopping_callbacks(self):
        callbacks = []
        for index in range(self.all_settings.get_early_stopping_callback_count()):
            early_stopping_callback = EarlyStopping(monitor=self.all_settings.get_early_stopping_callback_modes(index), 
                                                    patience=self.all_settings.get_early_stopping_callback_patiences(index), 
                                                    mode=self.all_settings.get_early_stopping_callback_modes(index), 
                                                    restore_best_weights=self.all_settings.get_early_stopping_callback_restore_best_weights(index))
            callbacks.append(early_stopping_callback)
            
        return callbacks
    