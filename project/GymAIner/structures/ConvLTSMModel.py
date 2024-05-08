from keras.models import Sequential
from keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping

from keras.utils import plot_model

from .DatasetHandler import DatasetHandler
from .ModelSettings import ModelSettings

class ConvLTSMModel:
    def __init__(self, settings):
        self.settings = settings
        
        return
    

    def init(self):
        self.all_settings = ModelSettings(dataset_path=self.settings["dataset"]["path"], 
                                          dataset_sequence_length=self.settings["dataset"]["sequence_length"], 
                                          dataset_label_count=self.settings["dataset"]["label_count"], 
                                          dataset_validation_ratio=self.settings["dataset"]["validation_ratio"], 
                                          dataset_test_ratio=self.settings["dataset"]["test_ratio"], 
                                          
                                          model_filter_counts=self.settings["model"]["filter_counts"], 
                                          model_kernel_sizes=self.settings["model"]["kernel_sizes"], 
                                          model_activations=self.settings["model"]["activations"], 
                                          model_data_formats=self.settings["model"]["data_formats"], 
                                          model_recurrent_dropouts=self.settings["model"]["recurrent_dropouts"], 
                                          model_return_sequences=self.settings["model"]["return_sequences"], 
                                          model_pool_sizes=self.settings["model"]["pool_sizes"], 
                                          model_paddings=self.settings["model"]["paddings"], 
                                          model_dropouts=self.settings["model"]["dropouts"], 
                                          
                                          early_stopping_callback_count=self.settings["early_stopping"]["count"], 
                                          early_stopping_callback_monitors=self.settings["early_stopping"]["monitors"], 
                                          early_stopping_callback_patiences=self.settings["early_stopping"]["patiences"], 
                                          early_stopping_callback_modes=self.settings["early_stopping"]["modes"], 
                                          early_stopping_callback_restore_best_weights=self.settings["early_stopping"]["restore_best_weights"], 
                                          
                                          compile_loss=self.settings["compile"]["loss"], 
                                          compile_optimizer=self.settings["compile"]["optimizer"], 
                                          compile_metrics=self.settings["compile"]["metrics"])
        
        self.dataset_handler = DatasetHandler(self.all_settings.get_dataset_path(), 
                                              self.all_settings.get_dataset_sequence_length(), 
                                              self.all_settings.get_dataset_test_ratio())
        
        self.model = self.create_model()
        self.early_stopping_callbacks = self.create_early_stopping_callbacks()
        self.model.compile(loss=self.all_settings.get_compile_loss(), 
                           optimizer=self.all_settings.get_compile_optimizer(), 
                           metrics=self.all_settings.get_compile_metrics())
        
        return
    

    def create_model(self):
        model = Sequential()
        
        model.add(ConvLSTM2D(filters=self.all_settings.get_model_filter_count(0), 
                             kernel_size=self.all_settings.get_model_kernel_size(0), 
                             activation=self.all_settings.get_model_activation(0), 
                             data_format=self.all_settings.get_model_data_format(0), 
                             recurrent_dropout=self.all_settings.get_model_recurrent_dropout(0), 
                             return_sequences=self.all_settings.get_model_return_sequence(0), 
                             input_shape = (self.dataset_handler.SEQUENCE_LENGTH, 
                                            self.dataset_handler.RESIZE_WIDTH, 
                                            self.dataset_handler.RESIZE_HEIGHT, 
                                            self.dataset_handler.COLOR_CHANNELS)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_model_pool_size(0), 
                               padding=self.all_settings.get_padding(0), 
                               data_format=self.all_settings.get_model_data_format(1)))
        
        model.add(TimeDistributed(Dropout(self.all_settings.get_model_dropout(0))))
        
        model.add(ConvLSTM2D(filters=self.all_settings.get_model_filter_count(1), 
                             kernel_size=self.all_settings.get_model_kernel_size(1), 
                             activation=self.all_settings.get_model_activation(1), 
                             data_format=self.all_settings.get_model_data_format(2), 
                             recurrent_dropout=self.all_settings.get_model_recurrent_dropout(1), 
                             return_sequences=self.all_settings.get_return_sequences(1)))

        model.add(MaxPooling3D(pool_size=self.all_settings.get_model_pool_size(1), 
                               padding=self.all_settings.get_padding(1), 
                               data_format=self.all_settings.get_model_data_format(3)))
        
        model.add(TimeDistributed(Dropout(self.all_settings.get_model_dropout(1))))
        
        model.add(ConvLSTM2D(filters=self.all_settings.get_model_filter_count(2), 
                             kernel_size=self.all_settings.get_model_kernel_size(2), 
                             activation=self.all_settings.get_model_activation(2), 
                             data_format=self.all_settings.get_model_data_format(4), 
                             recurrent_dropout=self.all_settings.get_model_recurrent_dropout(2), 
                             return_sequences=self.all_settings.get_return_sequences(2)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_model_pool_size(2), 
                               padding=self.all_settings.get_padding(2), 
                               data_format=self.all_settings.get_model_data_format(5)))
        
        model.add(TimeDistributed(Dropout(self.all_settings.get_model_dropout(2))))
        
        model.add(ConvLSTM2D(filters=self.all_settings.get_model_filter_count(3), 
                             kernel_size=self.all_settings.get_model_kernel_size(3), 
                             activation=self.all_settings.get_model_activation(3), 
                             data_format=self.all_settings.get_model_data_format(6), 
                             recurrent_dropout=self.all_settings.get_model_recurrent_dropout(3), 
                             return_sequences=self.all_settings.get_return_sequences(3)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_model_pool_size(3), 
                               padding=self.all_settings.get_padding(3), 
                               data_format=self.all_settings.get_model_data_format(7)))
        
        model.add(Flatten())
        
        model.add(Dense(units=self.all_settings.get_dataset_label_count(), 
                        activation=self.all_settings.get_model_activation(4)))

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
    