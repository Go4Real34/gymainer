from keras.models import Sequential
from keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense

from keras.utils import plot_model

from .DatasetHandler import DatasetHandler
from .ModelSettings import ModelSettings

class ConvLTSMModel:
    def __init__(self, settings):
        self.settings = settings
        
        self.model_settings = ModelSettings(dataset_path=self.settings["dataset_path"], 
                                            sequence_length=self.settings["sequence_length"], 
                                            validation_ratio=self.settings["validation_ratio"], 
                                            test_ratio=self.settings["test_ratio"], 
                                            filter_counts=self.settings["filter_counts"], 
                                            kernel_sizes=self.settings["kernel_sizes"], 
                                            activations=self.settings["activations"], 
                                            data_formats=self.settings["data_formats"], 
                                            recurrent_dropouts=self.settings["recurrent_dropouts"], 
                                            return_sequences=self.settings["return_sequences"], 
                                            pool_sizes=self.settings["pool_sizes"], 
                                            paddings=self.settings["paddings"], 
                                            dropouts=self.settings["dropouts"], 
                                            label_count=self.settings["label_count"])
        
        self.dataset_handler = DatasetHandler(self.model_settings.get_dataset_path(), 
                                              self.model_settings.get_sequence_length(), 
                                              self.model_settings.get_test_ratio())
        
        self.model = self.create_model()
        
        return
    
    def create_model(self):
        model = Sequential()
        
        model.add(ConvLSTM2D(filters=self.model_settings.get_filter_count(0), 
                             kernel_size=self.model_settings.get_kernel_size(0), 
                             activation=self.model_settings.get_activation(0), 
                             data_format=self.model_settings.get_data_format(0), 
                             recurrent_dropout=self.model_settings.get_recurrent_dropout(0), 
                             return_sequences=self.model_settings.get_return_sequence(0), 
                             input_shape = (self.dataset_handler.SEQUENCE_LENGTH, 
                                            self.dataset_handler.RESIZE_WIDTH, 
                                            self.dataset_handler.RESIZE_HEIGHT, 
                                            self.dataset_handler.COLOR_CHANNELS)))
        
        model.add(MaxPooling3D(pool_size=self.model_settings.get_pool_size(0), 
                               padding=self.model_settings.get_padding(0), 
                               data_format=self.model_settings.get_data_format(1)))
        
        model.add(TimeDistributed(Dropout(self.model_settings.get_dropout(0))))
        
        model.add(ConvLSTM2D(filters=self.model_settings.get_filter_count(1), 
                             kernel_size=self.model_settings.get_kernel_size(1), 
                             activation=self.model_settings.get_activation(1), 
                             data_format=self.model_settings.get_data_format(2), 
                             recurrent_dropout=self.model_settings.get_recurrent_dropout(1), 
                             return_sequences=self.model_settings.get_return_sequences(1)))

        model.add(MaxPooling3D(pool_size=self.model_settings.get_pool_size(1), 
                               padding=self.model_settings.get_padding(1), 
                               data_format=self.model_settings.get_data_format(3)))
        
        model.add(TimeDistributed(Dropout(self.model_settings.get_dropout(1))))
        
        model.add(ConvLSTM2D(filters=self.model_settings.get_filter_count(2), 
                             kernel_size=self.model_settings.get_kernel_size(2), 
                             activation=self.model_settings.get_activation(2), 
                             data_format=self.model_settings.get_data_format(4), 
                             recurrent_dropout=self.model_settings.get_recurrent_dropout(2), 
                             return_sequences=self.model_settings.get_return_sequences(2)))
        
        model.add(MaxPooling3D(pool_size=self.model_settings.get_pool_size(2), 
                               padding=self.model_settings.get_padding(2), 
                               data_format=self.model_settings.get_data_format(5)))
        
        model.add(TimeDistributed(Dropout(self.model_settings.get_dropout(2))))
        
        model.add(ConvLSTM2D(filters=self.model_settings.get_filter_count(3), 
                             kernel_size=self.model_settings.get_kernel_size(3), 
                             activation=self.model_settings.get_activation(3), 
                             data_format=self.model_settings.get_data_format(6), 
                             recurrent_dropout=self.model_settings.get_recurrent_dropout(3), 
                             return_sequences=self.model_settings.get_return_sequences(3)))
        
        model.add(MaxPooling3D(pool_size=self.model_settings.get_pool_size(3), 
                               padding=self.model_settings.get_padding(3), 
                               data_format=self.model_settings.get_data_format(7)))
        
        model.add(Flatten())
        
        model.add(Dense(units=self.model_settings.get_label_count(), 
                        activation=self.model_settings.get_activation(4)))

        model.summary()
        plot_model(model, show_shapes=True, show_layer_names=True)
        
        return model
    