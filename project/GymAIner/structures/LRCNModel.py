import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import matplotlib.pyplot as plt
import datetime as dt
import tensorflow as tf
import time

from keras.models import Sequential
from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, Dense
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from .DatasetHandler import DatasetHandler
from .LRCNModelSettings import LRCNModelSettings
from .Timer import Timer

class LRCNModel:
    def __init__(self, settings):
        self.settings = settings
        self.timer = Timer()
        
        return


    def init(self):
        self.all_settings = LRCNModelSettings(dataset_path=self.settings["dataset"]["path"], 
                                              dataset_sequence_length=self.settings["dataset"]["sequence_length"], 
                                              dataset_resize_width=self.settings["dataset"]["resize_width"], 
                                              dataset_resize_height=self.settings["dataset"]["resize_height"], 
                                              dataset_color_channels=self.settings["dataset"]["color_channels"], 
                                              dataset_validation_ratio=self.settings["dataset"]["validation_ratio"], 
                                              dataset_test_ratio=self.settings["dataset"]["test_ratio"], 
                                              
                                              model_filter_counts=self.settings["model"]["filter_counts"], 
                                              model_kernel_sizes=self.settings["model"]["kernel_sizes"], 
                                              model_paddings=self.settings["model"]["paddings"], 
                                              model_activations=self.settings["model"]["activations"],
                                              model_pool_sizes=self.settings["model"]["pool_sizes"], 
                                              model_dropouts=self.settings["model"]["dropouts"], 
                                              model_lstm_unit_count=self.settings["model"]["lstm_unit_count"], 
                                              
                                              early_stopping_callback_count=self.settings["early_stopping"]["count"], 
                                              early_stopping_callback_monitors=self.settings["early_stopping"]["monitors"], 
                                              early_stopping_callback_patiences=self.settings["early_stopping"]["patiences"], 
                                              early_stopping_callback_modes=self.settings["early_stopping"]["modes"], 
                                              early_stopping_callback_restore_best_weights=self.settings["early_stopping"]["restore_best_weights"], 
                                              
                                              compiling_loss=self.settings["compiling"]["loss"], 
                                              compiling_learning_rate=self.all_settings["compiling"]["learning_rate"], 
                                              compiling_metrics=self.settings["compiling"]["metrics"], 
                                              
                                              training_epochs=self.settings["training"]["epochs"], 
                                              training_batch_size=self.settings["training"]["batch_size"], 
                                              training_shuffle=self.settings["training"]["shuffle"], 
                                              
                                              statistics_graph_size=self.settings["statistics"]["graph_size"])
        
        self.dataset_handler = DatasetHandler(self.all_settings.get_dataset_path(), 
                                              self.all_settings.get_dataset_sequence_length(), 
                                              self.all_settings.get_dataset_resize_width(), 
                                              self.all_settings.get_dataset_resize_height(), 
                                              self.all_settings.get_dataset_color_channels(), 
                                              self.all_settings.get_dataset_test_ratio())
        self.dataset_handler.init()
        
        self.model = self.create_model()
        self.training_callbacks = self.create_early_stopping_callbacks()
        self.model.compile(loss=self.all_settings.get_compiling_loss(), 
                           optimizer=Adam(learning_rate=self.all_settings.get_compiling_learning_rate()), 
                           metrics=self.all_settings.get_compiling_metrics())

        return
        

    def execute(self):
        self.init()
        self.train()
        self.test()
        self.show_statistics()
        self.save()
        
        return
    

    def train(self):
        gpus = tf.config.list_physical_devices("GPU")
        device = ""
        if len(gpus) != 0:
            print("\nSupported NVIDIA GPU found! Training model on GPU for better performance.\n")
            device = "/device:GPU:0"
            
        else:
            device = "/device:CPU:0"
            
        print("Please wait while the program trains the model...", end="\n\n")
        
        self.timer.start()
        with tf.device(device):
            self.training_history = self.model.fit(x=self.dataset_handler.X_train, 
                                                   y=self.dataset_handler.Y_train, 
                                                   epochs=self.all_settings.get_training_epochs(), 
                                                   batch_size=self.all_settings.get_training_batch_size(), 
                                                   shuffle=self.all_settings.get_training_shuffle(), 
                                                   validation_data=(self.dataset_handler.X_validation, self.dataset_handler.Y_validation), 
                                                   callbacks=self.training_callbacks)
        self.timer.stop()
        print("Model trained succesfully.")
        
        time_elapsed = self.timer.get_formatted_time()
        print(f"Model Trained in {time_elapsed}.", end="\n\n")

        return
    

    def test(self):
        print("Please wait while the program tests the model...", end="\n\n")
        
        self.timer.start()
        self.evaluation_history = self.model.evaluate(self.dataset_handler.X_test, self.dataset_handler.Y_test)
        self.timer.stop()
        print("Model tested successfully.")
        
        time_elapsed = self.timer.get_formatted_time()
        print(f"Model Tested in {time_elapsed}.", end="\n\n")
        
        return
    

    def show_statistics(self):
        accuracy_history = self.training_history.history["accuracy"]
        val_accuracy_history = self.training_history.history["val_accuracy"]
        accuracy_history_indexes = range(len(accuracy_history))
        
        loss_history = self.training_history.history["loss"]
        val_loss_history = self.training_history.history["val_loss"]
        loss_history_indexes = range(len(loss_history))
        
        plt.figure(figsize=self.all_settings.get_statistics_graph_size())
        plt.gcf().canvas.set_window_title("Accuracy Statistics Over Time")
        plt.title("Accuracy Statistics Over Time")
        plt.plot(accuracy_history_indexes, accuracy_history, color="blue", label="Accuracy")
        plt.plot(accuracy_history_indexes, val_accuracy_history, color="red", label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy Value")
        plt.legend()
        
        plt.figure(figsize=self.all_settings.get_statistics_graph_size())
        plt.gcf().canvas.set_window_title("Loss Statistics Over Time")
        plt.title("Loss Statistics Over Time")
        plt.plot(loss_history_indexes, loss_history, color="blue", label="Loss")
        plt.plot(loss_history_indexes, val_loss_history, color="red", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.legend()
        
        plt.show()
        
        return
    

    def save(self):
        loss, accuracy = self.evaluation_history
        print("Accuracy: {:.2f}, Loss: {:.2f}%".format((accuracy * 100), (loss * 100)))

        time_format = "%d_%m_%Y__%H_%M_%S"
        current_date_time = dt.datetime.now()
        date_string = dt.datetime.strftime(current_date_time, time_format)
        
        model_file_name = f"LRCNModel_{date_string}__Accuracy_{accuracy}__Loss_{loss}"
        model_save_folder_path = os.path.join("models", "saves", "LRCN")
        if not os.path.exists(model_save_folder_path):
            os.makedirs(model_save_folder_path)

        model_save_path = os.path.join(model_save_folder_path, (model_file_name + ".h5"))
        
        model_plot_folder = os.path.join("models", "plots", "LRCN")
        if not os.path.exists(model_plot_folder):
            os.makedirs(model_plot_folder)
            
        model_plot_path = os.path.join(model_plot_folder, (model_file_name + ".png"))
        
        print("Please wait while the program saves the model...", end="\n\n")
        
        self.timer.start()
        self.model.save(model_save_path)
        plot_model(self.model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)
        self.timer.stop()
        print("Model saved successfully.")
        
        time_elapsed = self.timer.get_formatted_time()
        print(f"Model Saved in {time_elapsed}.")
        
        exact_model_save_path = os.path.abspath(model_save_path)
        exact_model_plot_path = os.path.abspath(model_plot_path)
        print(f"Model File Saved to {exact_model_save_path}")
        print(f"Model Plot Saved to {exact_model_plot_path}", end="\n\n")
        
        return
    

    def create_model(self):
        model = Sequential()
        
        model.add(TimeDistributed(Conv2D(filters=self.all_settings.get_model_filter_count(0), 
                                         kernel_size=self.all_settings.get_model_kernel_size(0), 
                                         padding=self.all_settings.get_model_padding(0), 
                                         activation=self.all_settings.get_model_activation(0), 
                                         input_shape=(self.dataset_handler.SEQUENCE_LENGTH, 
                                                      self.dataset_handler.RESIZE_HEIGHT, 
                                                      self.dataset_handler.RESIZE_WIDTH, 
                                                      self.dataset_handler.COLOR_CHANNELS))))
        
        model.add(TimeDistributed(MaxPooling2D(pool_size=self.all_settings.get_model_pool_size(0))))
        
        model.add(TimeDistributed(Dropout(self.all_settings.get_model_dropout(0))))
        
        model.add(TimeDistributed(Conv2D(filters=self.all_settings.get_model_filter_count(1), 
                                         kernel_size=self.all_settings.get_model_kernel_size(1), 
                                         padding=self.all_settings.get_model_padding(1), 
                                         activation=self.all_settings.get_model_activation(1))))

        model.add(TimeDistributed(MaxPooling2D(pool_size=self.all_settings.get_model_pool_size(1))))
        
        model.add(TimeDistributed(Dropout(self.all_settings.get_model_dropout(1))))
        
        model.add(TimeDistributed(Conv2D(filters=self.all_settings.get_model_filter_count(2), 
                                         kernel_size=self.all_settings.get_model_kernel_size(2), 
                                         padding=self.all_settings.get_model_padding(2), 
                                         activation=self.all_settings.get_model_activation(2))))
        
        model.add(TimeDistributed(MaxPooling2D(pool_size=self.all_settings.get_model_pool_size(2))))
        
        model.add(TimeDistributed(Dropout(self.all_settings.get_model_dropout(2))))

        model.add(TimeDistributed(Conv2D(filters=self.all_settings.get_model_filter_count(3), 
                                         kernel_size=self.all_settings.get_model_kernel_size(3), 
                                         padding=self.all_settings.get_model_padding(3), 
                                         activation=self.all_settings.get_model_activation(3))))
        
        model.add(TimeDistributed(MaxPooling2D(pool_size=self.all_settings.get_model_pool_size(3))))
        
        model.add(TimeDistributed(Flatten()))
        
        model.add(LSTM(self.all_settings.get_model_lstm_unit_count()))
        
        model.add(Dense(units=len(self.dataset_handler.labeled_video_paths), 
                        activation=self.all_settings.get_model_activation(4)))
        
        model.build(input_shape=(None, 
                                 self.dataset_handler.SEQUENCE_LENGTH, 
                                 self.dataset_handler.RESIZE_HEIGHT, 
                                 self.dataset_handler.RESIZE_WIDTH, 
                                 self.dataset_handler.COLOR_CHANNELS))
        
        model.summary()
        
        return model
    
    def create_early_stopping_callbacks(self):
        callbacks = []
        for index in range(self.all_settings.get_early_stopping_callback_count()):
            early_stopping_callback = EarlyStopping(monitor=self.all_settings.get_early_stopping_callback_monitors(index), 
                                                    patience=self.all_settings.get_early_stopping_callback_patiences(index), 
                                                    mode=self.all_settings.get_early_stopping_callback_modes(index), 
                                                    restore_best_weights=self.all_settings.get_early_stopping_callback_restore_best_weights(index))
            callbacks.append(early_stopping_callback)
            
        return callbacks
    