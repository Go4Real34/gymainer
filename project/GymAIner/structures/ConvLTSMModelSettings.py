class ConvLTSMModelSettings:
    def __init__(self, 
                 dataset_path, 
                 dataset_sequence_length, 
                 dataset_resize_width, 
                 dataset_resize_height, 
                 dataset_color_channels, 
                 dataset_validation_ratio, 
                 dataset_test_ratio, 
                 
                 model_filter_counts, 
                 model_kernel_sizes, 
                 model_activations, 
                 model_data_formats, 
                 model_recurrent_dropouts, 
                 model_return_sequences, 
                 model_pool_sizes, 
                 model_paddings, 
                 model_dropouts, 
                 
                 early_stopping_callback_count, 
                 early_stopping_callback_monitors, 
                 early_stopping_callback_patiences, 
                 early_stopping_callback_modes, 
                 early_stopping_callback_restore_best_weights, 
                 
                 compiling_loss, 
                 compiling_learning_rate, 
                 compiling_metrics, 
                 
                 training_epochs, 
                 training_batch_size, 
                 training_shuffle, 
                 
                 statistics_graph_size):
        
        self.dataset_path = dataset_path
        self.dataset_sequence_length = dataset_sequence_length
        self.dataset_resize_width = dataset_resize_width
        self.dataset_resize_height = dataset_resize_height
        self.dataset_color_channels = dataset_color_channels
        self.dataset_validation_ratio = dataset_validation_ratio
        self.dataset_test_ratio = dataset_test_ratio
        
        self.model_filter_counts = model_filter_counts
        self.model_kernel_sizes = model_kernel_sizes
        self.model_activations = model_activations
        self.model_data_formats = model_data_formats
        self.model_recurrent_dropouts = model_recurrent_dropouts
        self.model_return_sequences = model_return_sequences
        self.model_pool_sizes = model_pool_sizes
        self.model_paddings = model_paddings
        self.model_dropouts = model_dropouts
        
        self.early_stopping_callback_count = early_stopping_callback_count
        self.early_stopping_callback_monitors = early_stopping_callback_monitors
        self.early_stopping_callback_patiences = early_stopping_callback_patiences
        self.early_stopping_callback_modes = early_stopping_callback_modes
        self.early_stopping_callback_restore_best_weights = early_stopping_callback_restore_best_weights
        
        self.compiling_loss = compiling_loss
        self.compiling_learning_rate = compiling_learning_rate
        self.compiling_metrics = compiling_metrics

        self.training_epochs = training_epochs
        self.training_batch_size = training_batch_size
        self.training_shuffle = training_shuffle
        
        self.statistics_graph_size = statistics_graph_size
        
        return
    

    def get_dataset_path(self):
        return self.dataset_path
    
    def get_dataset_sequence_length(self):
        return self.dataset_sequence_length
    
    def get_dataset_resize_width(self):
        return self.dataset_resize_width
    
    def get_dataset_resize_height(self):
        return self.dataset_resize_height
    
    def get_dataset_color_channels(self):
        return self.dataset_color_channels
    
    def get_dataset_validation_ratio(self):
        return self.dataset_validation_ratio
    
    def get_dataset_test_ratio(self):
        return self.dataset_test_ratio
    

    def get_model_filter_count(self, layer):
        return self.model_filter_counts[layer]

    def get_model_kernel_size(self, layer):
        return self.model_kernel_sizes[layer]
    
    def get_model_activation(self, layer):
        return self.model_activations[layer]
    
    def get_model_data_format(self, layer):
        return self.model_data_formats[layer]

    def get_model_recurrent_dropout(self, layer):
        return self.model_recurrent_dropouts[layer]
    
    def get_model_return_sequence(self, layer):
        return self.model_return_sequences[layer]
    
    def get_model_pool_size(self, layer):
        return self.model_pool_sizes[layer]
    
    def get_model_padding(self, layer):
        return self.model_paddings[layer]
    
    def get_model_dropout(self, layer):
        return self.model_dropouts[layer]
    

    def get_early_stopping_callback_count(self):
        return self.early_stopping_callback_count
    
    def get_early_stopping_callback_monitors(self, index):
        return self.early_stopping_callback_monitors[index]
    
    def get_early_stopping_callback_patiences(self, index):
        return self.early_stopping_callback_patiences[index]

    def get_early_stopping_callback_modes(self, index):
        return self.early_stopping_callback_modes[index]
    
    def get_early_stopping_callback_restore_best_weights(self, index):
        return self.early_stopping_callback_restore_best_weights[index]
    

    def get_compiling_loss(self):
        return self.compiling_loss
    
    def get_compiling_learning_rate(self):
        return self.compiling_learning_rate
    
    def get_compiling_metrics(self):
        return self.compiling_metrics
    

    def get_training_epochs(self):
        return self.training_epochs
    
    def get_training_batch_size(self):
        return self.training_batch_size
    
    def get_training_shuffle(self):
        return self.training_shuffle

    def get_statistics_graph_size(self):
        return self.statistics_graph_size
    