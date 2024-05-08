import os
import cv2
import tqdm
import numpy as np

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .Timer import Timer

class DatasetHandler:
    def __init__(self, dataset_path, sequence_length, resize_width, resize_height, color_channels, test_ratio):
        self.DATASET_PATH = dataset_path
        self.SEQUENCE_LENGTH = sequence_length
        self.RESIZE_WIDTH = resize_width
        self.RESIZE_HEIGHT = resize_height
        self.COLOR_CHANNELS = color_channels
        self.TEST_RATIO = test_ratio
        
        self.timer = Timer()
        
        return
    

    def init(self):
        self.labeled_video_paths, self.labels, self.videos, self.included_video_paths = self.read_dataset()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.videos, self.labels, test_size=self.TEST_RATIO, shuffle=True)
        
        return
    
        
    def read_dataset(self):
        tagged_paths = {}
        categories = os.listdir(self.DATASET_PATH)
        for category in categories:
            if category not in tagged_paths:
                tagged_paths[category] = []
                
            folder_path = os.path.join(self.DATASET_PATH, category)
            video_list = os.listdir(folder_path)
            for video in video_list:
                video_path = os.path.join(folder_path, video)
                tagged_paths[category].append(video_path)
                
        labels = []
        video_frames = []
        included_video_paths = []

        print("Please wait while the program reads the dataset...", end="\n\n")
        
        self.timer.start()
        total_videos = sum([len(v) for k, v in tagged_paths.items()])
        with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]",
                       desc="Saving videos...", unit=" videos", total=total_videos, leave=False) as pbar_full:
            label_count = len(tagged_paths)
            
            with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]", 
                            desc="Reading folders...", unit=" folders", total=label_count, leave=False) as pbar_folders:
                for category_index, (category, video_path_list) in enumerate(tagged_paths.items()):
                    pbar_folders.set_description(f"Reading folder: \"{category}\"")
                    
                    video_count = len(video_path_list)
                    with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]", 
                                   desc="Reading videos...", unit= " videos", total=video_count, leave=False) as pbar_videos:
                        for video_path in video_path_list:
                            dataset_tag, main_folder, category, video_name = video_path.split('\\')
                            pbar_videos.set_description(f"Reading video: \"{video_name}\"")
                            
                            current_video_frames = self.read_video(video_path)
                            if len(current_video_frames) == self.SEQUENCE_LENGTH:
                                labels.append(category_index)
                                video_frames.append(current_video_frames)
                                included_video_paths.append(video_path)
                            
                            pbar_videos.update(1)
                            pbar_full.update(1)
                            
                    pbar_folders.update(1)
                    
        self.timer.stop()
        print("Dataset read successfully.")
        
        time_elapsed = self.timer.get_formatted_time()
        print(f"Dataset read in {time_elapsed}.", end="\n\n")
        
        label_encoder = LabelEncoder()
        integer_labels = label_encoder.fit_transform(labels)
        categorical_labels = to_categorical(integer_labels)
        
        numpy_array_video_frames = np.asarray(video_frames)
        
        return tagged_paths, categorical_labels, numpy_array_video_frames, included_video_paths


    def read_video(self, video_path):
        frames = []
        video_reader = cv2.VideoCapture(video_path)
        if not video_reader.isOpened():
            print("Error: Could not open video.")
            return
        
        frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(frame_count / self.SEQUENCE_LENGTH), 1)
        
        with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]", 
                       desc="Reading sequence frames...", unit=" frames", total=self.SEQUENCE_LENGTH, leave=False) as pbar_frames:
            for frame_index in range(self.SEQUENCE_LENGTH):
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, (frame_index * skip_frames_window))
                
                success, frame = video_reader.read()
            
                if frame is None:
                    break
            
                if not success:
                    print("Error: Could not read frame.")
                    break
                
                processed_frame = self.process_frame(frame)
                frames.append(processed_frame)
                pbar_frames.update(1)
                
        video_reader.release()
        
        return frames
    

    def process_frame(self, frame):
        resized_frame = cv2.resize(frame, (self.RESIZE_WIDTH, self.RESIZE_HEIGHT))
        normalized_frame = resized_frame / 255.0
        return normalized_frame
    