import os
import cv2
import numpy as np
from keras.models import load_model

from structures import DatasetHandler
from model_train import settings
from model_test import classes

def main():
    dataset_handler = DatasetHandler(False,
                                     None,
                                     settings["dataset"]["sequence_length"],
                                     settings["dataset"]["original_width"],
                                     settings["dataset"]["original_height"],
                                     settings["dataset"]["resize_width"],
                                     settings["dataset"]["resize_height"],
                                     settings["dataset"]["color_channels"],
                                     None,
                                     None)
    model = load_model("model.h5")
    
    video_reader = cv2.VideoCapture(1)
    video_fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    video_fps = video_reader.get(cv2.CAP_PROP_FPS)
    video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter("uploaded_videos/rendered_video.mp4", video_fourcc, video_fps, (video_width, video_height))
    
    record = False
    valid = False
    result_text = ""
    
    goal_name = ""
    goal_count = 0
    current_performs = 0
    current_lives = 3
    is_game_over = True
    while video_reader.isOpened():
        success, frame = video_reader.read()
        if not success:
            continue
        
        if is_game_over:
            goal_name = ' '.join([word.title() for word in input("Please enter the name of the exercise to be performed ('quit' to quit): " ).split(' ')])
            if goal_name.lower() == "quit":
                break
            
            valid_name = False
            for key, value in classes.items():
                if goal_name == value:
                    valid_name = True
                    break
                
            if not valid_name:
                print("You have entered wrong exercise name! Please enter a valid exercise name.")
                continue
            
            try:
                goal_count = int((input("Please enter the amount you want to perform this exercise ('-1' to quit): ")))
                if goal_count == -1:
                    break
                
                current_performs = 0
                current_lives = 3
                record = False
                valid = False
                is_game_over = False
                
    
            except ValueError:
                print("You have entered wrong value! Please enter a valid number.")
                continue
        
        pressed_key = cv2.waitKey(1)
        if pressed_key == ord('r'):
            record = True
            result_text = "Saving video frames..."
            
        elif pressed_key == ord('s'):
            if record:
                record = False
                valid = True
            
        elif pressed_key == ord('q'):
            break
        
        if record:
            video_writer.write(frame)
        
        elif not record and valid:
            video_writer.release()
            frames = dataset_handler.read_video("uploaded_videos/rendered_video.mp4")
            frames = np.expand_dims(np.array(frames), axis=0)
            predictions = model.predict(frames)
            max_index = np.argmax(predictions, axis=-1)[0]
            accuracy = predictions[0][max_index] * 100

            predicted_exercise = classes[max_index]
            if predicted_exercise == goal_name:
                current_performs += 1

            else:
                current_lives -= 1
                
            result_text = f"Class: {predicted_exercise}, Accuracy: {accuracy:.2f}%"
            video_writer = cv2.VideoWriter("uploaded_videos/rendered_video.mp4", video_fourcc, video_fps, (video_width, video_height))
            valid = False
        
        frame = cv2.resize(frame, (dataset_handler.ORIGINAL_WIDTH, dataset_handler.ORIGINAL_HEIGHT))
        if record:
            cv2.putText(frame, "Recording...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
        else:
            cv2.putText(frame, "Recording stopped.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
        if result_text:
            cv2.putText(frame, result_text, (10, dataset_handler.ORIGINAL_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
           
        top_right_text = f"Goal: {goal_name}\nCount: {goal_count}\nPerforms: {current_performs}\nLives: {current_lives}"
        if current_lives == 0:
            top_right_text = "You have lost the game!\nPlease select new\nexercise to be performed."
            is_game_over = True
            
        elif current_performs == goal_count and current_lives != 0:
            top_right_text = "You have won the game!\nPlease select new\nexercise to be performed."
            is_game_over = True
            
        y0, dy = 30, 30
        for i, line in enumerate(top_right_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(frame, line, (dataset_handler.ORIGINAL_WIDTH - 450, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Game", frame)

        if is_game_over:
            cv2.waitKey(1000)
            
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()
    if os.path.exists("uploaded_videos/rendered_video.mp4"):
        os.remove("uploaded_videos/rendered_video.mp4")
        
    return 0


if __name__ == "__main__":
    main()
    