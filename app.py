import streamlit as st
import cv2
import numpy as np
from streamlit import runtime
import tempfile

from views import *


def main():
    st.title("Deepfake detector")

    # File uploader for video
    video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

    # Slider from 1 to 100
    sequence_length = st.slider("Select a value", 1, 100)

    if video_file is not None:
        # Save the uploaded file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        video_file_name = os.path.basename(video_path)
        video_file_name_only = os.path.splitext(video_file_name)[0]

        st.write(f"You uploaded a video: {video_path},{video_file_name},{video_file_name_only}")
        
        # Play the video
        st.video(video_file)

        # Load validation dataset
        path_to_videos = [video_path]

        video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)

        # Load model
        if(torch.cuda.is_available()):
            model = Model(2).cuda()  # Adjust the model instantiation according to your model structure
        else:
            model = Model(2).cpu()  # Adjust the model instantiation according to your model structure

        
        path_to_model = os.path.join("model_84_acc_10_frames_final_data.pt")
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()

        start_time = time.time()
        # Display preprocessing images
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()

        print(f"Number of frames: {len(frames)}")
        # Process each frame for preprocessing and face cropping
        padding = 40
        faces_found = 0
        for i in range(sequence_length):
            if i >= len(frames):
                break
            frame = frames[i]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save preprocessed image
            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join('uploaded_images', image_name)
            img_rgb = pImage.fromarray(rgb_frame, 'RGB')
            img_rgb.save(image_path)
            preprocessed_images.append(image_name)

            # Face detection and cropping
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 0:
                continue

            top, right, bottom, left = face_locations[0]
            frame_face = frame[top - padding:bottom + padding, left - padding:right + padding]

            # Convert cropped face image to RGB and save
            rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
            image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
            image_path = os.path.join('uploaded_images', image_name)
            img_face_rgb.save(image_path)
            faces_found += 1
            faces_cropped_images.append(image_name)

        print("<=== | Videos Splitting and Face Cropping Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))

        # No face detected
        if faces_found == 0:
            return st.write("no_faces found")

        # Perform prediction
        try:
            heatmap_images = []
            output = ""
            confidence = 0.0

            for i in range(len(path_to_videos)):
                print("<=== | Started Prediction | ===>")
                prediction = predict(model, video_dataset[i], './', video_file_name_only)
                confidence = round(prediction[1], 1)
                output = "REAL" if prediction[0] == 1 else "FAKE"
                
                st.write("Prediction:", prediction[0], "==", output, "Confidence:", confidence)
                st.write("<=== | Prediction Done | ===>")
                st.write("--- %s seconds ---" % (time.time() - start_time))

                # Uncomment if you want to create heat map images
                # for j in range(sequence_length):
                #     heatmap_images.append(plot_heat_map(j, model, video_dataset[i], './', video_file_name_only))

            
        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            st.write(f"Exception occurred during prediction: {e}")

if __name__ == "__main__":
    main()
