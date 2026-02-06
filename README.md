# Speech-Recognition-RAVDESS
Submission for the second round of recruitments of AI Club, BITS Pilani. Uses the RAVDESS database to create a model that classifies audio files based on emotions. Made using tensorflow, keras and librosa.

ID Number: 2025AAPS0776P
Name: Tanish Jain

Prerequisite Requirements:
Your device must have python installed with the following libraries:
- sklearn
- librosa
- numpy
- seaborn
- random
- keras
- tensorflow

To run the predict.py file:
- Download the file to your device
- Run the command: python predict.py 'audio_file_path'

Model Performance:
Test Accuracy is 0.57
Macro F1 Score is 0.5342
If we look at the classification report and the confusion matrix, we can see that the model is able to predict all emotions with good accuracy except for the surprised emotion on which it does not show a good result and fails. It is confusing the surprised emotion with happy.

Angry vs Sad Log Mel Spectograms:
The angry spectogram is more spread into mid and higher frequencies while the sad spectogram is more concentrated in low frequencies. The pattern is more uniform and smooth as well.

Gender Wise Evaluation:
The model works better on female voices as compared to male voices as the Male Macro F1 is only 0.5170 as compared to the Female Macro F1 of 0.5695, a 5.25% improvement for female speakers.

Feature Extraction:
The model uses Log-Mel Spectograms as its features.

Data Augmentation:
To augment the audio, one of the following is randomly applied:
- Noise Injection: Adding random noise to the audio signal
- Pitch Shifting: Shifting the pitch up or down
- Time Stretching: Changing speed of speech
- No augmentation
We have done this as our dataset is small and keeping a dataset with no augmentation can create issues as the dataset would overfit to the provided audios and would not perform well on an audio it hasn't heard before.

Model Architecture:
The project uses a CNN (Convolutional Neural Network) built using Keras and Tensorflow.
Conv2D (32 filters, 3×3) + BatchNorm + MaxPool + Dropout(0.2)
Conv2D (64 filters, 3×3) + BatchNorm + MaxPool + Dropout(0.3)
Conv2D (128 filters, 3×3) + BatchNorm + MaxPool + Dropout(0.4)
Flatten
Dense (128) + Dropout(0.4)
Output Layer: Dense (8 classes, softmax)

Training Details:
Trained on 50 Epochs of Batch Size 32. The optimizer used is the Adam optimizer. The loss function is categorical_crossentropy.
