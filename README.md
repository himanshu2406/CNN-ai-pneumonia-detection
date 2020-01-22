# CNN-ai-pneumonia-detection

Detects whether a person has pneumonia or not based on a chest X-ray ; Has an accuracy of 98+ .

Trained on a dataset of 2.29 Gb (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

Model Used - CNN (Convolutional neural network)
Accuracy - 98% + ! (steps =500, epochs = 10, val_steps = 125 )

trained model size = 9 Mb so you don't have to worry about downloading the entire 2.29Gb to test the model


# HOW TO PREDICT ON NEW IMAGES :

go to https://colab.research.google.com/
press upload and upload the .h5 file (pneumonia_pred_new.h5)
press upload and upload the x-ray image you want to run the prediction on.

Then paste the following code in the notebook cell:
(Remember to replace the your_image.jpeg with the real name of the image)
Now run the code and see the prediction with accuracy after scrolling down !

CODE:-
https://pastebin.com/srVfAYLD


#mail - himanshuclash3@gmail.com
