### Object Detection Using Inceptionv4

This contains a python script for Object detection using pre-trained model Inception v4

#### Requirements :
1.  Python 3.5.x
2.  Tensorflow latest version

#### Steps followed :
1.  Git clone [pyAudioAnalysis repository](https://github.com/tyiannak/pyAudioAnalysis.git)
2.  Import required libraries
3.  Read input audio
4.  Some modifications are made in [audioSegmentation](script/audioSegmentation.py) python file as compared to the originalpython file from     the repository.
6.  Function to detect speaker id and speech and write data to CSV file
7.  Sppech recognition is done using [wit.ai](https://wit.ai/)

##### [Full python script for object detection](script/audioSegmentation.py)
