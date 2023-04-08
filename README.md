# Extracting-Fruit-Properties-to-Determine-Maturity


<h2> The Goal </h2>
The goal of the project is to extract properties of fruits(color and size) to use as criteria to determine fruit grade and maturity. The stereo camera used is (Camera Binoculars 3D-1mP02). 

<h2> Methods Used </h2>
A Machine learning model was trained based on the <super>YOLOv5 detection</super> and classification algorithm, the model was trained on a dataset of multiple fruits obtained from the Kaggle Website. The YOLO algorithms outputs a bounding box as well as a classification for the fruit. After a bounding box is obtained, next is isolating the object within the bounding box, that was done using the <super>Grabcut Alogorithm</super>. After the object is completely isolated from the raw frame we can go ahead and work on extracting its Size(dimension) and Color. <super>The K-means Clustering algorithm</super> (which is an unsurpevised machine learning algorithm) was used to determine the K Dominant colors of the isolated fruit. For extracting the dimensions, the <super>disparity</super> method in the stereovision system was used. A Relationship between the disparity and a coefficient <super>alpha</super> is extracted, <super>alpha</super> multiplied by the length in pixels from the frame outputs the actual length in cm. The final step is to produce a threshold of the image in addition to morphological operations to remove any small noise present in the image and that is to ouput a crisp threshold of the isolated fruit for further procesing.

<h2> The Result </h2>
The Results folder contains samples of the results. The project is not done yet, this repository contains the current progress. A full Report is currently being prepared to explain in-depth each method and algorithm utilized in each step of the project, As well as a research paper.

<h2> Acknowledgements </h2>
This project is a graduation project for my bachelor's degree in Mechatronics at Tishreen University.
Special acknowledgement to the supervisor of the project, Dr. Nael Daoud


