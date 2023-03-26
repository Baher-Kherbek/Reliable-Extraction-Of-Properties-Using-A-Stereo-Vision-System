# Extracting-Fruit-Properties-to-Determine-Maturity
This Project is the Graduation Project for my Bachelor's in Mechatronics

The aim of the project is to extract the two basic criterias, Color and Size, to determine fruit maturity. The stereo camera used is (Camera Binoculars 3D-1mP02). The steps of the project were briefly as follows:

A Machine learning model was trained based on the YOLOv5 detection and classification algorithm, the model was trained on a dataset of multiple of fruits. After a bounding box of the object was extracted the next step is to isolate the object within the bounding box and that was done using the Grabcut Alogorithm. After the object was completely isolated from the raw frame we can go ahead and work on extracting its dimensions and color. K-means Clustering algorithm was used to determine the dominant color of the isolated fruit. For extracting the dimensions, the disparity method in the stereovision system was used. The final step is to produce a threshold of the image in addition to a morphological operation to remove any small noise present in the image.

A full Report on the project is currently being prepared to explain in-depth each method and algorithm utilized in each step of the project.
