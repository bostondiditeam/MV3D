## How to generate tracklet_labels.xml file manually.
A sample is given under `if __name__ == '__main__':` in ./Tracklet_saver.py file. Please read the sample code and 
adjust the size, transition and rotation(normally not needed) for manually generalizing tracklet files. After 
modifying the variables, you can just run that Tracklet_saver.py file to get your manually labeled tracklet file. 

(The purpose for this step is for making sure whether Udacity's ground truth tracklet file for test dataset is wrong 
or not. If it's not accurate, how can we fix it.) 


## Some important points
Tracklet is an tracking problem.  
But in evaluation, I think we can skip since it's evaluated frame by frame from current [evaluation code provided by 
Udacity](./evaluate_tracklets.py). 

