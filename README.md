### 1. Project Overview
- Project Name: Detection of elbows from a image of a baseball player
- Why: Detecting information of elbows(skelton) from a image of a baseball player is the 1st step to analyze the movements of a baleball player.
- What: This project is to detect the locations of both elbows.


### 2. Architecture
  - Basic structure: AutoEncoder\
      Input: (512,512,3)
     Output: (512,512,3)
  - Model Summary: https://github.com/Yuito-sug/Automatic-Detection-of-locations-of-elbows/blob/main/code/model/model_structure.pdf

  - Details
    1. Input
       RGB images whose shape is (512,512).
    2. Output
       RGB images that the predicted locations of elbows are marked as white and other places are black.


### 3. Usage
You want to look at the code?
-> `code` folder
You want to look at the images?
-> `images` folder
   
