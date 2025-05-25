# CSC480-proj_4


***CNN: Detecting a Spaceship in a Noisy Grayscale Image***
- Input: 200 by 200 single channel image with exactly one spaceship
- Output: predict five real valued parameters describing spaceship's location, orientation, size:
  - X, Y position of spaceship
  - Yaw-- orientation (heading) in radians
  - Width-- size across axis perpendicular to yaw
  - Height-- size along the direction of yaw


  ***Implementation***
  ```make_data()```
   - generates labeled training and test data
  ```train_model()```
   - trains model using MSE loss
  ```evaluate_model()```
   - reports AP@0.7 on 1000 test samples
  ```show_examples()```
   - visualizes sample images and ground truth tables
  ```gen_model()```
   - Convulutional Neural Network
