# neuralnetwork
A ground-up implementation of neural networks for research

* Build a neural network architecture with original visualizations for training. 
* Being developed with this book https://www.overleaf.com/restricted?from=%2Fproject%2F5cbf900d4995674291e3da8d



Neural network

Todo:
- [ ] K means clustering
- [ ] Normalize all datasets
- [x] Gradient checking
- [x] Momentum
- [x] RMSProp
- [x] Adam
- [ ] Normalizing inputs/Data Normalization
- [ ] Learning rate decay
- [ ] Xavier initialization
- [x] L2 regularization
- [ ] Batch normalization
- [ ] Dropout



Dataset experiment:
	- [x] Create more random one dimensional datasets 
	- [ ] Create an algorithm to extract centers and radii

Research:
- [ ] Given any one-dimensional dataset, extract centers and radii, automatically create NN that can fit the dataset
	- [ ] Step 1: sort the input X and Y from smallest to largest
	- [ ] Look at Y with the largest cluster of consecutive 1s, using these points infer a center and radius and number of points in cluster. Continue until you get all positive clusters. The Clusters list contains clusters.keys = [1,2,3,4,5,6], and clusters[i].keys() = [C, R, points]
	- [ ] Build NN and initialize to Cs and Rs
	- [ ] Freeze all trainable parameters except for C1, R1 and create a masked set of labels Y that contain 1s int he radius of C1 but the rest zeros. Train it until 100% accuracy, then continue for C2, R2.

Neural Network automatic initialization:
- [x] Automatically initialize
- [ ] Find out other ways to initialize where centers and radii are more random and the initailization minimizes the lossmore random and the initailization minimizes the loss

Neural Network Improvements:
	- [ ] Program Adam optimizer

Visualization improvements:
	- [ ] Create two dimensional visualizations

- [x] Create accuracy function
- [x] Program trainability of parameters
- [x] Implement training and validation set during training and print loss/acc/val_loss/val_acc 
- [x] Program mini batch gradient descent

- [ ] Once ci and ri clusters are extracted, start with largest clusters then train, check accuracy,val_accuracy, keep adding layers until the val_ accuracy start decreasing (a sort of early stopping)

