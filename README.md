# Unsupervised Learning Using Tumor Cell Images
Using ideas from a previous work by Doerch et. al (https://arxiv.org/abs/1505.05192) we train an unsupervised neural network on images of cellular data, taken from a recent Kaggle competition, in order to predict four types of human cells. We demonstrate the feasibility of such methods to real world problems where, in most cases, data is scarce or not available. We use a combination of neural networks and statistical clustering algorithms.
### Overview
In September 2019 the ‘Recursion Cellular image classification’ Kaggle challenge was finished and the winner was awarded a 20,000$ prize. The challenge was to predict 1108 siRNA knock-down treatment effects on human cells, given only the images of those cells. siRNAs are artificial small RNAs that are experimentally introduced in cells to alter some specific protein levels. The regulation of protein levels in cells is critical to cell biology and altered regulation is frequently identified in diseases such as cancer. siRAN can downregulate protein expression but rarely abrogate its expression. Efficient siRNA can down-regulate protein production by 80% but this may vary depending on cell type and physiological context, and some treatment siRNAs may not all have a distinguishable phenotypic on cell pictures.

The four cell types that were used in the experiments are:

- U2OS: Human bone cancer cells
- RPE: Retinal pigment epithelium cells
- HUVEC: Human umbilical vein cells
- HEPG2: Human liver cancer cells

The challenge data contains around 400,00 images, partitioned to train (where labels are given) and test. Images are marked by experiment, plate, well and site. Each site was photographed six times in grayscale image, each one for six different parts of the cell (nucleus, mitochondria etc.)

![](/pics/pic1.png)

An example of plate, well and six channel images

The winner of the challenge reached an accuracy above 0.99. He implemented a bag of methods in designing a deep learning model that included:

 - Data augmentation
 - Cutmix regioal dropout (https://arxiv.org/abs/1905.04899)
 - EfficientNet (https://arxiv.org/abs/1905.11946)
 - Pseudo labeling (https://arxiv.org/abs/1908.02983)
 - ArcFace loss (https://arxiv.org/abs/1801.07698)
 - Cosine learning rate decay (https://arxiv.org/abs/1806.01593v2)
 - Nvidia apex.amp 16 bits float training (https://github.com/NVIDIA/apex)

![](/pics/pic2.png)
These eight images (taken from rxrx.ai) demonstrate the batch effect on two different cell types (in rows)

### Unsupervised Learning
In many real world machine learning problems, availability of data is the real bottleneck prohibiting the model from getting high accuracy. Even in cases where data is available, providing accurate labels for it is another challenging problem. Visual tasks, where deep learning excels, require huge amounts of data that can reach millions of images. These images are mandatory for the deep network to properly converge and learn the proper parameters weights. In state of the art networks, the number of parameters to learn reaches billions. This makes the problem of labeled data availability even harder.
In the biological domain, medical institutes such as hospitals etc. store huge amounts of unlabeled image data that was used, when taken, by a doctor for immediate ad-hoc analysis. These images contain unlabeled information on treatments, patient conditions etc. They have high potential to be used for many unsupervised tasks.

Doerch et. al. suggested a way to train a deep network on a large corpus of images, even if labeled data is unavailable. Their approach, ‘Self Supervised Learning’ can be summarized it by the following steps:

1. Find a related visual task that can use these unlabeled images.
2. This task should be hard enough for a deep network to train on.
3. This task should have labels, or you can prepare them automatically.
4. Train the network on the new task.
5. See if the network was able to extract semantic concepts from the original images, that can assist in solving the original, unlabeled, problem.

Doerch et. al. demonstrated their approach on the ImageNet natural images dataset. The ‘sub-task’ that they picked was to train a network for patch location. Each image was sliced to nine patches: center patch and 8 surrounding patches:

![](/pics/pic3.png)

The network was then trained to answer the following question: Given a center patch, and a random surrounding patch, find the correct relative location of the surrounding patch (that is one out of eight possibilities).

After a few weeks of training, the network was able to properly identify the correct position of the patches. The authors showed that their network can be used as a pretrained network for a supervised learning network, lowering its training time by magnitude and achieving better results.

In this work we try to implement the same approach of a self supervised network, using the Kaggle Recursion data. In our work we raised the following question:

- Can we use the unlabeled dataset of cell images to predict one out of four possible cell types?

Thus, we ignored the siRNA treatment altogether and instead we tried to predict a more modest class: the cell type of cells in the image, but doing it with no labels at all.

Our work adds the following novel ideas:

- Applying the method to biological data.
- Lowering training time to a few days by building a parameter effective model. This is also needed in order to cope with relatively small datasets.
- Add a clustering algorithm to bridge the gap from an unsupervised to a supervised problem, thus being able to use the model for prediction.

### Data Preparation

We used the original dataset from the Recursion competition. We concatenated the six image channels to one RGB image of resolution 512x512x3. We resized the resulting image to 400x400 pixels for efficiency.

Next we performed class balancing due to high representation of the HUVEC class. We selected 10,000 images from each cell type, and sliced it to ‘center’ and ‘random’ patches. Since in cellular images it is much harder to interpolate diagonal relationships between patches, we decided to use only the four patches around the center square, i.e. our class labels were ‘up’, ‘down’, ‘left’ and ‘right’. The shape of each patch image is 133x133x3.

After taking all possible options for each image we got 40,000 pair samples for each cell type and 160,000 pairs altogether.

![](/pics/pic4.png)

After initial tests we realized that our model specializes on edges detection, e.g. boundary lines between patches, identifying line continuation or color match. To overcome that and force the model to learn higher level features that identify different cell structures, e.g. size of nucleus etc, we cropped 15 pixels from the edges of each patch, arriving at the final dimensions of each patch: 100x100x3.

Labels (up, down, right, left) were one hot encoded and attached to the training data, which was assembled as a TFRecord (https://www.tensorflow.org/tutorials/load_data/tfrecord
TFRecord format allows skipping Numpy array translation, and instead is quickly loaded to the GPU memory. This allows a factor of speedup during training). Finally the image data was split to train and test data with a ratio of 9/1.

### Model Design
We used the same “Siamese” network approach from the original paper but with different layer components. In this method two streams of Convnets are simultaneously fed with the two patches, ‘center’ and ‘random’. The Convnets are forced to share their weights during training. This ensures that they focus only on shared characteristics of the images, not on features that assist in patch orientation. The patch location features are learned on later layers steps, only after the two Siamese streams are joined. In the top layers the two streams are joined and followed by dense layers.

![](/pics/pic5.png)


Patch A		Patch B

Following are the characteristics of our network:

- Batch size: 16
- Optimizer: Stochastic gradient descent with Nesterov acceleration (Y. Nesterov Introductory lecture on convex programming, 1998).
- Categorical cross entropy loss.
- Drop-outs to avoid overfitting.

Network architecture:

| Layer (type)            | Output Shape | Param # | Comment                                            |
|-------------------------|--------------|---------|----------------------------------------------------|
| Input_1                 | 133, 133, 3  | 0       |                                                    |
| Input_2                 | 133, 133, 3  | 0       |                                                    |
| Batch_normalization*    | 133, 133, 3  | 12      |                                                    |
| Conv2d + Relu           | 66, 66, 64   | 1792    | filters=64, kernel_size=(3, 3) strides=(2, 2) padding='valid' |
| Dropout                 | 66, 66, 64   | 0       |                                                    |
| Batch_normalization     | 66, 66, 64   | 256     |                                                    |
| Conv2d + Relu           | 32, 32, 16   | 9232    | filters=16, kernel_size=(3, 3), strides=(2, 2), padding='valid' |
| Dropout                 | 32, 32, 16   | 0       |                                                    |
| Batch_normalization     | 32, 32, 16   | 64      |                                                    |
| Conv2d + Relu           | 15, 15, 4    | 580     | filters=4, kernel_size=(3, 3), strides=(2, 2), padding='valid' |
| Batch_normalization     | 15, 15, 4    | 16      |                                                    |
| Add                     | 15, 15, 4    | 0       |                                                    |
| Max_pooling2d           | 5, 5, 4      | 0       |                                                    |
| Flatten                 | 100          | 0       |                                                    |
| Dense                   | 16           | 1616    |                                                    |
| Batch_normalization     | 16           | 64      |                                                    |
| Dense                   | 16           | 272     |                                                    |
| Batch_normalization     | 16           | 64      |                                                    |
| Dense                   | 4            | 68      |                                                    |


*Shared weights

### Model Training
We used a Linux server with one GeForce GTX Titan GPU, 16 GB of RAM. Training accuracy after two days (400 epochs) reached 0.65. Compared to a random classifier (accuracy 0.25) the network was able to specialize on patch location.

Analyzing the first few convolutional layers shows that the model indeed is learning geometrical characteristics of the images

![](/pics/pic6.png)

Feature vectors of the 16 filters in the first layer

We then looked at feature maps of different layers when predicting results of different cell types. For example for the following U2OS cells:

![](/pics/pic7.png)

We get as output from the first convolution layer:

![](/pics/pic8.png)



### Clustering Internal Layers

The next step is to remove the top layers of the model and check if internal layers are able to learn cell type properties. This is in order to use the model as a classifier for cell types. The following clustering approaches did not work, or are not scalable enough to handle the vector lengths of internal layers:

- PCA
- tSNE
- UMAP
- DBScan
- Spectral clustering
- Auto-encoders for dimension reduction

The clustering algorithm that was able to provide significant results is FastICA (see appendix). Here are the parameters used, together with the visualization of the embedding results. We can see that cell images of the same cell types are co-located in the embedded space:

![](/pics/pic9.png)


FastICA(2, max_iter=3000, tol=0.00001, fun='cube')

This method can now be used to classify new points:

1. Use only the first part of the ‘patch discovery’ mode.
2. Concatenate the FastICA clustering to it.
3. Identifying cell-type regions of the FastICA embedding results (i.e. which cell type belongs to which region) by using few (ten’th) labeled samples from each cell type. Run a prediction of a known (labeled) image and mark the embedded location.
4. A new point can now be classified by measuring its nearest labeled neighbors, taking their most common consensus.

We measured the ‘consensus’ factor of the FastICA embedding results by calculating the following:

- For each point A check the label of the point most close to it (call it B).
- Compare the label of point A to the label of point B.
- Sum the number of ‘agreements’ between points.

When running this calculation we received a result of 600 agreements out of 1000 points (some points next to the boundary lines ought to get a wrong consensus score).

# Unsupervised Learning Using AutoEncoders
Another approach we investigated was the implementation of "Unsupervised Deep Embedding for Clustering Analysis" (https://arxiv.org/pdf/1511.06335.pdf), a highly original unsupervised method.

In this approach, the network integrates both an encoder and decoder, along with a k-means layer. To enhance performance, we incorporated a ResNet backbone into our implementation. The coding techniques and training process are discussed in our notebook:

[Autoencoder notebook](/code/wbc.ipynb)