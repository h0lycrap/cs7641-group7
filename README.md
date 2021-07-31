# Learning Frameworks for Various Airfoils

## I. Introduction
Although Computational Fluid Dynamics (CFD) surpasses experimental techniques in terms of time, cost and overall simplicity (this is attributed to the various complexities associated with the experimental setup), it is a nontrivial technique which necessitates significant amounts of computational cost to obtain accurate results (i.e. results comparable to those obtained using experimentation). Such factors prove to be sufficiently problematic during iterative design procedures. Dimensionality is one such issue associated with CFD. Though dimensionality may be addressed through altering coordinates from Cartesian to PARSEC parameters [1], the resulting shape may not be optimal. In particular, obtaining an optimal airfoil shape for given flight conditions is an arduous task. Such an optimization problem incorporates many factors, often times involving latent variables in the process, and may even lead itself to a multidisciplinary design optimization (MDO) problem. To this end, Machine Learning presents itself as the hero in disguise; we may leverage learning to curtail the time necessary to ascertain the aforementioned optimal airfoil properties. [2][3][4]

## II. Problem Statement 
The problem at hand is presented as follows; given a dataset with airfoil parameters (features) and corresponding aerodynamic performances (labels), clustering will be conducted, and neural networks will be trained to predict sufficiently accurate aerodynamic properties of interest. The objective at hand is to determine an ideal clustering method as well as an optimal number of clusters for the cluster analysis. Additionally, we seek to determine an architecture, along with associated weights, which properly models the given network.

## III. Data Collection and Pre-Processing
The airfoil shapes were collected from the UIUC Airfoil Database, which contains the coordinates for approximately 1600 airfoils. The data for each airfoil is stored in a text file containing the name of the airfoil and the discrete x and y shape coordinates. The airfoil data has then been run through the XFOIL software [5], a fast flow solver that takes as an input the discrete airfoil shape coordinates, to obtain aerodynamic properties such as lift and drag coefficients, as well as other quantities of interest. These aerodynamic properties will act as the features for clustering the airfoils. 

Out of the 1600 initial airfoils in the dataset, only 600 were able to make XFoil converge. Indeed, XFoil is very sensitive to any shape discontinuity and the order of the shape coordinates in the shape file. The number of shape coordinates, and the order of the shape coordinates in each element of the dataset is inconsistent. Some of the airfoils in the dataset do not contain enoough coordinate points to make XFoil converge and compute the aerodynamic coefficients. In other words, the data cleaning was performed automatically by XFoil, keeping only the airfoils with enough information to extract relevant data.

As mentioned previously, the number of shape coordinates for each airfoil file is not consistent. In average, each file contains about 20 discrete points which were not extracted at the same x coordinate. Moreover, the discrete shape parameterization can lead to poor results when applied to a machine learning framework or an optimization process, as any small change in the discrete shape coordinates can lead to a degenerated shape that does not correspond to a feasible airfoil shape. The goal of the data preprocessing for this project is to reduce the dimensionality of the dataset, make it homogeneous, readable and practical for any machine learning framework. There exist multiple methods to perform this, such as changing the shape parameterization technique, using Bezier curves or the NACA 4-digit method. One of the methods that has proven to be effective in terms of representing airfoil shapes is the PARSEC method that only contains 11 shape parameters, reducing the dimensionality in half. Additionaly, the trailing edge thickness and location is assumed to be 0, which further reduces dimensions to 9. An illustration of the PARSEC parameterization is shown in the Figure 1[1].

<figure>
<center><img src="Images/parsec_parameters.PNG"></center>
<center><figcaption>Figure 1.Visualization of PARSEC parameters</figcaption></center>
</figure>

The final data pre-processing is to convert the airfoil dataset using discrete shape parameterization into a dataset using a PARSEC parameterization. To do so, an optimization algorithm is applied to each airfoil in the dataset in order to find the most fitting PARSEC parameters resulting in the closest airfoil shape to the original. The final dataset is then a list of 600 airfoils having as features the corresponding 9 PARSEC parameters and their lift coefficient, drag coefficient and voume as labels. Parameterization seems to work very well. Even for the worst parameterization, the parameterized airfoil quite resembles the actual airfoil. 

<figure>
<center><img src="Images/airfoil_parameterization.png"></center>
<center><figcaption>Figure 2. Airfoil parameterization results</figcaption></center>
</figure>

## IV. Methods
Both supervised and unsupervised learning will be used to classify airfoils based on their properties and to predict the airfoil class based on the PARSEC parameters. Firstly, clustering techniques were used to identify airfoil classes based on properties such as lift, drag and volume. Several clustering techniques were used, including K-Means and Gaussian Mixture Models (GMMs). Next, an artificial neural network will be used to implement a technique such as logistic regression to classify airfoils into classes. The PARSEC parameters will act as the features.

### Airfoil Clustering

Several clustering algorithms were implemented. The data consists of three features: the lift coefficient (Cl), the drag coefficient (Cd) and the volume. A visualization of the data is shown below. 

<figure>
<center><img src="Images/airfoil_data.png"></center>
<center><figcaption>Figure 3. Visualization of airfoil data</figcaption></center>
</figure>

There are several things that can be noted about the training data: 
1. Most of the data is not clearly separated in easily identifiable clusters. Therefore, cluster assignments cannot be made through visual inspection.
2. There is one identifiable pattern in the data: several airfoils have a lift coefficient (Cl) of zero. These airfoils are symmetric. The data was captured at an angle of attack of zero, and in this case, the Cl for symmetric airfoils is also zero.
3. We do not have access to any ground truth data that clearly associates each airfoil with a particular cluster or class. Typically, different clustering algorithms would be compared based on their performance with respect to some ground truth, but since it is not available, it is not easy to compare clustering techniques directly. Therefore, we have separately evaluated each technique used and contrasted the results, rather than evaluating their performance relative to each other.

First, GMM models were trained on the data with a varying number of clusters/components. A visualization of the result using six clusters is shown below. 

<figure>
<center><img src="Images/gmm_six_clusters.png"></center>
<center><figcaption>Figure 4. GMM with six clusters</figcaption></center>
</figure>

We can see that the airfoils are clustered mainly based on their Cl values. The GMM also identifies most of the symmetric airfoils and places them within a separate cluster.

The GMM models were evaluated using two metrics:
1. The silhouette coefficient
2. The Davies-Bouldin index

The results for a number of clusters varying from 2 to 12 can be seen below.

<figure>
<center><img src="Images/gmm_sil_coeff.png"></center>
<center><figcaption>Figure 5. Silhouette coefficient for GMM models</figcaption></center>
</figure>

Ideally, the silhouette coefficient should be as close to 1 as possible. The best value is obtained with only 2 clusters, after which there is a steep drop. The next best value is obtained at 7 clusters.

<figure>
<center><img src="Images/gmm_dbindex.png"></center>
<center><figcaption>Figure 6. Davies-Bouldin Index for GMM models</figcaption></center>
</figure>

The Davies-Bouldin index should ideally be as low as possible. Once again, the best value is obtained with just 2 clusters, with the next best value at 7 clusters.

The GMM models were trained several times. The same general trend was observed for the evaluation metrics, though there were slight differences in the optimal number of clusters. If the models are evaluated based on the metrics alone, it suggests that 2 clusters are ideal. However, if only 2 clusters are used, the model simply splits the airfoils evenly near the mean of the Cl values. On the other hand, if a greater number of clusters are used, the model may identify patterns within the data. For example, the similarity in the symmetric airfoils is identified and the GMM places them in a single cluster. Therefore, we selected 6-8 clusters as the optimal number based on the results from the GMMs.

K-Means was also used to cluster the airfoils.

We leverage K-Means clustering on the data using six clusters, in a similar fashion to that done using the GMM implementation. Below we provide results to facilitate visualizing the labelled data.
<figure>
<center><img src="Images/kmeansclustereddata.png"></center>
<center><figcaption>Figure 7. K-Means Implemented for 6 Clusters</figcaption></center>
</figure>

We visualize the labeled data trained using K-Means for 6 clusters/components. 
<figure>
<center><img src="Images/2DVisualizationCombined.png"></center>
<center><figcaption>Figure 8. 2D Plot to Visualize Labelled Data</figcaption></center>
</figure>

From the figures above, it is evident that the airfoils are indeed clustered based on lift coefficient values. Additionally, the K-Means algorithm is able to identify airfoils with lift coefficients of zero, (these are our symmetric airfoils, which have a theoretical lift coefficient value of zero) and categorize them into a single group. This is indeed comparable to the approach taken by the GMM implementation as well. 

Following this, we analyze the data using an Expectation Maximization framework. This entails randomly selecting cluters, assigning labels based on the nearest clusters (accomplished using pairwise distance arithmetic), subsequently determining new centers and finally ensuring that there is convergence. This approach is also able to categorize airfoils with zero lift coefficients into a single group. 

<figure>
<center><img src="Images/EMResults.png"></center>
<center><figcaption>Figure 9. Clustering Results obtained using Expectation Maximization</figcaption></center>
</figure>

Finally, we analyze the data using Spectral Clustering. In essence, this technique leverages nearest-neighbor graphs to cluster unorganized data into groups based on common features. Below we provide a graph of the results obtained using spectral clustering. 

<figure>
<center><img src="Images/spectralclustering.png"></center>
<center><figcaption>Figure 10. Clustering Results obtained using Spectral Clustering</figcaption></center>
</figure>

Finally, we provide the folloiwng 3D plot to visualize the labelled data. 

<figure>
<center><img src="Images/3Ddata.png"></center>
<center><figcaption>Figure 11. 3D Plot to Visualize Labelled Data </figcaption></center>
</figure>

After evaluating the results from the clustering, we decided to use the cluster assignments determined by K-Means. Ideally, we would like to classify a test airfoil into a certain range of Cl/Cd/volume. The K-Means clustering results in the simplest division of the training data, allowing us to easily map each cluster top a range of Cl values. On the other hand, the clustering from GMM, for instance, is much more complex and it is not easy to readily map each cluster to a range of values.

### Airfoil Classification Using Neural Network

Based on the results from the clustering, all of the training airfoils will be labelled with the appropriate cluster assignment from KMM. These act as the labels for classification, while the PARSEC parameters act as the features. The output of the network is vector with 6 elements which is one-hot encoding of different clusters. Following tables summerizes two best models.

(Placeholder for Gabriel's model)

|Type of layer | Number of Neurons | Activation function |
|--------------|-------------------|---------------------|
|Dense         |9                  | Tanh                |
|Dense         |12                 | Tanh                |
|Dense         |18                 | Tanh                |
|Dense         |24                 | Tanh                |
|Dense         |24                 | Tanh                |
|Dense         |18                 | Tanh                |
|Dense (output)|6                  | Softmax             |


(Paragraph about the Gabriel's model)

Training on 1000 epochs, the accuracy using the second model is high as 90%. However, using 10-fold cross validation, it is shown that the model is overfitting. Therefore, L2 regularization with regularization constant of 1.5e-3 is done to all the hidden layers. After regularization, the accuracy fell to approximately 70% but overfitting does not occur.

<figure>
<center><img src="Images/model_billy.png" height="60%"></center>
<center><figcaption>Figure 13. Accuracy versus epoch for the second model </figcaption></center>
</figure>


## V. Results and Discussion
This project will enable the creation of a low-runtime model that maps an airfoil shape to its aerodynamic performance. Ideally, the predictions made should be comparable to those made by a software such as XFOIL. In turn, this model could be used within a larger project relating to aerodynamic shape optimization. 

## VI. Conclusions

## Sources
1. Sobieczky, H.: Parametric Airfoils and Wings, Notes on Numerical Fluid Mechanics, edited by K. Fujii and G.S. Dulikravich, Vol. 68, Vieweg Verlag, 1998, pp. 71-88
2. Chen W., Chiu K., Fuge M., Aerodynamic Design Optimization and Shape Exploration using Generative Adversarial Network, University of Maryland, College Park, Maryland, 20742
3. Di Angelo L., Di Stefano P., "An evolutionary geometric primitive for automatic design synthesis of functional shapes: The case of airfoils"
4. Zhang Y., Sung W., Mavris D., Application of Convolution Neural Network to Predict Airfoil Lift Coefficient, Georgia Institute of Technology, Atlanta, Georgia, 30332
5. Drela, M.: XFOIL: Subsonic Airfoil Development System, Massachusetts Institute of Technology. https://web.mit.edu/drela/Public/web/xfoil/ 






