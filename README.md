# Learning Frameworks for Various Airfoils

## I. Introduction
Although Computational Fluid Dynamics (CFD) surpasses experimental techniques in terms of time, cost and overall simplicity (this is attributed to the various complexities associated with the experimental setup), it is a nontrivial technique which necessitates significant amounts of computational cost to obtain accurate results (i.e. results comparable to those obtained using experimentation). Such factors prove to be sufficiently problematic during iterative design procedures. Dimensionality is one such issue associated with CFD. Though dimensionality may be addressed through altering coordinates from Cartesian to PARSEC parameters [1], the resulting shape may not be optimal. In particular, obtaining an optimal airfoil shape for given flight conditions is an arduous task. Such an optimization problem incorporates many factors, often times involving latent variables in the process, and may even lead itself to a multidisciplinary design optimization (MDO) problem. To this end, Machine Learning presents itself as the hero in disguise; we may leverage learning to curtail the time necessary to ascertain the aforementioned optimal airfoil properties. [2][3][4]
## II. Problem Statement 
The problem at hand is presented as follows; given a dataset with airfoil parameters (features) and corresponding aerodynamic performances (labels), clustering will be conducted, and neural networks will be trained to predict sufficiently accurate aerodynamic properties of interest. The objective at hand is to determine an ideal clustering method as well as an optimal number of clusters for the cluster analysis. Additionally, we seek to determine an architecture, along with associated weights, which properly models the given network.

## III. Data Collection and Pre-Processing
The airfoil shapes has be collected from the UIUC Airfoil Database, which contains the coordinates for approximately 1600 airfoils. Each Arifoil data is stored in a text file containing the name of the airfoil and the discrete x and y shape coordinates. The airfoil data has then be ran through the XFOIL software [5], a fast flow solver that takes as an input the discrete airfoil shape coordinates, to obtain aerodynamic properties such as lift and drag coefficients, as well as other quantities of interest. These aerodynamic properties will act as the features for clustering the airfoils. 

Out of the 1600 initial airfoils in the dataset, only 600 were able to make XFoil converge. Indeed, XFoil is very sensitive to any shape discontinuity and the order of the shape coordinates in the shape file. The number of shape coordinates, and the order of the shape coordinates in each element of the dataset is inconsistent. Some of the airfoils in the dataset do not contain enoough coordinate points to make XFoil converge and find the aerodynamic coefficients. In other words, the data cleaning was performed automatically by XFoil keeping only the airfoils with enough information to extract the data.

As mentioned previously, the number of shape coordinates for each airfoil file is not consistent. In average, each file contains about 20 discrete points which were not extracted at the same x coordinate. Moreover, the discrete shape parameterization can lead to poor results when applied to a machine learning framework or an optimization process as any small change in the discrete shape coordinates can lead to a degenerated shape that does not correspond to a feasible airfoil shape. The goal of the data preprocessing for this project is to reduce the dimensionality of the dataset, make it homogeneous, readable and practicalfor any machine learning framework. There exist multiple methods to perform this such as changing the shape parameterization technique, such as using Bezier curves or the NACA 4-digit method. One of the methods that has proven to be effective in terms of representing airfoil shapes is the PARSEC method that only contains 11 shape parameters, reducing the dimensionality in half. [INSERT PARSEC FIGURE] An illustration of the PARSEC parameterization is shown in Figure XX.

The final data pre-processing is to convert the airfoil dataset using discrete shape parameterization into a dataset using a PARSEC parameterization. To do so, an optimization algorithm is applied to each airfoil in the dataset in order to find the most fitting PARSEC parameters resulting in the closest airfoil shape to the original. The final dataset is then a list of 600 airfoils having as features the corresponding 11 PARSEC parameters and their lift and drag coefficients as labels. 

## IV. Methods
Both supervised and unsupervised learning will be used to classify airfoils based on their properties and to predict the airfoil class based on the PARSEC parameters. Firstly, clustering techniques were used to identify airfoil classes based on properties such as lift, drag and volume. Several clustering techniques were used, including K-Means and Gaussian Mixture Models (GMMs). Next, an artificial neural network will be used to implement a technique such as logistic regression to classify airfoils into classes. The PARSEC parameters will act as the features.

### Airfoil Clustering

Several clustering algorithms were implemented. The data consists of three features: the lift coefficient (Cl), the drag coefficient (Cd) and the volume. A visualization of the data is shown below. 

![Visualization of airfoil data](Images/airfoil_data.png?raw=true "Visualization of airfoil data")

Visualization of airfoil data


GMM models were trained on the data with a varying number of clusters/components. A visualization of the result using six clusters is shown below. 

![GMM with six clusters](Images/gmm_six_clusters.png?raw=true)

GMM with six clusters

The GMM models were evaluated using two metrics:
1. The silhouette coefficient
2. 2. The Davies-Bouldin index

The results for a number of clusters varying from 2 to 12 can be seen below. 

![Silhouette coefficient for GMM models](Images/gmm_sil_coeff.png?raw=true)

Silhouette coefficient for GMM models

![Davies-Bouldin Index for GMM models](Images/gmm_dbindex.png?raw=true)

Davies-Bouldin Index for GMM models


### Airfoil Classification

Based on the results from the clustering, all of the training airfoils will be labelled with the appropriate cluster assignment. These act as the labels for classification, while the PARSEC parameters act as the features.

## V. Potential results and Discussion
This project will enable the creation of a low-runtime model that maps an airfoil shape to its aerodynamic performance. Ideally, the predictions made should be comparable to those made by a software such as XFOIL. In turn, this model could be used within a larger project relating to aerodynamic shape optimization. 

## Sources
1. Sobieczky, H.: Parametric Airfoils and Wings, Notes on Numerical Fluid Mechanics, edited by K. Fujii and G.S. Dulikravich, Vol. 68, Vieweg Verlag, 1998, pp. 71-88
2. Chen W., Chiu K., Fuge M., Aerodynamic Design Optimization and Shape Exploration using Generative Adversarial Network, University of Maryland, College Park, Maryland, 20742
3. Di Angelo L., Di Stefano P., "An evolutionary geometric primitive for automatic design synthesis of functional shapes: The case of airfoils"
4. Zhang Y., Sung W., Mavris D., Application of Convolution Neural Network to Predict Airfoil Lift Coefficient, Georgia Institute of Technology, Atlanta, Georgia, 30332
5. Drela, M.: XFOIL: Subsonic Airfoil Development System, Massachusetts Institute of Technology. https://web.mit.edu/drela/Public/web/xfoil/ 






