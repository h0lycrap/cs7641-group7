# Learning Frameworks for Various Airfoils

## I. Introduction
Even though Computational Fluid dynamics (CFD) is relative cheaper compared to physical experiments, it is very complicated and computationally expensive to get accurate results. Such disadvantages can be problematic during designing processes which require multiple design iterations. Even if the dimensionality of the airfoil is reduced from Cartesian coordinates to PARSEC parameters, finding optimal airfoil is not a trivial task. However, machine learning can be used to curtail time to determine optimal airfoil properties by providing prediction almost instantaneously.


## II. Problem Statement 
Given dataset with airfoil parameters and associated aerodynamic performances, clustering and neural network will be trained to predict aerodynamic properties as accurate as possible. The goal is to find ideal clustering method and number of clusters for clustering and appropriate architecture of the network and its associated weight for neural network. 


## III. Data Collection
The airfoil shapes will be collected from the UIUC Airfoil Database, which contains the coordinates for approximately 1600 airfoils. The airfoil data will then be run through the XFOIL software to obtain aerodynamic properties such as lift and drag coefficients, as well as other quantities of interest. These aerodynamic properties will act as the features for clustering the airfoils.

Next the x,y coordinates for the airfoils can be converted to PARSEC parameters for each airfoil. The PARSEC parameters will act as the features for the airfoil classification, with the previously identified clusters acting as the labels. 

## IV. Methods
Both supervised and unsupervised learning will be used to classify airfoils based on their properties and to predict the airfoil class based on the PARSEC parameters. Firstly, clustering techniques will be used to identify airfoil classes based on properties such as lift, drag and moment coefficients. Given that the distribution within each cluster is not known apriori, Gaussian Mixture Model and Density-based clustering will be used. Metrics such as Beta-CV will be used to evaluate alternate clustering approaches and number of clusters. Next, an artificial neural network will be used to implement a technique such as logistic regression to classify airfoils into classes. The PARSEC parameters will act as the features.

## V. Potential results and Discussion
Clustering and neural network should provide relationship between parsec parameters and aerodynamic performances and quick estimate of them. Ideally both method should provide prediction comparable to XFOIL.

## Sources
1. Chen W., Chiu K., Fuge M., Aerodynamic Design Optimization and Shape Exploration using Generative Adversarial Network, University of Maryland, College Park, Maryland, 20742
2. Di Angelo L., Di Stefano P., "An evolutionary geometric primitive for automatic design synthesis of functional shapes: The case of airfoils"
3. Zhang Y., Sung W., Mavris D., Application of Convolution Neural Network to Predict Airfoil Lift Coefficient, Georgia Institute of Technology, Atlanta, Georgia, 30332





