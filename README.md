# Learning Frameworks for Various Airfoils

## I. Introduction
Although Computational Fluid Dynamics (CFD) surpasses experimental techniques in terms of time, cost and overall simplicity (this is attributed to the various complexities associated with the experimental setup), it is a nontrivial technique which necessitates significant amounts of computational cost to obtain accurate results (i.e. results comparable to those obtained using experimentation). Such factors prove to be sufficiently problematic during iterative design procedures. Dimensionality is one such issue associated with CFD. Though dimensionality may be addressed through altering coordinates from Cartesian to PARSEC parameters [1], the resulting shape may not be optimal. In particular, obtaining an optimal airfoil shape for given flight conditions is an arduous task. Such an optimization problem incorporates many factors, often times involving latent variables in the process, and may even lead itself to a multidisciplinary design optimization (MDO) problem. To this end, Machine Learning presents itself as the hero in disguise; we may leverage learning to curtail the time necessary to ascertain the aforementioned optimal airfoil properties. [1][2][3]
## II. Problem Statement 
The problem at hand is presented as follows; given a dataset with airfoil parameters (features) and corresponding aerodynamic performances (labels), clustering will be conducted, and neural networks will be trained to predict sufficiently accurate aerodynamic properties of interest. The objective at hand is to determine an ideal clustering method as well as an optimal number of clusters for the cluster analysis. Additionally, we seek to determine an architecture, along with associated weights, which properly models the given network.

## III. Data Collection
The airfoil shapes will be collected from the UIUC Airfoil Database, which contains the coordinates for approximately 1600 airfoils. The airfoil data will then be run through the XFOIL software [2] to obtain aerodynamic properties such as lift and drag coefficients, as well as other quantities of interest. These aerodynamic properties will act as the features for clustering the airfoils.

Next the coordinates for the airfoils can be converted to PARSEC parameters for each airfoil. The PARSEC parameters will act as the features for the airfoil classification, with the previously identified clusters acting as the labels. 

## IV. Methods
Both supervised and unsupervised learning will be used to classify airfoils based on their properties and to predict the airfoil class based on the PARSEC parameters. Firstly, clustering techniques will be used to identify airfoil classes based on properties such as lift, drag and moment coefficients. Given that the distribution within each cluster is not known apriori, Gaussian Mixture Model and Density-based clustering will be used. Metrics such as Beta-CV will be used to evaluate alternate clustering approaches and number of clusters. Next, an artificial neural network will be used to implement a technique such as logistic regression to classify airfoils into classes. The PARSEC parameters will act as the features.

## V. Potential results and Discussion
This project will enable the creation of a low-runtime model that maps an airfoil shape to its aerodynamic performance. Ideally, the predictions made should be comparable to those made by a software such as XFOIL. In turn, this model could be used within a larger project relating to aerodynamic shape optimization. 

## Sources
1. Sobieczky, H.: Parametric Airfoils and Wings, Notes on Numerical Fluid Mechanics, edited by K. Fujii and G.S. Dulikravich, Vol. 68, Vieweg Verlag, 1998, pp. 71-88
2. Drela, M.: XFOIL: Subsonic Airfoil Development System, Massachusetts Institute of Technology. https://web.mit.edu/drela/Public/web/xfoil/ 
3. Chen W., Chiu K., Fuge M., Aerodynamic Design Optimization and Shape Exploration using Generative Adversarial Network, University of Maryland, College Park, Maryland, 20742
4. Di Angelo L., Di Stefano P., "An evolutionary geometric primitive for automatic design synthesis of functional shapes: The case of airfoils"
5. Zhang Y., Sung W., Mavris D., Application of Convolution Neural Network to Predict Airfoil Lift Coefficient, Georgia Institute of Technology, Atlanta, Georgia, 30332





