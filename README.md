# Learning Frameworks for Various Airfoils

## I. Introduction
Computational Fluid dynamics (CFD) is very complicated and computationally expensive to use. Such disadvantage can be problematic, especially during desiging process which requires multiple design iteration processes. Machine learning can be used to curtail time to determine optimal airfoil properties. To reduce the dimensionality of the airfoil, instead of describing airfoil with Cartesian coordinates, PARSEC parameterization is used.


In essence, for a given parsec parameter range, we can predict the range of lift and drag and then group thes data into classes based on these ranges. 


Introduce Gabriel's paper, Martin's paper, what has been done in the past.

Morphing wing aircraft are flight vehicles which are able to alter their geometric configurations to interact with their environments in an optimal fashion.

## II. Problem Statement 
Every airfoil has unique aerodynamic performances such as lift, drag and moment. Given dataset with airfoil parameters and associated aerodynamic performances, clustering and neural network will be trained to predict aerodynamic properties as accurate as possible. 


Introduce lift and drag. Why in particular we are interested in the optimization techniques that we are. So, reducing drag will be commensurate with savings on fuel for a given flight mission, etc. 
Look at lift, drag, moment and volume. We investigate the volume as a parameter due to its role in the structural and fuel storage considerations of the wing. In particular, the airfoil section must have sufficient thickness to withstand the various aerodynamic loads imparted to the section. Additionally, there need be enough space for fuel storage.


## III. Data Collection
The airfoil shapes will be collected from the UIUC Airfoil Database, which contains the coordinates for approximately 1600 airfoils. The airfoil data will then be run through the XFOIL software to obtain aerodynamic properties such as lift and drag coefficients, as well as other quantities of interest. These aerodynamic properties will act as the features for clustering the airfoils.

Next the xy coordinates for the airfoils can be converted to PARSEC parameters for each airfoil. The PARSEC parameters will act as the features for the airfoil classification, with the previously identified clusters acting as the labels. 

## IV. Methods
Two methods will be used to classify airfoils based on their properties and to predict the airfoil class based on the PARSEC parameters. Firstly, clustering techniques will be used to identify airfoil classes based on properties such as lift and drag coefficients. Given that the distribution within each cluster is not known apriori, Gaussian Mixture Model and/or Density-based clustering will be used. Measures such as Beta-CV will be used to evaluate alternate clustering approaches. Next, a neural netwrok will be used to implement a technique such as logictic regression to classify arifoils into classes. The PARSEC parameters will act as the features. XFOIL can be used to generate test data to compute the accuracy of the classification.

## V. Potential results and Discussion
Clustering and neural network should provide relationship between parsec parameters and aerodynamic performances and quick estimate of them.

## Sources
1. Chen W., Chiu K., Fuge M., Aerodynamic Design Optimization and Shape Exploration using Generative Adversarial Network, University of Maryland, College Park, Maryland, 20742
2. Di Angelo L., Di Stefano P., "An evolutionary geometric primitive for automatic design synthesis of functional shapes: The case of airfoils"





