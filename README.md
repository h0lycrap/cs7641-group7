# Learning Frameworks for Various Airfoils

## I. Introduction
CFD too complicated and computationally expensive to use, especially during desiging process because various configurations and designs need to be evaluated to find optimal design.
use ML to curtail time to determine optimal airfoil properties.

In essence, for a given parsec parameter range, we can predict the range of lift and drag and then group thes data into classes based on these ranges. 


Introduce Gabriel's paper, Martin's paper, what has been done in the past.

Morphing wing aircraft are flight vehicles which are able to alter their geometric configurations to interact with their environments in an optimal fashion.

## II. Problem Statement 
Introduce lift and drag. Why in particular we are interested in the optimization techniques that we are. So, reducing drag will be commensurate with savings on fuel for a given flight mission, etc. 
Look at lift, drag, moment and volume. We investigate the volume as a parameter due to its role in the structural and fuel storage considerations of the wing. In particular, the airfoil section must have sufficient thickness to withstand the various aerodynamic loads imparted to the section. Additionally, there need be enough space for fuel storage.


## III. Data Collection
UIUC Airfoil Database for the data set. 

Wish to determine a correlation between PARSEC parameters and our categories of interest.

Run data in XFOIL. 

The coordinate of various airfoil will be collected from UIUC Airfoil Database. The aerodynamic properties of each airfoil, such as lift coefficient, drag coefficient, moment coefficient ...etc, will be computing using XFOIL.

## IV. Methods
Two methods will be used to identify the properties of airfoils. First method is to used clustering; given the distribution within each cluster is not known apripori, Gaussian Mixture model and/or Density-based clustering will be used. Within each cluster, the airfoils will be parameterized using PARSEC to find commonality within the clusters. Beta-CV, and Normalized cut will be used to evaluate quality of the clustering. The second method is to use artificial neural network to predict the property given defining parameters. The accuracy of the neural network will be computed using the labeled data from XFOIL. 



## V. Potential results and Discussion






