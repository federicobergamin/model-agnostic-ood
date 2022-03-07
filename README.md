# model-agnostic-ood

Code for the paper "Model-agnostic out-of-distribution detection using combined statistical tests" (AISTATS 2022)

Currently polishing the code. It will be available for the main conference. 

We will sharing only the code for computing our statistics for the different considered model. For the models we will refer to the following GitHub repository:
- PixelCNN: we used Lucas Caccia's repository. [Link](https://github.com/pclucas14/pixel-cnn-pp)
- Flows: we use Polina Kirichenko's repository. [Link](https://github.com/PolinaKirichenko/flows_ood)
- HAVE: we used Jakob Havtorn's repository. [Link](https://github.com/JakobHavtorn/hvae-oodd)

Therefore the part of loading the models and the dataset in our files are repository specific. If you want to use the same methods with different model you should change the initial part of our files.
