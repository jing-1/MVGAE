# MVGAE: Multi-modal Variational Graph Auto-encoder for Recommendation Systems
This is our implementation of MVGAE for recommendation systems associated with:

>**MVGAE: Multi-modal Variational Graph Auto-encoder for Recommendation Systems,**  
>Jing Yi and Zhenzhong Chen  
 
## Environment Requirement
- Pytorch == 1.4.0
- torch-cluster == 1.5.4
- torch-geometric == 1.4.1
- torch-scatter == 2.0.4
- torch-sparse == 0.6.1
- torch-spline-conv == 1.2.1
## Model
- BaseModel.py: Implementation of graph convolutional operator using Pytorch Geometric library.
- Model.py: Implementation of graph convolutional networks (GCNs), Product-of-exprets (PoE) and our MVGAE.
### **If you find our codes helpful, please kindly cite the following paper. Thanks!**
	@article{mvgae,
	  title={Multi-modal Variational Graph Auto-encoder for Recommendation Systems},
	  author={Yi, Jing and Chen, Zhenzhong},
	  year={2021},
	}
