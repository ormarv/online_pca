# Readme


This is the final project for the course Data Science in Practice at Télécom Paris. There are two available versions, based on different Incremental/Online PCA algorithms.


The two implemented algorithms are PerturbPCA in [perturb_ipca.py](./perturb_ipca.py) and ReducedPCA in [reduced_ipca.py](./reduced_ipca.py). PerturbPCA is based on the algorithm in [[1]](https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf) and it's simplified [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html) implementation. While different in some details, our implementation follows similar logic to the sklearn code by avoiding an occasional optimization relying on QR decompositions. ReducedPCA is based on the algorithm in [[2]](https://home.ttic.edu/~klivescu/papers/arora_etal_allerton2012.pdf). The names PerturbPCA and ReducedPCA are based on an organization of the literature made in [[3]](https://arxiv.org/pdf/1511.03688.pdf), which splits online PCA algorithms into different types. 

The  code is [river](https://github.com/online-ml/river) compatible and ReducedPCA and PerturbPCA can be integrated into river pipeline as a dimmensionality reduction preprocessing step (for an example see [test](./tests.ipynb)). 


**Contents of each file**
* perturb_ipca.py contains an implementation of PerturbPCA. It is the similar to the scikit-learn implementation.
* reduced_ipca.py contains the implementation  ReducedPCA.
* utils.py contains shared utility functions.
* tests.ipynb contains various tests of both implementations.


**Referemces**

[1] D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual   Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3, pp. 125-141, May 2008. 


[2] Arora, R., Cotter, A., Livescu, K., and Srebo, N. (2012). Stochastic optimization for PCA and PLS. In 50th Annual Conference on Communication, Control, and Computing (Allerton), pages 861–868.


[3] Cardot, H., & Degras, D. (2018). Online principal component analysis in high dimension: Which algorithm to choose?. International Statistical Review, 86(1), 29-50.
