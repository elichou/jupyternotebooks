# jupyternotebooks
Notebooks

### GFN_models/
Ce dossier contient trois modèles keras: 
1. input : image pair, output : reconstructed image
2. input : posture pair, output : latent repr array
3. input : posture pair, output : reconstructed posture

### Visual_learning_reaching/ 
Ce dossier contient **armcodlib** ainsi que trois modèles alternatifs implémentant différents types de gated autoencoders.

Dans **armcodlib/** on trouvera le notebook principal:
* **Visual_learning_reaching/armcodlib/Visual_learning_3d.ipynb**
Ainsi qu'une version plus légère sans les outputs:
* **Visual_learning_reaching/armcodlib/Visual_learning_3d-Copy1.ipynb**

On trouvera dans armcodlib.py les fonctions utilisées dans le notebook.
