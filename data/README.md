## **Data Download Instructions**

Please download all the data from the shared link (https://hkustconnect-my.sharepoint.com/:f:/g/personal/cyanao_connect_ust_hk/EoZVw2KQMpRLvzMc4z01zzUBiyommxJg2_T_XHxVhbyjqw?e=y27b02) under this folder.



## **Data Loading Code**

You can load the data using the following Python code: 

```python
import numpy as np
```



**Load intrinsic matrices**

```python
K1, K2 = np.loadtxt('../data/Intrinsic4Recon.npz')
K1, K2 = K1.reshape(3, 3), K2.reshape(3, 3)
```



**Load the points to be visualized**

```python
coords = np.loadtxt('../data/VisPts.npz')
x1s = coords[:, 0].astype(np.int32)
y1s = coords[:, 1].astype(np.int32)
```



**Load ground truth poses** 

```python
GT_Pose = np.loadtxt('../data/GTPoses.npz').reshape(-1, 3, 4)
```

The `GT_Pose` is an $n \times 3 \times 4$ matrix, where each entry represents a transformation matrix that combines rotation (a $3 \times 3$ matrix) and translation (a 3D vector) in a single structure. The last column of the matrix corresponds to the translation component.