import numpy as np
from pactl.nn.projectors import SparseOperator

np.random.seed(seed=21)


D = int(2.e3)
dds = [10, 100, 250, 500, 1000, 1500]
conds = []

for dd in dds:
    P = SparseOperator(D, dd, params=None, names=None)
    P = P.to_dense()
    P /= np.sqrt(D)
    print(P.shape)

    aux = P.T @ P
    cond = np.linalg.cond(aux)
    print(f'Cond: {cond:1.3e}')
    cond2 = np.linalg.cond(aux) ** 2.
    print(f'Cond: {cond:1.3e}')
    conds.append(cond2)


np.save('dds.npy', arr=np.array(dds))
np.save('means.npy', arr=np.array(conds))
