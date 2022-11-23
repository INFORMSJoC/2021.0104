import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from deslib.dcs import MCB
from deslib.dcs import OLA
from deslib.dcs import Rank
from deslib.des import DESP
from deslib.des import KNORAE
from deslib.des import KNORAU

rng = np.random.RandomState(123456)

data = fetch_openml(name='diabetes', version=1, cache=False, as_frame=False)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

# Normalizing the dataset to have 0 mean and unit variance.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training a pool of classifiers using the bagging technique.
pool_classifiers = BaggingClassifier(DecisionTreeClassifier(random_state=rng),
                                     random_state=rng)
pool_classifiers.fit(X_train, y_train)


mcb = MCB(pool_classifiers, with_IH=True, random_state=rng)
ola = OLA(pool_classifiers, with_IH=True, random_state=rng)
rank = Rank(pool_classifiers, with_IH=True, random_state=rng)
des_p = DESP(pool_classifiers, with_IH=True, random_state=rng)
kne = KNORAE(pool_classifiers, with_IH=True, random_state=rng)
knu = KNORAU(pool_classifiers, with_IH=True, random_state=rng)
list_ih_values = [0.0, 1./7., 2./7., 3./7.]

list_ds_methods = [method.fit(X_train, y_train) for method in
                   [mcb, ola, rank, des_p, kne, knu]]
names = ['MCB', 'OLA', 'Mod. Rank', 'DES-P', 'KNORA-E', 'KNORA-U']

# Plot accuracy x IH
fig, ax = plt.subplots()
for ds_method, name in zip(list_ds_methods, names):
    accuracy = []
    for idx_ih, ih_rate in enumerate([0.0, 0.14, 0.28, 0.42]):
        ds_method.IH_rate = ih_rate
        accuracy.append(ds_method.score(X_test, y_test))
    ax.plot(list_ih_values, accuracy, label=name)

plt.xticks(list_ih_values)
ax.set_ylim(0.65, 0.80)
ax.set_xlabel('IH value', fontsize=13)
ax.set_ylabel('Accuracy on the test set (%)', fontsize=13)
ax.legend()

plt.show()