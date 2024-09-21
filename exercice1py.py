import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error


# SVM/SVC models
#

for DataFileName in ['SVM_dataset_1.csv','SVM_dataset_2.csv']:
    print("--------------------------------------")
    print("Processing file %s..."%(DataFileName))
    df = pd.read_csv(DataFileName)
    T = len(df.index)
    X = np.zeros((T, 2))
    X[:, 0] = df['X1'].values
    X[:, 1] = df['X2'].values
    Y = df['Y'].values

    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    # SVC = SVM classifier
    svm = SVC(kernel='linear')
    svm.fit(X, Y)

    # Let's compute the MSE & hit ratio
    Y_hat = svm.predict(X)
    MSE = mean_squared_error(Y_hat,Y)
    print("MSE = ",MSE)
    HitRatio = np.sum(Y_hat == Y) / T
    print("Hit Ratio = %1.1f %%"%(100. * HitRatio))

    # mesh
    min_val_x = X[:, 0].min()
    max_val_x = X[:, 0].max()
    min_val_y = X[:, 1].min()
    max_val_y = X[:, 1].max()

    x_min, x_max = min_val_x - 1, max_val_x + 1
    y_min, y_max = min_val_y - 1, max_val_y + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # prediction on a mesh to represent the model
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap='Spectral', alpha=0.8)
    colors = ['blue' if val == 1 else 'red' for val in Y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('SVM model ' + DataFileName)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

plt.show()