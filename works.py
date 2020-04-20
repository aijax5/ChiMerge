from discretization.chi_merge import *
from sklearn.datasets import load_iris

# df = pd.read_csv("dataset/iris/train.csv")
# features = list(df.columns.values)
# X = df[features[0:-1]]
# y = df[features[-1]]
def load_iris_for_here():
    data = load_iris()
    features = pd.DataFrame(data['data'], columns=['a', 'b', 'c', 'd'])
    target = pd.Series(data=data['target'], name='target')

    return features, target

X , y = load_iris_for_here()
chi_merge = ChiMerge(con_features=X.columns, significance_level=0.1, n_jobs=-3)

op = chi_merge.fit_transform(X=X,y=y)
print("op: \n",op)