import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(dataloc='../data/'):
    df = pd.read_csv(dataloc)
    col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df.columns = col_names
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    classes = {val: idx for idx, val in enumerate(sorted(df['class'].unique()))}
    y_train = [classes[y] for y in y_train]
    y_test = [classes[y] for y in y_test]
    return {
        'train': list(zip(X_train.values.tolist(), y_train)),
        'test': list(zip(X_test.values.tolist(), y_test))}