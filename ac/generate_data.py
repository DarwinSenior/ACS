import numpy as np
import scipy as sp
import pandas as pd
import glob

def extract_feature(arr: np.ndarray):
    arr = arr.astype(np.float64)
    f1 = np.mean(arr)
    f2 = sp.std(arr, ddof=1)
    f3 = np.mean(np.abs(arr[:-1]-arr[1:]))
    f4 = f3 / f2
    f5 = np.mean(np.abs(arr[:-2]-arr[2:]))
    f6 = f5 / f2
    return f1,f2,f3,f4,f5,f6
features = ['X-axis', 'Y-axis', 'Z-axis', 'Celsius', 'EDA(uS)']
def extract_extra_features(data: pd.DataFrame):
    x,y,z,t,e = [data[c].as_matrix().astype(np.float64)
            for c in features]
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    f1 = np.mean(np.sqrt(dx**2+dy**2+dz**2))
    f2 = np.max(t) - np.min(t)
    f3 = np.max(e) - np.min(e)
    return f1, f2, f3

columns = ['%s-%s'%(attr, value)
        for attr in ['X', 'Y', 'Z', 'T', 'E']
        for value in ['m', 's', 'md', 'nmd', 'md2', 'nmd2']]
def extract_colum(filepath: str, extra=False):
    data = pd.read_csv(filepath)
    fs = np.zeros((6, 5))
    for i, column in enumerate(features):
        fs[:,i] = extract_feature(data[column].as_matrix())
    filename = filepath.split('/')[-1].replace('.csv', '').lower()
    name, emotion, number = filename.split('_')
    df0 = pd.DataFrame([[filename, emotion, int(number)]], columns=['name', 'emotion', 'numerical'])
    df1 = pd.DataFrame([fs.ravel()], columns=columns)
    if extra:
        df2 = pd.DataFrame([extract_extra_features(data)], columns=['dist-m', 't-diff', 'e-diff'])
        return pd.concat([df0, df1, df2], axis=1)
    else:
        return pd.concat([df0, df1], axis=1)

if __name__=='__main__':
    filepaths = glob.glob('DataSet/*.csv')
    df: pd.DataFrame
    df = pd.concat(map(extract_colum, filepaths), ignore_index=True)
    df.to_csv('DY_featureSet1.csv')
    df = pd.concat((extract_colum(f, True) for f in filepaths), ignore_index=True)
    df.to_csv('DY_featureSet2.csv')
