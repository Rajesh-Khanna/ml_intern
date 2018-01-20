import pandas as pd
from sklearn.cluster import MeanShift,KMeans
def load_data():
    df = pd.read_csv('data/olympics.csv',skiprows=1)
    old_names = df.filter(like='01 !').columns.tolist()+df.filter(like='02 !').columns.tolist()+df.filter(like='03 !').columns.tolist()
    new_names = ['Gold','Gold','Gold','Silver','Silver','Silver','Bronze','Bronze','Bronze']
    df = df.rename(columns=dict(zip(old_names,new_names))) 
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: x[:x.find('(')])
    df = df.set_index(df.columns[0]) 
    df = df.drop(['Total'])
    return df

def first_country(df):
    return df.iloc[0]

def gold_medal(df):
    return df.ix[:,11].idxmax()

def biggest_difference_in_gold_medal(df):
    return (df.ix[:,1]-df.ix[:,6]).idxmax()

def get_points(df):
    return df.assign(Points= df.ix[:,11]*3 + df.ix[:,12]*2 + df.ix[:,13])

def kmeans(df):
    #We can use multiple k values and see which one gives best clustering. but to keep it short i used meanshift
    ms = MeanShift()
    ms.fit(df)
    k = len(ms.cluster_centers_)
    km = KMeans(n_clusters = k)
    km.fit(df)
    return km.cluster_centers_

df = load_data()
c1 = first_country(df)
gc = gold_medal(df)
dif = biggest_difference_in_gold_medal(df)
df = get_points(df)
print(kmeans(df))