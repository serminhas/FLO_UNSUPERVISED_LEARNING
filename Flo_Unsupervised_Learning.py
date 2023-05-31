import warnings
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 80)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


df_ = pd.read_csv("FLO_UNSUPERVISED_LEARNING/flo_data_20k.csv")
df=df_.copy()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile().T)

check_df(df)


df['tenure'] = (pd.to_datetime('today') - pd.to_datetime(df['first_order_date'])).dt.days

df['recency'] = (pd.to_datetime('today') - pd.to_datetime(df['last_order_date'])).dt.days

df['frequency'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']

df['monetary'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=4, car_th=20)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=False)

def one_hot_encoder(df, categorical_cols, drop_first=False):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df

df = one_hot_encoder(df, cat_cols, drop_first=True)

#With K-Means

sc = MinMaxScaler((0, 1))
df_new = sc.fit_transform(df[num_cols])
df_new = pd.DataFrame(df_new, columns=num_cols)
df_new = df_new.merge(df.iloc[:, -7:], left_index=True, right_index=True)

kmeans = KMeans()
ssd = []
K = range(1, 40)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df_new)
    ssd.append(kmeans.inertia_)

kmeans.get_params()
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

plt.plot(ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show(block=True)

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df_new)
elbow.show(block=True)
elbow.elbow_value_

#Model with K-Means
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_new)

clusters = kmeans.labels_

df["cluster"] = clusters
df["cluster"] = df["cluster"] + 1

def Ratio_(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

Ratio_(df, "cluster")

df.groupby("cluster").agg(["mean", "median", "count", "sum"]).T

#Hierchical Clustering

hi_ward = linkage(df_new, "ward")

plt.figure(figsize=(20, 10))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hi_ward,
            truncate_mode="lastp",
            p=7,
            show_contracted=True,
            leaf_font_size=10)
plt.show(block=True)


plt.figure(figsize=(20, 10)) #grafiğin boyutunu belirler
plt.title("Hiyerarşik Kümeleme Dendogramı")
dend = dendrogram(hi_ward,
                 truncate_mode="lastp",
                  p=7,
                  show_contracted=True,
                  leaf_font_size=10)
plt.axhline(y=60, color="b", linestyle="--")
plt.axhline(y=50, color="r", linestyle="--")
plt.show(block=True)

#Model with Hierchical Clustering
h_cluster = AgglomerativeClustering(n_clusters=6, linkage="ward")
h_clusters = h_cluster.fit_predict(df_new)
df["h_cluster"] = h_clusters
df["h_cluster"] = df["h_cluster"] + 1

def Ratio_(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

Ratio_(df, "h_cluster")

df.groupby("h_cluster").agg(["mean", "median", "count", "sum"]).T