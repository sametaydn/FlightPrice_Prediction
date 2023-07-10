import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
#from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from pandas_profiling import ProfileReport
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Veri Yükleme
def reading_data(business_path, economy_path):
    """
    Verilen pathlerdeki verileri okuyup birleştirir.
    :param business_path: Business satışlarına ait bilgileri içerir.
    :param economy_path: Economy satışlarına ait bilgileri içerir.
    :return: Business ve Economy verileri birleştirilmiş bir pandas DataFrame
    """
    business = pd.read_csv(business_path)
    economy = pd.read_csv(economy_path)
    business['class'] = 'business'
    economy['class'] = 'economy'
    df = pd.concat([economy, business])
    return df


def check_df(dataframe, head=5):
    """
    Verilen dataframe objesinin genel özelliklerini ekrana bastırır
    :param dataframe: İncelenmek istenen veriyi içeren pandas DataFrame objesi
    :param head: Ne kadar verinin ekrana bastırılması istendiği
    """
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
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("#####################################################")

def cleaning(df):
    """
    Verilen df içerisindeki gereksiz değerleri temizler
    :param df: Temizlenmesi istenen veriyi içeren pandas DataFrame
    :return: Temizlenmiş veriyi döndürür
    """
    # Stop sütunundaki gereksiz karakterleri silme ve temizleme
    df["stop"] = df["stop"].str.replace("\t", "")
    df["stop"] = df["stop"].str.replace("\n", "")
    df.loc[df["stop"] == 'non-stop ', "stop"] = "0"
    df['stop'] = df['stop'].str[0]
    df['stop'] = df['stop'].astype(int)

    # price object --> int dönüşümü
    df['price'] = df['price'].str.replace(',', '')
    df['price'] = df['price'].astype(int)
    return df


def get_part_of_day(x):
    """
    İçerisine aldığı saat bilgisine uygun gün dilimini döndürür.
    :param x: Saat bilgisi
    :return: Karşılık geldiği dün dilimi
    """
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return 'Afternoon'
    elif (x > 16) and (x <= 20):
        return 'Evening'
    elif (x > 20) and (x <= 24):
        return 'Night'
    elif x <= 4:
        return 'Late Night'


def feature_engineering(df):
    """
    Verilen veriseti üzerinde öznitelik mühendisliği yapar.
    :param df: İşlenmek istenen veri seti
    :return: Yeni sütunlar eklenmiş DataFrame
    """
    # Kalkış ve varış saatlerini kategorik hale çevirme
    df['dep_time_h'] = pd.to_datetime(df['dep_time'], format='%H:%M').dt.hour
    df['arr_time_h'] = pd.to_datetime(df['arr_time'], format='%H:%M').dt.hour
    df['departure_time_ranges'] = df.dep_time_h.apply(lambda x: get_part_of_day(x))
    df['arrival_time_ranges'] = df.arr_time_h.apply(lambda x: get_part_of_day(x))

    # Tarihlerdeki günleri çekme
    df['day_name'] = pd.to_datetime(df['date'], format="%d-%M-%Y").dt.day_name()

    # Haftasonu bilgisini ekleme
    df['is_weekend'] = np.where(df['day_name'].isin(['Sunday', 'Saturday']), 1, 0)

    # Uçuş süresini dakikaya çevirme
    h = df['time_taken'].str.extract('(\d+)h', expand=False).astype(float) * 60
    m = df['time_taken'].str.extract('(\d+)m', expand=False).astype(float)
    df['duration_min'] = h.add(m, fill_value=0).astype(int)

    #df['date'] = pd.to_datetime(df['date'], format='%d-%M-%Y')
    df.drop(["ch_code", "num_code", "dep_time", "arr_time", "time_taken", "dep_time_h", "arr_time_h"],
            axis=1, inplace=True)
    return df



def preprocessing(df):
    """
    Ön işleme adımlarını barındırır
    :param df: pandas DataFrame
    :return: Ön işleme yaplmış DataFrame
    """
    df = cleaning(df)
    df = feature_engineering(df)
    return df


def cat_summary(dataframe, col_name, plot=False):
    """
    Kategorikal sütunlar için özet çıkartır.
    :param dataframe: Veriyi içeren dataframe
    :param col_name: Özet çıkartılacak sütunun adı
    :param plot: Görselleştirme yapılıp yapılmayacağı
    :return:
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    """
    Sayısal sütunlar için özet çıkartır.
    :param dataframe: Veriyi içeren dataframe
    :param numerical_col: Özet çıkartılacak sütunun adı
    :param plot: Görselleştirme yapılıp yapılmayacağı
    :return:
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Kategorik sütunlar ile hedef sütunu arasındaki ilişkiyi gözlemler
    :param dataframe: Veriyi içeren dataframe
    :param target: Tahminlenmesi istenen sütun
    :param categorical_col: Hedef sütunu ile ilişkisi gözlemlenecek sütun adı
    :return:
    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "TARGET_COUNT": dataframe.groupby(categorical_col)[target].count()}))
    print("#####################################")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Verisetinin içerdiği kategorik, sayılsal, sayısal olan ama az değer içerdiği için aslında kategorik verilerin
    bulunmasını sağlar.
    :param dataframe: İncelenecek veriseti
    :param cat_th: Sayısal sütunaların içerdiği değer sayısı bu değerden küçükse bu sütun kategoriktir.
    :param car_th: Kategorik sütunların içerdiği değer sayısı bu değerden büyükse bu sütun gereksiz olabilir.
    :return: kategorik sütunların adları, sayısal sütunların adları, kategorik ama kardinal sütunların adları
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
    print("####################Sütun Tipleri###################")
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print("#####################################################")
    return cat_cols, num_cols, cat_but_car


def correlation_analysis(dataframe):
    """
    Verilen dataframe içerisinde koralasyon inceler.
    :param dataframe: İncelenmesi istenen veriyi barındıran dataframe
    :return: Koralasyon matrisini görselleştirir
    """
    corr_matrix = dataframe.corr()
    plt.figure(figsize=(10, 10))
    # create mask
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
    # create heatmap with mask,diverging color palette and cutoff threshold
    sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1)
    # show plot
    plt.title("Korelasyon Analizi")
    plt.show()


# Aykırı değer limitlerinin bulunması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    """
    Sayılsan sütunun low_quantile-1.5 IQR  ve up_quantile+1.5 IQR  değerlerini döndürür.
    :param dataframe: İncelenecek sütunun içeren veriseti
    :param variable: İncelenecek sütunun adı
    :param low_quantile: Hangi yüzdeden öncesi için değerlendirilmeli
    :param up_quantile: Hangi yüzdeden sonrası için değerlendirilmeli
    :return: low_quantile-1.5 IQR  ve up_quantile+1.5 IQR
    """
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    """
    outlier_thresholds fonksiyonunun çıktısındaki  low_limit ve up_limit arasında kalmayan değerleri  low_limit ve
    up_limit'e eşitler
    :param dataframe: Aykırılardan temizlenmesi istenen sütunu içeren veriseti
    :param variable: Aykırılardan temizlenmesi istenen sütun adı
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Veride aykırı değer olup olmadığını kontrol eder.
    :param dataframe: Aykırı değer kontrolü yapılacak sütunu içeren veriseti
    :param col_name: Aykırı değer kontrolü yapılacak sütunun adı
    :param q1: outlier_thresholds fonksiyonu için low_quantile
    :param q3: outlier_thresholds fonksiyonu için up_quantile
    :return: Veri aykırı değer içeriyorsa True, içermiyorsa False
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def missing_values_table(dataframe, na_name=False):
    """
    Veride boş değer olup olmadığının kontrolünü yapar.
    :param dataframe: Boş değer kontrolü yapılacak olan dataframe
    :param na_name: Veride boş değerlerin özel bir adlandırması varsa
    :return: Boş değer olan sütunların listesi
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def rare_analyser(dataframe, target, cat_cols):
    """
    Verisetindeki kategorik sütunlarda her bir değeri, verinin geneline oranlar ve hedef sütunu için ortalama hesaplar.
    :param dataframe: İncelenecek verisetini içeren DataFrame
    :param target: Tahminlenmesi istenen sütun adı
    :param cat_cols: Kategorik sütunların listesi
    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    """
    Veri setindeki kategorik sütunların aldığı değerlerin, verinin geneline oranı eğer rare_perc'ten küçükse "Rare"
    olarak isimlendirir.
    :param dataframe:  İncelenecek verisetini içeren DataFrame
    :param rare_perc: "Rare" etiketi için gereken maximum oran
    :return:
    """
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    print(rare_columns)
    # rare sınıfa sahip değişkenlerin indexleri bulunup yerlerine Rare yazılır
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def analysis(df):
    """
    Bütün analiz işlemlerimizi içeren fonksiyon.
    :param df: Analiz yapılacak verisetini içeren DataFrame
    :return: Analiz sonrası oluşan yeni dataframe
    """
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # Kategorik Değişken Analizi (Analysis of Categorical Variables)
    print("##########Kategorik Değer Analizi##############")
    for col in cat_cols:
        cat_summary(df, col)

    # Sayısal Değişken Analizi (Analysis of Numerical Variables)
    print("##########Numerik Değer Analizi##############")
    for col in num_cols:
        num_summary(df, col)

    # Hedef Değişken Analizi (Analysis of Target Variable)
    print("##########Hedef Değişken Analizi##############")
    for col in cat_cols:
        target_summary_with_cat(df, "price", col)


    # Bağımlı değişkenin incelenmesi
    df["price"].hist(bins=100)
    plt.title("Price histogram dağılımı")
    plt.show()

    # Korelasyon analizi
    correlation_analysis(df)

    print("##########Aykırı Değer Kontrolü##############")
    # Aykırı değer kontrolü
    for col in num_cols:
        if col != "price":
            print(col, check_outlier(df, col, 0.05, 0.95))

    # Aykırı değer traşlama
    for col in num_cols:
        if col != "price":
            replace_with_thresholds(df, col)

    # Eksik Değer Analizi
    print("##########Eksik Değer Analizi##############")
    missing_values_table(df)

    # Kategorik sütunların aldığı değerlerin azlıklarınının kontrolü
    print("##########Rare Değer Analizi##############")
    rare_analyser(df, "price", cat_cols)

    # Kategorik sütunların nadir gelen değerlerinin tek bir değerde toplanması
    df = rare_encoder(df, 0.01)
    df['airline'].value_counts()

    return df


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    İçerisine verilen dataframedeki categorical_cols listesine one-hot encoder uygular.
    :param dataframe: Encode edilecek veriyi içeren DataFrame
    :param categorical_cols: Encode edilecek sütunların adlarını içeren liste
    :param drop_first: Encode edilirken ilk değerin düşürülüp düşürülmeyeceğini belirleyen pandas get_dummies parametresi.
    :return: Verilen kategorik değerleri encode edilmiş veriseti
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def encoding(df):
    """
    Veri değiştirme işlemlerini bir araya toplayan fonksiyon
    :param df: Veri değiştirme uygulanacak DataFrame
    :return: Değişim yapılmış DataFrame
    """
    cat_cols, _, _ = grab_col_names(df, car_th=40)

    # One-hot encoder
    df = one_hot_encoder(df, cat_cols, drop_first=True)

    # Standartlaştırma
    scaler = StandardScaler()
    df['duration_min'] = scaler.fit_transform(df['duration_min'].values.reshape(-1, 1))
    #joblib.dump(scaler, "scaler_2.bin")
    return df


def base_models(X, y, scoring="roc_auc"):
    """
    Belirlenen modelleri train veriseti ile eğitir.
    :param X: Eğitim için kullanılacak veriseti
    :param y: Eğitim için kullanılacak hedef değişkeni
    :param scoring: Skorlamanın tutulacağı metrikler
    :return: Sonuçların listesi
    """
    results = []
    print("Base Models....")
    classifiers = [("LR", LinearRegression(n_jobs=-1)),
                   ('KNN', KNeighborsRegressor(n_jobs=-1)),
                   ("DTR", DecisionTreeRegressor()),
                   ("RF", RandomForestRegressor(n_jobs=-1)),
                   ('Adaboost', AdaBoostRegressor()),
                   ('GBM', GradientBoostingRegressor()),
                   ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)),
                   ('CatBoost', CatBoostRegressor(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring, n_jobs=-1)
        results.append(cv_results)
        print(f"{scoring[0]}: {round(cv_results['test_r2'].mean()*-1, 4)} ({name}) ")
        print(f"{scoring[1]}: {round(cv_results['test_neg_mean_absolute_error'].mean()* -1, 4)} ({name}) ")
        print(f"{scoring[2]}: {round(cv_results['test_neg_root_mean_squared_error'].mean()* -1, 4)} ({name}) ")
    return results


def hyperparameter_optimization(X, y, model, cv=3, scoring="roc_auc"):
    """
    Verilen model bilgileri ile hiperparametre optimizasyonu yapar.
    :param X: Eğitim için kullanılacak veriseti
    :param y: Eğitim için kullanılacak hedef değişkeni
    :param model: (model adı, model objesi, denenecek hiper parametre sözlüğü) bilgilerini içeren liste
    :param cv: Kfold değeri
    :param scoring: Skorlamanın tutulacağı metrikler
    :return: En iyi sonuç alan final model objesini parametreleri ile döndürür
    """
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in model:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"{scoring[0]}: {round(cv_results['test_r2'].mean() * -1, 4)} ({name}) ")
        print(f"{scoring[1]}: {round(cv_results['test_neg_mean_absolute_error'].mean() * -1, 4)} ({name}) ")
        print(f"{scoring[2]}: {round(cv_results['test_neg_root_mean_squared_error'].mean() * -1, 4)} ({name}) ")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=2).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"{scoring[0]} (After): {round(cv_results['test_r2'].mean() * -1, 4)} ({name}) ")
        print(f"{scoring[1]} (After): {round(cv_results['test_neg_mean_absolute_error'].mean() * -1, 4)} ({name}) ")
        print(f"{scoring[2]} (After): {round(cv_results['test_neg_root_mean_squared_error'].mean() * -1, 4)} ({name}) ")
        best_models[name] = final_model
    return final_model




if __name__ == '__main__':
    df = reading_data("business.csv", "economy.csv")
    ################################################
    # 1. Genel Resim
    ################################################
    check_df(df, head=5)
    ################################################
    # 2. Veri Düzenleme
    ################################################
    df = preprocessing(df)
    ##################################
    # 3. Veri Analizi
    ##################################
    df = analysis(df)
    ##################
    # 4. Veri Çevirme One-Hot Encoding
    ##################
    df = encoding(df)
    check_df(df)
    ######################################################
    # 5. Base Models
    ######################################################
    y = df["price"]
    X = df.drop(["price", "date"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)
    train_results = base_models(X_train, y_train, scoring=["r2", "neg_mean_absolute_error","neg_root_mean_squared_error"])
    print(pd.DataFrame(train_results))

    ######################################################
    # 6. Hiper parametre Optimizasyonu
    ######################################################
    xgboost_params = {"learning_rate": [0.1, 0.5, 0.01, 0.05],
                      "max_depth": [5, 7, 10, 12, 15],
                      "n_estimators": [50, 100, 150, 200]}

    best_models = hyperparameter_optimization(X_train, y_train,
                                              [('XGBoost',
                                               XGBRegressor(use_label_encoder=False,
                                                            eval_metric='logloss',
                                                            n_jobs=-1),
                                               xgboost_params)],
                                              scoring=["r2", "neg_mean_absolute_error",
                                                       "neg_root_mean_squared_error"])

    rf_params = {'max_depth': [10, 20, None],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2, 5, 10],
                 'n_estimators': [50, 100, 150]}

    best_models = hyperparameter_optimization(X_train, y_train,
                                              [('RF',
                                                RandomForestRegressor( n_jobs=-1),
                                                rf_params)],
                                              scoring=["r2", "neg_mean_absolute_error",
                                                       "neg_root_mean_squared_error"])
    best_models = best_models.fit(X_train, y_train)
    #import pickle
    #pickle.dump(best_models, open("rf_model.pkl", "wb"))
    ######################################################
    # 7. Test Sonuçları
    ######################################################
    print(r2_score(y_test, best_models.predict(X_test)))
    print(mean_absolute_error(y_test, best_models.predict(X_test)))
    print(np.sqrt(mean_squared_error(y_test, best_models.predict(X_test))))
