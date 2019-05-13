import ast
import json
import re

from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize
from pandas.io.json import json_normalize
from sklearn import model_selection
from sklearn.externals._arff import xrange
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder, StandardScaler


def json_to_name_str(data):
    # data = str(data).replace('\'', '\"')
    # data = str(data).replace('None', '\"\"')
    j1 = ast.literal_eval(data)
    # j1:dict = json.loads(str(data))
    str_array:list = []
    for item in j1:
        str_array.append(item['name'])
    return str_array


def json_to_name_str1(data):
    j1 = ast.literal_eval(data)
    str_array:list = []
    for item in j1:
        str_array.append(item['id'])
        if len(str_array) == 3:
            return str_array
    return str_array
    # return str_array
    # if len(j1) > 0:
    #     return j1[0]['id']
    # else:
    #     return None


cast = pd.read_csv('../data/credits.csv', converters={'crew': json_to_name_str1, 'cast': json_to_name_str1},  low_memory=False)
data = pd.read_csv('../data/movies_metadata.csv', converters={'genres': json_to_name_str}, low_memory=False)
# cast.head()
data = data.drop(columns=['adult', 'belongs_to_collection', 'homepage', 'imdb_id','poster_path', 'production_companies', 'production_countries', 'spoken_languages', 'status'])
print(data.columns)

new_cast: dict = dict()
new_cast['cast_1'] = cast.cast.str[0]
new_cast['cast_2'] = cast.cast.str[1]
new_cast['cast_3'] = cast.cast.str[2]

new_cast['crew_1'] = cast.crew.str[0]
new_cast['crew_2'] = cast.crew.str[1]
new_cast['crew_3'] = cast.crew.str[2]

# print(new_cast)
# data['crew'] = cast['crew']
# data['cast'] = cast['cast']

data = data.join(pd.DataFrame(new_cast))
# print(data.head())
# data.info()

# составляем словарь
# def WordsDic(dataset):
#     WD = []
#     for i in dataset.index:
#         for j in xrange(len(dataset[i])):
#             if dataset[i][j] in WD:
#                 None
#             else:
#                 WD.append(dataset[i][j])
#     return WD

# def splitstring(str):
#     words = []
#     #разбиваем строку по символам из []
#     for i in re.split('[;,.,\n,\s,:,-,+,(,),=,/,«,»,@,\d,!,?,"]',str):
#         #берём только "слова" длиной 2 и более символов
#         if len(i) > 1:
#             #не берем слова-паразиты
#             if i in garbage_list:
#                 None
#             else:
#                 words.append(i)
#     return words


#объявляем функцию приведения строки к нижнему регистру
def lower(str):
    return str.lower()


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# print(missing_values_table(data))

missing_df = missing_values_table(data)
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))

data = data.drop(columns=list(missing_columns))
print(missing_values_table(data))

types = data.dropna(subset=['vote_average'])
types = types['original_language'].value_counts()
types = list(types[types.values > 100].index)

# get data where language is not none
data = data[pd.notnull(data['overview'])]
data = data[pd.notnull(data['original_language'])]
data = data[pd.notnull(data['runtime'])]
data = data[data['genres'].map(len) != 0]
data = data[pd.notnull(data['crew_1'])]
data = data[pd.notnull(data['cast_1'])]
data = data[pd.notnull(data['crew_2'])]
data = data[pd.notnull(data['cast_2'])]
data = data[pd.notnull(data['crew_3'])]
data = data[pd.notnull(data['cast_3'])]
# data = data[data['crew'].map(len) != 0]
# data = data[data['cast'].map(len) != 0]
# data = pd.concat([pd.DataFrame(json_normalize(x)) for x in data['genres']], ignore_index=True)
# print(data)

# print(data['genres'])

# string to binary
le = LabelEncoder()
le.fit(data['original_language'].astype(str))
data['language'] = le.transform(data['original_language'])

mlb = MultiLabelBinarizer()
mlb.fit(data['genres'])
# print(mlb.classes_)
# data['genres'] = mlb.fit_transform(data['genres'])
data = data.join(pd.DataFrame(mlb.fit_transform(data['genres']), columns=mlb.classes_, index=data.index))
print('КОЛОНКИ')
print(data.columns)
# print(data['genres'])

# mlb = MultiLabelBinarizer()
# mlb.fit(data['crew'])
# print(len(mlb.classes_))
# # data['crew1'] = mlb.transform(data['crew'])
# # data['crew'] = mlb.fit_transform(data['crew'])
# data = data.join(pd.DataFrame(mlb.fit_transform(data['crew']), columns=mlb.classes_, index=data.index), lsuffix='_left', rsuffix='_right')

# mlb = MultiLabelBinarizer()
# mlb.fit(data['cast'])
# print(len(mlb.classes_))
# # mlb.transform(data['cast'])
# # data['cast'] = mlb.fit_transform(data['cast'])
# data = data.join(pd.DataFrame(mlb.fit_transform(data['cast']), columns=mlb.classes_, index=data.index), lsuffix='_left', rsuffix='_right')

# ОЧЕНЬ ДОЛГО
# # применяем функцию к каждой ячейке столбца Content
# data['overview'] = data.overview.apply(lower)
# garbage_list = [u'this', u'it', u'and', u'on', u'an', u'a']
# # составляем словарь
# data['Words'] = data.overview.apply(splitstring)
# # тестовая выборка в 30:
# issues_train, issues_test, labels_train, labels_test = model_selection.train_test_split(data.Words,
#                                                                                         data.language,
#                                                                                         test_size=0.3)
# # применяем функцию к данным
# words = WordsDic(issues_train)
# print('Words length ',  len(words))
#
# # объявляем матрицу размером len(issues_train) на len(words), заполняем её нулями
# train_matrix = np.zeros((len(issues_train),len(words)))
# # заполняем матрицу, проставляя в [i][j] количество вхождений j-го слова из words в i-й объект обучающей выборки
# for i in xrange(train_matrix.shape[0]):
#     for j in issues_train[issues_train.index[i]]:
#         if j in words:
#             train_matrix[i][words.index(j)]+=1


# Plot of distribution of scores for building categories
figsize(12, 10)

# Plot each building
for b_type in types:
    # Select the building type
    subset = data[data['original_language'] == b_type]

    # Density plot of Energy Star scores
    sns.kdeplot(subset['vote_average'].dropna(),
                label=b_type, shade=False, alpha=0.8)

# label the plot
plt.xlabel('Energy Star Score', size=12)
plt.ylabel('Density', size=20)
plt.title('Density Plot of Energy Star Scores by Language', size=28)
# plt.show()


# Find all correlations and sort
correlations_data = data.corr()['vote_average'].sort_values()

# Print the most negative correlations
print(correlations_data.head(15), '\n')

# Print the most positive correlations
print(correlations_data.tail(15))
data = data.drop(['genres', 'original_language', 'original_title', 'overview', 'title', 'release_date', 'video', 'id', 'popularity', 'budget'], axis=1)


X = data.drop('vote_average', axis=1)
y = data['vote_average']
x_train, x_test, y_train, y_test = train_test_split(X, y)

# Среднее значение
mean = x_train.mean(axis=0)
# Стандартное отклонение
std = x_train.std(axis=0)
print((y_test.to_list())[0])
print(data.dtypes)
print(mean, std)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)
mse, mae = model.evaluate(x_test, y_test, verbose=0)
pred = model.predict(x_test)
print("Средняя абсолютная ошибка :", mae)
print("Предсказанная оценка:", pred[1][0], ", правильная оценка:", (y_test.to_list())[0])



# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# mlp = MLPRegressor(hidden_layer_sizes=(15,15,15),max_iter=500)
# mlp.fit(X_train,y_train)
#
# predictions = mlp.predict(X_test)
# print(mlp.score(X_test, y_test))
