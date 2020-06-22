import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import os, sys

module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

# Написанные нами функции
from src.utils import prefilter_items, postfilter, create_dataset, dataset_processing
from src.recommenders import MainRecommender, Ranking

data = pd.read_csv('../raw_data/retail_train.csv')
item_features = pd.read_csv('../raw_data/product.csv')
user_features = pd.read_csv('../raw_data/hh_demographic.csv')

# column processing
item_features.columns = [col.lower() for col in item_features.columns]
user_features.columns = [col.lower() for col in user_features.columns]

item_features.rename(columns={'product_id': 'item_id'}, inplace=True)
user_features.rename(columns={'household_key': 'user_id'}, inplace=True)

# data filtering
data = prefilter_items(data, item_features=item_features, take_n_popular=5000)

candidates = pd.DataFrame(data['user_id'].unique())
candidates = candidates.rename(columns={0: 'user_id'})

recommender = MainRecommender(data)

# Рекомендации по BM25 взвешиванию
candidates['candidates'] = candidates['user_id'].apply(lambda x: recommender.get_bm25_recommendations(x, N=100))

# Создадим датафрейм в целевой переменной и добавим фичи юзеров и товаров
targets = create_dataset(data=data, data_candidates=candidates, users_info=user_features, items_info=item_features)

# Сгенерируем новые фичи
targets = dataset_processing(dataset=targets, data=data, items_info=item_features)

# Разобьем даатсет
X_train = targets.drop('target', axis=1)
y_train = targets[['target']]

cat_feats = [
    'user_id',
    'item_id',
    'department',
    'commodity_desc',
    'sub_commodity_desc',
    'curr_size_of_product',
    'income_desc',
    'homeowner_desc',
    'hh_comp_desc',
    'household_size_desc',
    'kid_category_desc']

# Тренируем модель
ranger = Ranking(X_train, y_train, cat_feats)

# Делаем прогноз
predictions = ranger.make_predict(targets, X_train)

# Применяем постфильтер
predictions['lgbm_rec'] = predictions['lgbm_rec'].apply(lambda x: postfilter(x, item_info=item_features, N=5))

# Выгружаем результаты
predictions.to_csv('../predictions/Yuriy_Ryabinin_top5_recommendations.csv', sep='|', index=None)
