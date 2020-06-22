import pandas as pd
import numpy as np


def prefilter_items(data, item_features, purchases_weeks=22, take_n_popular=5000):
    """Пред-фильтрация товаров

    Input
    -----
    data: pd.DataFrame
        Датафрейм с информацией о покупках
    item_features: pd.DataFrame
        Датафрейм с информацией о товарах
    """

    # Уберем товары с нулевыми продажами
    data = data[data['quantity'] != 0]

    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 5 месяцев
    purchases_last_week = data.groupby('item_id')['week_no'].max().reset_index()
    weeks = purchases_last_week[
        purchases_last_week['week_no'] > data['week_no'].max() - purchases_weeks].item_id.tolist()
    data = data[data['item_id'].isin(weeks)]

    # Уберем не интересные для рекоммендаций категории (department)
    department_size = pd.DataFrame(item_features.groupby('department')['item_id'].nunique(). \
                                   sort_values(ascending=False)).reset_index()

    department_size.columns = ['department', 'n_items']

    rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
    items_in_rare_departments = item_features[
        item_features['department'].isin(rare_departments)].item_id.unique().tolist()

    data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] >= 0.7]

    # Уберем слишком дорогие товары
    data = data[data['price'] < 50]

    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер не покупил товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    # ...

    return data


def postfilter(recommendations, item_info, N=5):
    """Пост-фильтрация товаров
    
    Input
    -----
    recommendations: list
        Ранжированный список item_id для рекомендаций
    item_info: pd.DataFrame
        Датафрейм с информацией о товарах
    """

    # Уникальность
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]

    # Разные категории
    categories_used = []
    final_recommendations = []

    CATEGORY_NAME = 'sub_commodity_desc'

    iterate = unique_recommendations.copy()
    for item in iterate:
        category = item_info.loc[item_info['item_id'] == item, CATEGORY_NAME].values[0]

        if category not in categories_used:
            final_recommendations.append(item)

            unique_recommendations.remove(item)
            categories_used.append(category)

    n_rec = len(final_recommendations)
    if n_rec < N:
        final_recommendations.extend(unique_recommendations[:N - n_rec])
    else:
        final_recommendations = final_recommendations[:N]

    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recommendations


def create_dataset(data, data_candidates, users_info, items_info):
    """Создание датасета с целевой переменной

    Input
    -----
    data: pd.DataFrame
        Датафрейм с информацией о покупках
    data_candidates: pd.DataFrame
        Датафрейм с информацией о кандидатах
    users_info: pd.DataFrame
        Датафрейм с информацией о пользователях
    items_info: pd.DataFrame
        Датафрейм с информацией о товарах
    """
    # Разворачиваем списки кандидатов для каждого юзера в столбец
    s = data_candidates.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'item_id'

    data_candidates = data_candidates.drop('candidates', axis=1).join(s)

    # Формируем датасет с целевой переменной
    targets = data[['user_id', 'item_id']].copy()
    targets['target'] = 1  # тут только покупки 

    targets = data_candidates.merge(targets, on=['user_id', 'item_id'], how='left')

    targets['target'].fillna(0, inplace=True)

    # Удаляем появившиеся дубликаты
    targets = targets.drop_duplicates(keep='first')

    # Добавляем фичи юзеров и товаров
    targets = targets.merge(items_info, on='item_id', how='left')
    targets = targets.merge(users_info, on='user_id', how='left')

    targets.fillna('Unknown', inplace=True)

    return targets


def dataset_processing(dataset, data, items_info):
    """Создание датасета с целевой переменной

    Input
    -----
    dataset: pd.DataFrame
        Датафрейм для обучения модели второго уровня
    data: pd.DataFrame
        Датафрейм с информацией о покупках
    items_info: pd.DataFrame
        Датафрейм с информацией о товарах
    """

    # Переведем "brand" из категориальной фичи в числовую
    dataset['brand'] = dataset['brand'].map({'National': 1, 'Private': 0})

    # Переведем "marital_status_code" из категориальной фичи в числовую
    dataset['marital_status_code'] = dataset['marital_status_code'].map({'A': 1, 'U': 2, 'B': 3})
    dataset['marital_status_code'].fillna(dataset['marital_status_code'].median(), inplace=True)

    # Переведем "age_desc" из категориальной фичи в числовую
    dataset['age_desc'] = dataset['age_desc']. \
        map({'19-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6, 'Unknown': 7})

    # Средний чек
    mean_check = data.groupby(['user_id', 'basket_id'])['sales_value'].sum().reset_index()
    mean_check = mean_check.groupby('user_id')['sales_value'].mean().reset_index()
    mean_check.rename(columns={'sales_value': 'mean_check'}, inplace=True)

    # Кол-во покупок в каждой категории
    num_purchases_in_category = data.merge(items_info, on='item_id', how='left')
    num_purchases_in_category.loc[num_purchases_in_category['department'] == ' ', 'department'] = 'Unknown'
    num_purchases_in_category = num_purchases_in_category.groupby(['user_id', 'department'])[
        'quantity'].count().reset_index()
    num_purchases_in_category.rename(columns={'quantity': 'num_purchases_in_category'}, inplace=True)

    # Цена
    price = data.groupby('item_id')['sales_value'].sum() / data.groupby('item_id')['quantity'].sum()
    price = price.groupby('item_id').mean().reset_index()
    price.columns = ['item_id', 'price']
    price['price'].fillna(0, inplace=True)

    # (Средняя сумма покупки 1 товара в каждой категории (берем категорию item_id)) - (Цена item_id)
    mean_price_in_category = data.merge(items_info, on='item_id', how='left')
    mean_price_in_category.loc[mean_price_in_category['department'] == ' ', 'department'] = 'Unknown'
    department_quantity = mean_price_in_category.groupby('department')['quantity'].sum().reset_index()
    mean_price_in_category = mean_price_in_category.groupby('department')['sales_value'].sum().reset_index()
    mean_price_in_category['mean_price_in_category'] = mean_price_in_category['sales_value'] / department_quantity[
        'quantity']
    mean_price_in_category.drop(['sales_value'], axis=1, inplace=True)

    dataset = dataset.merge(mean_check, on='user_id', how='left')
    dataset = dataset.merge(num_purchases_in_category, on=['user_id', 'department'], how='left')
    dataset = dataset.merge(price, on='item_id', how='left')
    dataset = dataset.merge(mean_price_in_category, on='department', how='left')

    dataset['num_purchases_in_category'].fillna(dataset['num_purchases_in_category'].median(), inplace=True)

    # Разница  между средней ценой в категории и ценой товара
    dataset['dif_price'] = dataset['mean_price_in_category'] - dataset['price']

    return dataset
