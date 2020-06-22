import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender, TFIDFRecommender, BM25Recommender, \
    ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

# Ранжирующая модель
from lightgbm import LGBMClassifier


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # Создаем топ покупок
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        # Матрица user-item
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        # Взвешивание
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

            # Обучение и рекомендации
        self.model = self.fit(self.user_item_matrix)
        # self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        # self.cosin_recommender = self.fit_cosin_recommender(self.user_item_matrix)
        # self.tfidf_recommender = self.fit_tfidf_recommender(self.user_item_matrix)
        # self.tfidf100_recommender = self.fit_tfidf100_recommender(self.user_item_matrix)
        self.bm25_recommender = self.fit_bm25_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='sales_value',  # Можно пробоват другие варианты
                                          aggfunc='sum',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_cosin_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        cosin_recommender = CosineRecommender(K=2, num_threads=0)
        cosin_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return cosin_recommender

    @staticmethod
    def fit_tfidf_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        tfidf_recommender = TFIDFRecommender(K=6, num_threads=0)
        tfidf_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return tfidf_recommender

    @staticmethod
    def fit_tfidf100_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        tfidf100_recommender = TFIDFRecommender(K=100, num_threads=0)
        tfidf100_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return tfidf100_recommender

    @staticmethod
    def fit_bm25_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        bm25_recommender = BM25Recommender(K=6, K1=1.2, B=.76, num_threads=0)
        bm25_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return bm25_recommender

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=0)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.1, iterations=40, num_threads=0):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новый user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        top_rec = recs[1][0]
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        recommendations.extend(self.overall_top_purchases[:N])

        unique_recommendations = []
        [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]

        unique_recommendations = unique_recommendations[:N]

        # if len(recommendations) < N:
        #    recommendations.extend(self.overall_top_purchases[:N])
        #    recommendations = recommendations[:N]

        return unique_recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        try:
            res = [self.id_to_itemid[rec[0]] for rec in
                   model.recommend(userid=self.userid_to_id[user],
                                   user_items=csr_matrix(self.user_item_matrix).tocsr(),  # на вход user-item matrix
                                   N=N,
                                   filter_already_liked_items=False,
                                   filter_items=[self.itemid_to_id[999999]],
                                   recalculate_user=True)]
            res = self._extend_with_top_popular(res, N=N)

        except:
            res = self.overall_top_purchases[:N]

        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_cosin_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.cosin_recommender, N=N)

    def get_tfidf_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.tfidf_recommender, N=N)

    def get_tfidf100_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.tfidf100_recommender, N=N)

    def get_bm25_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.bm25_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self.get_similar_item(x)).tolist()

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        print(user)
        res = []

        self._update_dict(user_id=user)
        # Находим топ-N похожих пользователей
        try:
            similar_users = self.model.similar_users(self.userid_to_id[user], N=N + 1)
            similar_users = [rec_usr[0] for rec_usr in similar_users]
            similar_users = similar_users[1:]

            for usr in similar_users:
                res.extend(self.get_own_recommendations(self.id_to_userid[usr], N=1))

            res = self._extend_with_top_popular(res, N=N)
        except:
            res = self.overall_top_purchases[:N]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res


class Ranking:
    """Ранжирование кандидатов с помощью LGBM
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    X_train: pd.DataFrame
        Датафрейм с фичами для обучения модели второго уровня
    y_train: pd.Series
        Целевая переменная
    cat_feats: list
        Список категориальных фичей
    """

    def __init__(self, X_train, y_train, cat_feats):
        self.model = self.model_fit(X_train, y_train, cat_feats)

    def model_fit(self, X_train, y_train, cat_feats):
        """Обучение модели"""
        X_train[cat_feats] = X_train[cat_feats].astype('category')

        lgb = LGBMClassifier(objective='binary', max_depth=11, n_estimators=1005, categorical_column=cat_feats,
                             random_state=42)

        lgb.fit(X_train, y_train)

        return lgb

    def make_predict(self, data, X_train):
        """Прогноз модели"""
        preds = self.model.predict_proba(X_train)

        # Берем вероятность класса 1
        proba_1 = [i[1] for i in preds]

        result = data[['user_id', 'item_id']]
        result['proba'] = proba_1
        result.sort_values(['user_id', 'proba'], ascending=False, inplace=True)
        result = result.groupby('user_id').head(20)
        # Соберем результаты в строку
        result = result.groupby('user_id')['item_id'].unique().reset_index()
        result.columns = ['user_id', 'lgbm_rec']

        return result
