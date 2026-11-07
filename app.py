import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


st.title('Подборка вин')
model = load_model()
test = load_test_data()
X_test = test.drop(columns='target')
y_test = test['target']
features = X_test.columns.tolist()
config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
threshold = st.sidebar.slider('Порог отбора', 0.0, 1.0, float(config['threshold']),
                              step=0.0001, format='%.4f')

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs(
    ['Оценка вина', 'Модельные метрики', 'Загрузка CSV', 'Что выучила модель']
)

with single_tab:
    values = {}
    for feature in features:
        if feature == 'is_red':
            values[feature] = int(st.toggle('Красное вино (выкл. — белое)',
                                            value=bool(X_test[feature].median())))
        else:
            values[feature] = st.number_input(
                feature,
                min_value=float(X_test[feature].min()),
                max_value=float(X_test[feature].max()),
                value=float(X_test[feature].median()),
                format='%.5f',
            )
    wine = pd.DataFrame([values], columns=features)
    probability = model.predict_proba(wine)[0, 1]
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.write('Решение:', 'включить в подборку' if probability >= threshold else 'не включать в подборку')

with metrics_tab:
    scores = test_proba(model, X_test)
    prediction = (scores >= threshold).astype(int)
    matrix = metrics.confusion_matrix(y_test, prediction, labels=[0, 1])
    tn, fp = matrix[0]
    precision = metrics.precision_score(y_test, prediction, zero_division=0)
    recall = metrics.recall_score(y_test, prediction, zero_division=0)
    f1 = metrics.f1_score(y_test, prediction, zero_division=0)
    st.metric('Precision', f'{precision:.3f}')
    st.metric('Recall', f'{recall:.3f}')
    st.metric('F1', f'{f1:.3f}')

    fig, ax = plt.subplots(figsize=(4, 3))
    metrics.ConfusionMatrixDisplay(matrix, display_labels=['остальные', 'хорошие']).plot(
        ax=ax, cmap='Blues', colorbar=False,
    )
    ax.set(xlabel='решение модели', ylabel='реальный класс', title='Матрица ошибок')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, scores)
    pr, rc, _ = metrics.precision_recall_curve(y_test, scores)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(8, 3))
    ax_roc.plot(fpr, tpr)
    ax_roc.scatter(fp / (fp + tn), recall, color='red')
    ax_roc.set(xlabel='FPR', ylabel='recall', title='ROC-кривая')
    ax_pr.plot(rc, pr)
    ax_pr.scatter(recall, precision if prediction.any() else 1, color='red')
    ax_pr.set(xlabel='recall', ylabel='precision', title='PR-кривая')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if not prediction.any():
        st.caption('Подборка пуста. На PR-кривой отмечена её условная конечная точка (0, 1).')
    st.write('Красная точка — текущий порог. Если повысить порог, в подборку попадёт меньше вин '
             'и часть хороших можно пропустить. Если понизить порог, хороших вин найдётся больше, '
             'но может стать больше лишних рекомендаций.')

with batch_tab:
    upload = st.file_uploader('CSV с признаками вин без столбца target', type='csv')
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            missing = [feature for feature in features if feature not in batch.columns]
            if missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(missing))
            else:
                batch_scores = model.predict_proba(batch[features])[:, 1]
                result = batch.assign(probability=batch_scores,
                                      selected=(batch_scores >= threshold).astype(int))
                st.dataframe(result)
                st.download_button('Скачать результаты', result.to_csv(index=False),
                                   file_name='wine_predictions.csv', mime='text/csv')
        except (ValueError, OSError):
            st.error('Не удалось обработать CSV. Признаки должны быть числами без пропусков.')

with weights_tab:
    classifier = model.named_steps['logisticregression']
    weights = pd.Series(classifier.coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(weights.index, weights)
    ax.set_xlabel('коэффициент после масштабирования')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(weights[weights > 0].tail(3).index[::-1]) + '.')
    st.write('Сильнее всего против: ' + ', '.join(weights[weights < 0].head(3).index) + '.')
