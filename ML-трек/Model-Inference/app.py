"""Taipy: оценка модели SOC по размеченным карточкам и срезам."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from taipy.gui import Gui
from data_processing import load_data, filter_data, metrics

DATA = load_data()
criticality_options = ['Все'] + sorted(DATA['Критичность'].unique().tolist())
label_options = ['Все'] + sorted(DATA['Label_Status'].unique().tolist())
criticality = 'Все'
label_status = 'Все'
start_date = str(DATA['Дата'].min().date())
end_date = str(DATA['Дата'].max().date())


def fmt_pct(value):
    return '—' if value is None else f'{value:.2f}%'


def theme(fig, height=390):
    fig.update_layout(
        height=height, font=dict(size=12), title_font=dict(size=16),
        margin=dict(l=34, r=28, t=64, b=52),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend_title_text=''
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(160,170,185,.18)', automargin=True)
    return fig


def build_dashboard(selected_criticality, selected_label, date_from, date_to):
    subset = filter_data(DATA, selected_criticality, selected_label, date_from, date_to)
    m = metrics(subset)
    valid = subset[subset['valid_for_metrics']].copy()

    counts = subset['y_pred'].map({0: 'Легитимно (0)', 1: 'Инцидент (1)'}).value_counts()
    predictions = pd.DataFrame({
        'Решение': ['Легитимно (0)', 'Инцидент (1)'],
        'Карточек': [int(counts.get('Легитимно (0)', 0)), int(counts.get('Инцидент (1)', 0))]
    })
    f_predictions = px.bar(predictions, x='Решение', y='Карточек', text='Карточек',
                           title='1. Решения модели (включая карточки без метки)')
    f_predictions.update_traces(textposition='outside', cliponaxis=False)

    matrix = [[m['tn'], m['fp']], [m['fn'], m['tp']]]
    f_confusion = go.Figure(go.Heatmap(
        z=matrix, x=['Модель: 0', 'Модель: 1'],
        y=['Факт: 0', 'Факт: 1'], text=matrix, texttemplate='%{text}',
        colorscale='Blues', showscale=False, hovertemplate='%{x} / %{y}: %{z}<extra></extra>'
    ))
    f_confusion.update_layout(title='2. Матрица ошибок (только валидные метки)')
    f_confusion.update_yaxes(autorange='reversed')

    if m['valid_n']:
        crit = pd.crosstab(valid['Критичность'], valid['error_type'])
        for c in ['TN', 'FP', 'FN', 'TP']:
            if c not in crit.columns:
                crit[c] = 0
        crit = crit.reset_index()
        crit['Истинных инцидентов'] = crit['FN'] + crit['TP']
        crit['Доля FN, %'] = 100 * crit['FN'] / crit['Истинных инцидентов'].replace(0, float('nan'))
    else:
        crit = pd.DataFrame(columns=['Критичность', 'FN', 'Доля FN, %', 'Истинных инцидентов'])
    f_severity = px.bar(crit, x='Критичность', y='FN',
                        hover_data=['Истинных инцидентов', 'Доля FN, %'],
                        title='3. Пропущенные инциденты по критичности (FN, шт.)')
    f_severity.update_yaxes(title='Количество FN')

    bounds = [0, .05, .10, .20, .40, .60, .80, 1.0000001]
    groups = ['<5%', '5–10%', '10–20%', '20–40%', '40–60%', '60–80%', '80–100%']
    valid['Интервал'] = pd.cut(valid['true_incident_proba'], bins=bounds,
                              labels=groups, include_lowest=True, right=False)
    errors = valid[valid['error_type'].isin(['FP', 'FN'])]
    err = pd.crosstab(errors['Интервал'], errors['error_type']).reindex(groups, fill_value=0)
    for c in ('FP', 'FN'):
        if c not in err:
            err[c] = 0
    err = err[['FN', 'FP']].rename_axis('Интервал').reset_index().melt(
        id_vars='Интервал', var_name='Тип ошибки', value_name='Карточек'
    )
    f_score = px.bar(err, x='Интервал', y='Карточек', color='Тип ошибки', barmode='group',
                     color_discrete_map={'FN': '#d35b52', 'FP': '#e6a344'},
                     title='4. Ошибки модели по оценке вероятности инцидента')

    metric_data = pd.DataFrame({
        'Метрика': ['Precision', 'Recall', 'Доля FN (FNR)', 'Accuracy'],
        'Значение, %': [m['precision'], m['recall'], m['fnr'], m['accuracy']]
    })
    f_metrics = px.bar(metric_data, x='Метрика', y='Значение, %',
                       title='5. Показатели качества на размеченном срезе',
                       text='Значение, %')
    f_metrics.update_traces(texttemplate='%{text:.1f}%', textposition='outside', cliponaxis=False)
    f_metrics.update_yaxes(range=[0, 105])

    charts = [f_predictions, f_confusion, f_severity, f_score, f_metrics]
    for fig in charts:
        theme(fig)
    f_confusion.update_layout(height=390)

    if not m['valid_n']:
        note = 'В выбранном срезе нет карточек с валидной бинарной меткой; метрики и матрица ошибок не определены.'
    else:
        note = ('Метрики качества рассчитаны только по размеченным карточкам.'
                'Неразмеченные и исключённые карточки учитываются только в распределении решений модели.')
    return (
        f"{m['all_n']:,}".replace(',', ' '), f"{m['valid_n']:,}".replace(',', ' '),
        str(m['fn']), fmt_pct(m['recall']), fmt_pct(m['precision']),
        '≈ 0,18 (18%)', note, *charts
    )


(total_text, labelled_text, fn_text, recall_text, precision_text, threshold_text,
 note_text, fig_predictions, fig_confusion, fig_severity, fig_score, fig_metrics) = build_dashboard(
    criticality, label_status, start_date, end_date
)


def on_change(state, var_name, var_value):
    if var_name not in {'criticality', 'label_status', 'start_date', 'end_date'}:
        return
    values = build_dashboard(state.criticality, state.label_status, state.start_date, state.end_date)
    (state.total_text, state.labelled_text, state.fn_text, state.recall_text,
     state.precision_text, state.threshold_text, state.note_text,
     state.fig_predictions, state.fig_confusion, state.fig_severity,
     state.fig_score, state.fig_metrics) = values


page = '''
# SOC · Оценка инференса ML-модели

Модель **wlarf-dec-jun-v3** · Время: UTC

<|part|class_name=filter-panel|
### Фильтры
<|layout|columns=1 1 1 1|columns[mobile]=1|gap=1rem|
<|part|
**Критичность**
<|{criticality}|selector|lov={criticality_options}|dropdown|>
|>
<|part|
**Статус разметки**
<|{label_status}|selector|lov={label_options}|dropdown|>
|>
<|part|
**Дата начала (ГГГГ-ММ-ДД)**
<|{start_date}|input|>
|>
<|part|
**Дата конца (ГГГГ-ММ-ДД)**
<|{end_date}|input|>
|>
|>
|>

<|layout|columns=1 1 1|columns[mobile]=1|gap=1rem|
<|part|class_name=kpi-card|
**Всего предсказаний**
### <|{total_text}|text|>
|>
<|part|class_name=kpi-card|
**С валидной меткой**
### <|{labelled_text}|text|>
|>
<|part|class_name=kpi-card|
**Пропущенные инциденты (FN)**
### <|{fn_text}|text|>
|>
|>
<|layout|columns=1 1 1|columns[mobile]=1|gap=1rem|
<|part|class_name=kpi-card|
**Recall**
### <|{recall_text}|text|>
|>
<|part|class_name=kpi-card|
**Precision**
### <|{precision_text}|text|>
|>
<|part|class_name=kpi-card|
**Рабочий порог (по данным)**
### <|{threshold_text}|text|>
|>
|>

<|part|class_name=info-panel|
<|{note_text}|text|>
|>

## Оценка модели

<|layout|columns=1 1|columns[mobile]=1|gap=1rem|
<|part|class_name=chart-panel|
<|{fig_predictions}|chart|figure={fig_predictions}|>
|>
<|part|class_name=chart-panel|
<|{fig_confusion}|chart|figure={fig_confusion}|>
|>
|>

<|part|class_name=chart-panel|
<|{fig_severity}|chart|figure={fig_severity}|>
|>

<|layout|columns=1 1|columns[mobile]=1|gap=1rem|
<|part|class_name=chart-panel|
<|{fig_score}|chart|figure={fig_score}|>
|>
<|part|class_name=chart-panel|
<|{fig_metrics}|chart|figure={fig_metrics}|>
|>
|>

_Не трактуйте ошибки на отдельных малочисленных категориях как устойчивые закономерности. Порог 0,18 восстановлен из сохранённых предсказаний._
'''

if __name__ == '__main__':
    Gui(page).run(title='SOC · Model Inference', port=5001, use_reloader=False)
