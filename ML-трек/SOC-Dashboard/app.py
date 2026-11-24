from pathlib import Path

import pandas as pd
import plotly.express as px
from taipy.gui import Gui

DATA = Path(__file__).resolve().parent / 'data' / 'iavd_incidents_raw_target.parquet'
if not DATA.exists():
    raise FileNotFoundError(f'Поместите датасет в {DATA}')

COLUMNS = ['EntityId', 'Создан', 'Критичность', 'Вердикт', 'Наименование']
df = pd.read_parquet(DATA, columns=COLUMNS)
df['Создан'] = pd.to_datetime(df['Создан'], utc=True)
df['Дата'] = df['Создан'].dt.date
df['Час'] = df['Создан'].dt.hour
df['Наименование'] = df['Наименование'].fillna('Не указано').astype(str)

criticality_options = ['Все'] + sorted(df['Критичность'].dropna().unique().tolist())
verdict_options = ['Все'] + sorted(df['Вердикт'].dropna().unique().tolist())
criticality = 'Все'
verdict = 'Все'
start_date = str(df['Дата'].min())
end_date = str(df['Дата'].max())


def compute(selected_criticality, selected_verdict, date_from, date_to):
    subset = df
    if selected_criticality != 'Все':
        subset = subset[subset['Критичность'] == selected_criticality]
    if selected_verdict != 'Все':
        subset = subset[subset['Вердикт'] == selected_verdict]
    try:
        start = pd.to_datetime(date_from).date()
        end = pd.to_datetime(date_to).date()
        subset = subset[(subset['Дата'] >= start) & (subset['Дата'] <= end)]
    except (ValueError, TypeError):
        pass

    n = len(subset)
    fp_n = int((subset['Вердикт'] == 'False Positive').sum())
    fp_pct = 100 * fp_n / n if n else 0

    verdict_counts = subset['Вердикт'].value_counts().rename_axis('Вердикт').reset_index(name='Карточек')
    f1 = px.bar(verdict_counts, x='Карточек', y='Вердикт', orientation='h', title='1. Распределение вердиктов')
    f1.update_layout(yaxis={'categoryorder': 'total ascending'})

    crit = subset.groupby('Критичность', dropna=False).agg(
        Всего=('EntityId', 'size'),
        FP=('Вердикт', lambda x: int((x == 'False Positive').sum()))
    ).reset_index()
    crit['Доля FP, %'] = (100 * crit['FP'] / crit['Всего']).fillna(0)
    f2 = px.bar(crit, x='Критичность', y='Доля FP, %', title='2. Доля FP по критичности')
    f2.update_yaxes(range=[0, 100])

    hourly = subset.groupby('Час').size().reindex(range(24), fill_value=0).reset_index(name='Карточек')
    f3 = px.bar(hourly, x='Час', y='Карточек', title='3. Создание карточек по часам (UTC)')

    rule_counts = subset['Наименование'].value_counts().head(10).rename_axis('Правило').reset_index(name='Карточек')
    f4 = px.bar(rule_counts, x='Карточек', y='Правило', orientation='h', title='4. Топ-10 правил по числу карточек')
    f4.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=30))

    stats = subset.groupby('Наименование').agg(
        Всего=('EntityId', 'size'),
        FP=('Вердикт', lambda x: int((x == 'False Positive').sum()))
    )
    stats = stats[stats['Всего'] >= 100].copy()
    stats['Доля FP, %'] = 100 * stats['FP'] / stats['Всего']
    stats = stats.sort_values(['Доля FP, %', 'Всего'], ascending=[False, False]).head(10).reset_index()
    f5 = px.bar(stats, x='Доля FP, %', y='Наименование', orientation='h',
                title='5. Топ-10 правил по доле FP (от 100 карточек)')
    f5.update_layout(yaxis={'categoryorder': 'total ascending'})
    f5.update_xaxes(range=[0, 100])
    for fig in (f1, f2, f3, f4, f5):
        fig.update_layout(
            height=350,
            margin=dict(l=18, r=24, t=58, b=45),
            font=dict(size=12),
            title_font=dict(size=17),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(160,170,185,0.15)')
    for fig in (f4, f5):
        fig.update_layout(height=460, margin=dict(l=30, r=28, t=58, b=46))
        fig.update_yaxes(tickfont=dict(size=10), automargin=True)
    return f'{n:,}'.replace(',', ' '), f'{fp_n:,}'.replace(',', ' '), f'{fp_pct:.2f}%', f1, f2, f3, f4, f5


total_text, fp_text, fp_percent, fig_verdict, fig_criticality, fig_hours, fig_rules, fig_rule_fp = compute(
    criticality, verdict, start_date, end_date
)


def on_change(state, var_name, var_value):
    if var_name not in {'criticality', 'verdict', 'start_date', 'end_date'}:
        return
    results = compute(state.criticality, state.verdict, state.start_date, state.end_date)
    (state.total_text, state.fp_text, state.fp_percent, state.fig_verdict,
     state.fig_criticality, state.fig_hours, state.fig_rules, state.fig_rule_fp) = results


page = '''
# SOC Dashboard

Мониторинг карточек инцидентов · Все временные значения — UTC

<|part|class_name=filter-panel|
### Фильтры
<|layout|columns=1 1 1 1|columns[mobile]=1|gap=1rem|
<|part|
**Критичность**
<|{criticality}|selector|lov={criticality_options}|dropdown|>
|>
<|part|
**Вердикт**
<|{verdict}|selector|lov={verdict_options}|dropdown|>
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
**Всего карточек**

### <|{total_text}|text|>
|>
<|part|class_name=kpi-card|
**False Positive**

### <|{fp_text}|text|>
|>
<|part|class_name=kpi-card|
**Доля FP**

### <|{fp_percent}|text|>
|>
|>

## Аналитика потока

<|layout|columns=1 1|columns[mobile]=1|gap=1rem|
<|part|class_name=chart-panel|
<|{fig_verdict}|chart|figure={fig_verdict}|>
|>
<|part|class_name=chart-panel|
<|{fig_criticality}|chart|figure={fig_criticality}|>
|>
|>

<|part|class_name=chart-panel|
<|{fig_hours}|chart|figure={fig_hours}|>
|>

## Правила корреляции SIEM

<|part|class_name=chart-panel|
<|{fig_rules}|chart|figure={fig_rules}|>
|>

<|part|class_name=chart-panel|
<|{fig_rule_fp}|chart|figure={fig_rule_fp}|>
|>

_В рейтинге по доле FP учитываются только правила, у которых после фильтрации осталось не менее 100 карточек._
'''

if __name__ == '__main__':
    Gui(page).run(title='SOC Dashboard', port=5000, use_reloader=False)
