"""Чтение и проверка данных второго задания (SOC Model Inference)."""
from pathlib import Path
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent / 'data' / 'MODEL_PREDICTIONS_target.parquet'
COLUMNS = [
    'incident_id', 'pred_id', 'Дата', 'Flag_ValidLabel', 'Label_Status',
    'y', 'y_pred', 'true_incident_proba', 'confusion', 'Критичность',
    'version', 'latency_s'
]
CONFUSION = ('TN', 'FP', 'FN', 'TP')


def load_data(path=DATA_FILE):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Не найден файл {path}. Поместите его в папку data/.')
    df = pd.read_parquet(path, columns=COLUMNS)
    if df['pred_id'].duplicated().any():
        raise ValueError('Встречены повторяющиеся pred_id. Требуется отдельная проверка данных.')
    df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce', utc=True)
    df['true_incident_proba'] = pd.to_numeric(df['true_incident_proba'], errors='coerce')
    df['Критичность'] = df['Критичность'].fillna('Не указана').astype(str)
    df['Label_Status'] = df['Label_Status'].fillna('Неизвестно').astype(str)
    is_valid = df['Flag_ValidLabel'].eq(1) & df['y'].isin([0, 1]) & df['y_pred'].isin([0, 1])
    df['valid_for_metrics'] = is_valid
    expected = pd.Series('', index=df.index, dtype='object')
    expected.loc[is_valid & df.y.eq(0) & df.y_pred.eq(0)] = 'TN'
    expected.loc[is_valid & df.y.eq(0) & df.y_pred.eq(1)] = 'FP'
    expected.loc[is_valid & df.y.eq(1) & df.y_pred.eq(0)] = 'FN'
    expected.loc[is_valid & df.y.eq(1) & df.y_pred.eq(1)] = 'TP'
    bad = (df['confusion'].fillna('').astype(str).str.strip() != expected)
    if bad.any():
        raise ValueError(f'Несовпадения confusion с y/y_pred: {int(bad.sum())}')
    df['error_type'] = expected
    return df


def filter_data(df, criticality='Все', label_status='Все', date_from='', date_to=''):
    result = df
    if criticality != 'Все':
        result = result[result['Критичность'].eq(criticality)]
    if label_status != 'Все':
        result = result[result['Label_Status'].eq(label_status)]
    if date_from:
        start = pd.to_datetime(date_from, utc=True, errors='coerce')
        if pd.notna(start):
            result = result[result['Дата'] >= start]
    if date_to:
        end = pd.to_datetime(date_to, utc=True, errors='coerce')
        if pd.notna(end):
            result = result[result['Дата'] < end + pd.Timedelta(days=1)]
    return result


def metrics(data):
    labelled = data.loc[data['valid_for_metrics']]
    c = labelled['error_type'].value_counts()
    tp, tn, fp, fn = (int(c.get(s, 0)) for s in ('TP', 'TN', 'FP', 'FN'))
    def ratio(a, b):
        return (100 * a / b) if b else None
    return {
        'all_n': len(data), 'valid_n': len(labelled), 'tp': tp, 'tn': tn,
        'fp': fp, 'fn': fn,
        'precision': ratio(tp, tp + fp),
        'recall': ratio(tp, tp + fn),
        'fnr': ratio(fn, tp + fn),
        'accuracy': ratio(tp + tn, len(labelled)),
        'fpr': ratio(fp, fp + tn),
    }
