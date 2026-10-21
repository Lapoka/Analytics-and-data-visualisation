import pandas as pd
import pyarrow.parquet as pq

path = "iavd_incidents_raw_target.parquet"

# Посмотреть структуру без загрузки всех данных
file = pq.ParquetFile(path)

print("Количество строк:", file.metadata.num_rows)
print("Количество столбцов:", len(file.schema.names))
print("Названия столбцов:", file.schema.names)

columns = [
    "EntityId",
    "Критичность",
    "Вердикт",
    "Создан"
]

df = pd.read_parquet(path, columns=columns)

print(df.head())
print(df.info())

# Сохраняем только необходимые данные
df.to_parquet(
    "soc_analysis.parquet",
    index=False
)

