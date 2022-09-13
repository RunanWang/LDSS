import pandas as pd
import os
from pathlib import Path

file = Path("../../data/dmv/dmv.csv")
data = pd.read_csv(file)
df = pd.DataFrame(data)

update_path = Path("../../data/dmv/update")
if not os.path.exists(update_path):
    os.mkdir(update_path)

update_table_path = update_path / "table"
if not os.path.exists(update_table_path):
    os.mkdir(update_table_path)

append_table_path = update_path / "append"
if not os.path.exists(append_table_path):
    os.mkdir(append_table_path)

print(df['Reg_Valid_Date'].quantile(0.75))
df_all = df[df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(0.75)]
df_all = df_all.reset_index(drop=True)
file_path = update_table_path / "75.csv"
df_all.to_csv(file_path, index=False)

print(df['Reg_Valid_Date'].quantile(0.8))
df_all = df[df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(0.8)]
df_all = df_all.reset_index(drop=True)
file_path = update_table_path / "80.csv"
df_all.to_csv(file_path, index=False)
df_append = df[(df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(0.8)) & (df['Reg_Valid_Date']>df['Reg_Valid_Date'].quantile(0.75))]
df_append = df_append.reset_index(drop=True)
file_path = append_table_path / "80.csv"
df_append.to_csv(file_path, index=False)
print(df_append.size)

print(df['Reg_Valid_Date'].quantile(0.85))
df_all = df[df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(0.85)]
df_all = df_all.reset_index(drop=True)
file_path = update_table_path / "85.csv"
df_all.to_csv(file_path, index=False)
df_append = df[(df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(0.85)) & (df['Reg_Valid_Date']>df['Reg_Valid_Date'].quantile(0.75))]
df_append = df_append.reset_index(drop=True)
file_path = append_table_path / "85.csv"
df_append.to_csv(file_path, index=False)
print(df_append.size)

print(df['Reg_Valid_Date'].quantile(0.9))
df_all = df[df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(0.9)]
df_all = df_all.reset_index(drop=True)
file_path = update_table_path / "90.csv"
df_all.to_csv(file_path, index=False)
df_append = df[(df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(0.9)) & (df['Reg_Valid_Date']>df['Reg_Valid_Date'].quantile(0.75))]
df_append = df_append.reset_index(drop=True)
file_path = append_table_path / "90.csv"
df_append.to_csv(file_path, index=False)
print(df_append.size)

print(df['Reg_Valid_Date'].quantile(0.95))
df_all = df[df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(0.95)]
df_all = df_all.reset_index(drop=True)
file_path = update_table_path / "95.csv"
df_all.to_csv(file_path, index=False)
df_append = df[(df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(0.95)) & (df['Reg_Valid_Date']>df['Reg_Valid_Date'].quantile(0.75))]
df_append = df_append.reset_index(drop=True)
file_path = append_table_path / "95.csv"
df_append.to_csv(file_path, index=False)
print(df_append.size)

df_all = df
file_path = update_table_path / "100.csv"
df_all.to_csv(file_path, index=False)
df_append = df[(df['Reg_Valid_Date']<=df['Reg_Valid_Date'].quantile(1.0)) & (df['Reg_Valid_Date']>df['Reg_Valid_Date'].quantile(0.75))]
df_append = df_append.reset_index(drop=True)
file_path = append_table_path / "100.csv"
df_append.to_csv(file_path, index=False)
print(df_append.size)