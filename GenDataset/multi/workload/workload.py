from collections import OrderedDict
from typing import Dict, NamedTuple, Optional, Tuple, List, Any
import pickle
import pandas as pd

from constants import PKL_PROTO, TEMP_ROOT
from GenDataset.multi.dataset.table import Database
from GenDataset.multi.dataset.data import list_to_str


class Query(NamedTuple):
    # [{table1: join_column, ...}]
    join: List[Tuple[str, str]]
    # {table1: {column1: {[left, right], ...}, ...}
    predicates: Dict[str, Dict[str, Optional[Tuple[str, Any]]]]


class Label(NamedTuple):
    cardinality: int
    selectivity: float


def new_query(database: Database, used_table: List[str]) -> Query:
    join = []
    predicates = {}
    for t in used_table:
        join.append((t, database.join_col[t]))
        predicates[t] = OrderedDict.fromkeys(database.table[t].data.columns, None)
    return Query(join=join, predicates=predicates)


def gen_row_num(query: Query, database: Database):
    num = 1
    table_num = 0
    for table, col in query.join:
        num *= database.table[table].row_num
        table_num += 1
    return table_num, num


def query_2_sql(query: Query, table_dict, aggregate=True, split=False, dbms='postgres'):
    preds = []
    for t in query.predicates.keys():
        for col, pred in query.predicates[t].items():
            if pred is None:
                continue
            op, val = pred
            if op == '[]':
                if split:
                    preds.append(f"{table_dict[t]}.{col} >= {val[0]}")
                    preds.append(f"{table_dict[t]}.{col} <= {val[1]}")
                else:
                    preds.append(f"({table_dict[t]}.{col} between {val[0]} and {val[1]})")
            else:
                preds.append(f"{table_dict[t]}.{col} {op} {val}")
    join_condition = []
    join_table, join_col = query.join[0]
    i = 1
    while i < len(query.join):
        join_table2, join_col2 = query.join[i]
        join_condition.append(f"{table_dict[join_table]}.{join_col} = {table_dict[join_table2]}.{join_col2}")
        i += 1
    preds = join_condition + preds
    table_avatar_list = []
    for t in list(query.predicates.keys()):
        table_avatar_list.append(t + " " + table_dict[t])
    if dbms == 'mysql':
        return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM `{', '.join(table_avatar_list)}` WHERE {' AND '.join(preds)}"
    return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM {', '.join(table_avatar_list)} WHERE {' AND '.join(preds)}"


def sql_2_query(sql, database):
    from_index = sql.find("FROM")
    where_index = sql.find("WHERE")
    sql_tables = sql[from_index + 5:where_index]
    used_tables = []
    avatar_2_table = {"mc": "movie_companies", "t": "title", "mi_idx": "movie_info_idx",
                      "mi": "movie_info", "mk": "movie_keyword", "ci": "cast_info"}
    table_2_avatar = {"movie_companies": "mc", "title": "t", "movie_info_idx": "mi_idx",
                      "movie_info": "mi", "movie_keyword": "mk", "cast_info": "ci"}
    cond_ops = [">", "<", "="]
    sql_conditions = sql[where_index + 6:-2].split("AND")
    for t in database.table_list:
        t_avatar = f"{t} {table_2_avatar[t]}"
        if t_avatar in sql_tables:
            used_tables.append(t)
    query = new_query(database, used_tables)
    for cond in sql_conditions:
        # determine operation
        cond_op = ""
        for op in cond_ops:
            if op in cond:
                cond_op = op
                break
        cond_split = cond.split(cond_op)
        cond_front = cond_split[0].replace(" ", "")
        cond_back = cond_split[1].replace(" ", "")
        # exclude table connections
        table_conn = False
        for avatar in avatar_2_table.keys():
            if avatar + "." in cond_back:
                table_conn = True
        if not table_conn:
            t = avatar_2_table[cond_front.split(".")[0]]
            a = cond_front.split(".")[1]
            if query.predicates[t][a] is None:
                query.predicates[t][a] = cond_op, int(cond_back)
            else:
                _, v1 = query.predicates[t][a]
                v2 = int(cond_back)
                query.predicates[t][a] = "[]", (min(v1, v2), max(v1, v2))
    return query


def query_2_vec(query: Query, database: Database):
    vec = {}
    # predicates
    for t in query.predicates.keys():
        for col, pred in query.predicates[t].items():
            vec[f"qf_{t}_{col}_ql"] = 0.0
            vec[f"qf_{t}_{col}_qr"] = 1.0
            if pred is None:
                vec[f"uc_{t}_{col}"] = 0
                continue
            vec[f"uc_{t}_{col}"] = 1
            op, val = pred
            if op == '[]':
                vec[f"qf_{t}_{col}_ql"] = database.table[t].columns[col].normalize([val[0]]).item()
                vec[f"qf_{t}_{col}_qr"] = database.table[t].columns[col].normalize([val[1]]).item()
            elif op == '>=' or op == '>':
                vec[f"qf_{t}_{col}_ql"] = database.table[t].columns[col].normalize([val]).item()
            elif op == '<=' or op == '<':
                vec[f"qf_{t}_{col}_qr"] = database.table[t].columns[col].normalize([val]).item()
            elif op == '=':
                vec[f"qf_{t}_{col}_ql"] = database.table[t].columns[col].normalize([val]).item()
                vec[f"qf_{t}_{col}_qr"] = database.table[t].columns[col].normalize([val]).item()
            else:
                raise NotImplementedError
    # joins
    for join_table, join_col in database.join_col.items():
        if (join_table, join_col) in query.join:
            vec[f"ut_{join_table}_{join_col}"] = 1
        else:
            vec[f"ut_{join_table}_{join_col}"] = 0
    return vec


def query_2_triple(query: Query, table_name: str, with_none: bool = True, split_range: bool = False
                   ) -> Tuple[List[str], List[str], List[Any]]:
    """return 3 lists with same length: cols(columns names), ops(predicate operators), vals(predicate literals)"""
    cols = []
    ops = []
    vals = []
    for c, p in query.predicates[table_name].items():
        if p is not None:
            if split_range is True and p[0] == '[]':
                cols.append(c)
                ops.append('>=')
                vals.append(p[1][0])
                cols.append(c)
                ops.append('<=')
                vals.append(p[1][1])
            else:
                cols.append(c)
                ops.append(p[0])
                vals.append(p[1])
        elif with_none:
            cols.append(c)
            ops.append(None)
            vals.append(None)
    return cols, ops, vals


def get_frac(op, total, part):
    if op == '[]':
        if total[0] > part[1] or total[1] < part[0]:
            return 0
        if total[0] > part[0]:
            left = min(part[0], total[0])
            right = min(part[1], total[1])
            return (right - left) / (part[1] - part[0])
        if total[1] < part[1]:
            left = max(part[0], total[0])
            right = max(part[1], total[1])
            return (right - left) / (part[1] - part[0])
        return 1
    elif op == '>=':
        if total > part[1] or total < part[0]:
            return 0
        if total > part[0]:
            left = min(part[0], total)
            right = part[1]
            return (right - left) / (part[1] - part[0])
        return 1
    elif op == '<=':
        if total > part[1] or total < part[0]:
            return 0
        if total < part[1]:
            left = part[0]
            right = max(part[1], total)
            return (right - left) / (part[1] - part[0])
        return 1
    elif op == '=':
        if part[0] <= total < part[1]:
            return 1 / (part[1] - part[0])
        return 0


def query_2_histogram_vec(query: Query, histo):
    vec = {}
    for t in query.predicates.keys():
        for col, pred in query.predicates[t].items():
            for cate in histo[t].col_2_histo_bin[col]:
                vec[f"qh_{t}_{col}_{list_to_str(cate)}"] = 0
            if pred is None:
                pass
            else:
                op, val = pred
                if op == '[]':
                    for cate in histo[t].col_2_histo_bin[col]:
                        vec[f"qh_{t}_{col}_{list_to_str(cate)}"] = get_frac(op, val, cate)
                elif op == '>=':
                    for cate in histo[t].col_2_histo_bin[col]:
                        vec[f"qh_{t}_{col}_{list_to_str(cate)}"] = get_frac(op, val, cate)
                elif op == '<=':
                    for cate in histo[t].col_2_histo_bin[col]:
                        vec[f"qh_{t}_{col}_{list_to_str(cate)}"] = get_frac(op, val, cate)
                elif op == '=':
                    for cate in histo[t].col_2_histo_bin[col]:
                        vec[f"qh_{t}_{col}_{list_to_str(cate)}"] = get_frac(op, val, cate)
    return vec


def dump_queryset(name: str, queryset: Dict[str, List[Query]]) -> None:
    query_path = TEMP_ROOT / "workload"
    query_path.mkdir(exist_ok=True)
    with open(query_path / f"{name}.pkl", 'wb') as f:
        pickle.dump(queryset, f, protocol=PKL_PROTO)


def load_queryset(name: str) -> Dict[str, List[Query]]:
    query_path = TEMP_ROOT / "workload"
    with open(query_path / f"{name}.pkl", 'rb') as f:
        return pickle.load(f)


def dump_sql(name: str, table_dict, group: str = 'test'):
    csv_path = TEMP_ROOT / "workload" / f"{name}_sql.csv"
    queryset = load_queryset(name)
    data = []
    for query in queryset[group]:
        sql = query_2_sql(query, table_dict)
        data.append([sql])
    df = pd.DataFrame(data, columns=['SQL'])
    df.to_csv(csv_path)
