import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2
from treelib import Tree
from enum import IntEnum
import re
import math
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict

from agent.expander.config import load_settings
from agent.planner.models import (
    PathConstraint,
    extract_sql_tables,
    normalize_sql,
    sql_matches_constraint,
)
from pool import get_connection_config

np.set_printoptions(threshold=np.inf)

device = torch.device('cpu')

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def reset_output_files(dbname, target):

    output_files = [
        "sql_templates_tempt.log",
        "bad_sql.log",
        f"RLSQL_{dbname}_pg_{target}.log",
    ]
    for file_name in output_files:
        Path(file_name).write_text("", encoding="utf-8")


def derive_time_window(target, accept_min_ratio, accept_max_ratio):

    return target * accept_min_ratio, target * accept_max_ratio


SELECT_FROM_PATTERN = re.compile(r"\bselect\b(.*?)\bfrom\b", re.IGNORECASE | re.DOTALL)


def select_projection_items(sql: str) -> list[str]:

    match = SELECT_FROM_PATTERN.search(sql)
    if match is None:
        return []
    return [item.strip().lower() for item in match.group(1).split(",") if item.strip()]


def is_trivial_single_table_scan_seed(sql: str, summary: dict[str, Any] | None) -> tuple[bool, str]:

    normalized = normalize_sql(sql)
    if summary is None:
        return False, "Missing summary; skip simple scan filtering"
    if any(token in normalized for token in (" join ", " where ", " having ", " group by ", " order by ")):
        return False, "Contains join/where/having/group by/order by"

    tables = extract_sql_tables(normalized, summary)
    if len(tables) != 1:
        return False, "Not a single-table query"

    projections = select_projection_items(normalized)
    if not projections:
        return False, "Could not parse projection columns"
    if "*" in projections:
        return True, "Bare select ... from single-table query"
    return True, "Bare select ... from single-table query"



operator = ['=', '!=', '>', '<', '<=', '>=']
order_by_key = ['DESC', 'ASC']

predicate_type = ['between', 'like', 'not like']

conjunction = ['and']
aggregate = ['count', 'max', 'min', 'avg', 'sum']

keyword = ['select', 'from', 'aggregate', 'where', 'having', 'order by']
join = ['join', 'cartesian']




parse_node = []

token_name = []
token_reward = []
token_count = []

leaf_token_name = []
leaf_token_reward = []
leaf_token_count = []
parse_connect = []

def sql_literal(s):

    if isinstance(s, str):
        return "'" + s.replace("'", "''") + "'"
    return str(s)

class DataNode(object):

    def __init__(self, action_index, datatype=None, key_type=None):
        self.action_index = action_index
        self.datatype = datatype
        self.key_type = key_type


class RelationGraph(object):

    def __init__(self):
        self.relation_graph = {}
    def add_relation(self, begin, to, relation):
        if begin not in self.relation_graph.keys():
            self.relation_graph[begin] = {}
        self.relation_graph[begin][to] = relation

    def get_relation(self, table):
        return set(self.relation_graph.get(table, {}).keys())

    def get_relation_key(self, begin, end):
        return self.relation_graph[begin][end]








class DataType(IntEnum):
    VALUE = 0
    TIME = 1
    CHAR = 2


AGGREGATE_CONSTRAINTS = {
    DataType.VALUE.value: ['count', 'max', 'min', 'avg', 'sum'],
    DataType.CHAR.value: ['count', 'max', 'min'],
    DataType.TIME.value: ['count', 'max', 'min']
}

TYPE_OPERATOR_CONSTRAINTS = {
    DataType.VALUE.value: ['=', '!=', '>', '<', '<=', '>='],
    DataType.TIME.value: ['=', '!=', '>', '<', '<=', '>='],
    DataType.CHAR.value: ['=', '!='],
}





def transfer_field_type(database_type):
    data_type = [['integer', 'numeric'],
                 ['date']]
    if database_type in data_type[0]:
        return DataType.VALUE.value
    elif database_type in data_type[1]:
        return DataType.TIME.value
    else:
        return DataType.CHAR.value


def allowed_operator_tokens(word_num_map, datatype):
    return [word_num_map[token] for token in TYPE_OPERATOR_CONSTRAINTS[datatype]]


def extract_key(data, key, result=None):
    if result is None:
        result = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                result.append(v)
            extract_key(v, key, result)
    elif isinstance(data, list):
        for item in data:
            extract_key(item, key, result)
    return result

def loadfile(filepath):
    lists = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lists.append(line.rstrip('\n'))
        f.close()
        lists = list(set(lists))
        return lists


def connect_server(dbname):
    connection_config = get_connection_config()
    connection_config["database"] = dbname
    db = psycopg2.connect(**connection_config)
    cursor = db.cursor()
    return db, cursor


def get_table_structure(cursor):

    cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\' AND table_type = \'BASE TABLE\';')
    tables = cursor.fetchall()

    exclude_tables = ['pg_stat_statements']
    tables = [table for table in tables if table[0] not in exclude_tables]

    schema = {}
    for table_info in tables:
        table_name = table_info[0]
        sql = 'SELECT column_name, data_type FROM information_schema.columns WHERE table_name = \'' + table_name + '\';'
        cursor.execute(sql)
        columns = cursor.fetchall()
        schema[table_name] = {}
        for col in columns:
            schema[table_name][col[0]] = [transfer_field_type(col[1])]
    return schema


def load_statitics(tables, types, attributes):
    conn = psycopg2.connect(dbname='tpcc10', user='123', password='123', host='172.6.31.13', port='5432')
    cur = conn.cursor()
    statistics = {}
    for table in tables:
        cur.execute('select tablename, attname, avg_width, n_distinct, most_common_vals, most_common_freqs, histogram_bounds from pg_stats where tablename = \''+table+'\';')
        rows = cur.fetchall()
        for row in rows:
            attname = str(row[1])
            print (table, attname)
            T = types[table][attributes[table].index(attname)]
            avg_width = float(row[2])
            n_distinct = int(row[3])
            most_common_vals, most_common_freqs, histogram_bounds = None, None, None
            if row[4] is not None:
                most_common_vals = [T(x.strip()) for x in str(row[4]).strip('{}').split(',')]
            if row[5] is not None:
                most_common_freqs = [float(x.strip()) for x in str(row[5]).strip('[]').split(',')]
            if row[6] is not None:
                histogram_bounds = [T(x.strip()) for x in str(row[6]).strip('{}').split(',')]
            statistics[attname] = {'avg_width': avg_width, 'n_distinct': n_distinct,
                                   'most_common_vals': most_common_vals, 'most_common_freqs': most_common_freqs,
                                   'histogram_bounds': histogram_bounds, 'type': T}
    return statistics


def selectivity_estimation(attname, op, value, stat):
    statistic = stat[attname]
    T = statistic['type']
    value = T(value)
    avg_width = statistic['avg_width']
    n_distinct = statistic['n_distinct']
    most_common_vals, most_common_freqs, histogram_bounds = statistic['most_common_vals'], statistic['most_common_freqs'], statistic['histogram_bounds']
    selectivity = 0.0
    if op in ['<', '<=']:
        if histogram_bounds is not None:
            for idx, v in enumerate(histogram_bounds):
                if v >= value:
                    if idx > 0:
                        if value is float or value is int:
                            selectivity += float(value - histogram_bounds[idx-1]) / (v - histogram_bounds[idx-1]) / len(histogram_bounds)
                        selectivity += float(idx-1) / len(histogram_bounds)
                    break
        if most_common_vals is not None:
            if op == '<':
                for idx, val in enumerate(most_common_vals):
                    if val < value:
                        selectivity += most_common_freqs[idx]
            elif op == '<=':
                for idx, val in enumerate(most_common_vals):
                    if val <= value:
                        selectivity += most_common_freqs[idx]
    elif op in ['>', '>=']:
        if histogram_bounds is not None:
            for idx in reversed(range(len(histogram_bounds))):
                v = histogram_bounds[idx]
                if v <= value:
                    if idx < len(histogram_bounds) - 1:
                        if value is float or value is int:
                            selectivity += float(value - v) / (histogram_bounds[idx+1] - v) / len(histogram_bounds)
                        selectivity += float(len(histogram_bounds) - 2 - idx) / len(histogram_bounds)
                    break
        if most_common_vals is not None:
            if op == '>':
                for idx, val in enumerate(most_common_vals):
                    if val > value:
                        selectivity += most_common_freqs[idx]
            elif op == '>=':
                for idx, val in enumerate(most_common_vals):
                    if val >= value:
                        selectivity += most_common_freqs[idx]
    elif op == '=' and most_common_vals is not None and value in most_common_vals:
        selectivity += most_common_freqs[most_common_vals.index(value)]
    if selectivity == 0.0:
        if histogram_bounds is not None:
            selectivity += 1.0 / len(histogram_bounds) / avg_width
    return selectivity


def get_tables_sample_data(dbname, schema):
    sample_data = {}
    root_path = os.path.abspath('.')
    for table_name in schema:
        sample_data[table_name] = {}
        for field in schema[table_name]:
            path = root_path + '/' + dbname + '/' + table_name + '/' + field + '.txt'
            sample_data[table_name][field] = loadfile(path)
    return sample_data


def cal_expect_cost():
    with open('./queries/sql_cost') as f:
        total = 0
        count = 0
        for line in f.readlines():
            line = line.strip(('\n'))
            total += float(line)
            count += 1
    f.close()
    return total / count

class GenSqlEnv(object):

    def __init__(
        self,
        metric,
        dbname,
        target,
        accept_min_ratio,
        accept_max_ratio,
        timeout_ratio,
        from_mode,
        max_steps,
        path_constraint: PathConstraint | None = None,
        summary: dict[str, Any] | None = None,
    ):



        self.target = target
        self.need_target = target
        self.accept_min_ratio = accept_min_ratio
        self.accept_max_ratio = accept_max_ratio
        self.timeout_ratio = timeout_ratio
        self.accept_min, self.accept_max = derive_time_window(
            target,
            accept_min_ratio,
            accept_max_ratio,
        )
        self.from_mode = from_mode
        self.max_steps = max_steps
        self.path_constraint = path_constraint
        self.summary = summary

        self.log_target = math.log(metric, 0.95)
        self.dbname = dbname
        self.db, self.cursor = connect_server(dbname)

        self.step_reward = 0
        self.total_reward = 0
        self.reward_count = 0
        self.total_token_reward = dict()
        self.bug_reward = -5.0
        self.unused_episode = 0
        self.last_query_success = False
        self.last_runtime_status = ""
        self.last_runtime = 0.0

        self.statistic_flag = 0
        self.order_by_statistic_flag = 0

        self.terminal_word = " "

        self.word_num_map, self.num_word_map, self.relation_tree, self.relation_graph, self.foreign_key_table, self.foreign_key_column = self._build_relation_env()

        self.action_space = self.observation_space = len(self.word_num_map)


        self.select_space = []
        self.from_space = []
        self.where_space = []
        self.group_by_space = []
        self.having_space = []
        self.order_by_space = []
        self.aggregate_space = []

        self.group_key = False

        self.operator = [self.word_num_map[x] for x in operator]

        self.predicate_type = [self.word_num_map[x] for x in predicate_type]
        self.conjunction = [self.word_num_map[x] for x in conjunction]

        self.keyword = [self.word_num_map[x] for x in keyword]

        self.join = [self.word_num_map[x] for x in join]

        self.attributes = []

        table_node = self.relation_tree.children(self.relation_tree.root)
        self.tables = [field.identifier for field in table_node]
        for node in table_node:
             self.attributes += [field.identifier for field in self.relation_tree.children(node.identifier)]
        self.allowed_table_tokens = self._build_allowed_table_tokens()
        self.allowed_join_edges = self._build_allowed_join_edges()


        self.select_clause = self.from_clause = self.where_clause = self.group_by_clause = self.having_clause =\
            self.order_by_clause = self.aggregate_clause = self.delete_clause = self.update_clause = self.set_clause = ""

        self.master_control = {
            'select': [self.select_observe, self.select_action],
            'from': [self.from_observe, self.from_action],
            'where': [self.where_observe, self.where_action],

            'having': [self.having_observe, self.having_action],
            'order by': [self.order_by_observe, self.order_by_action],
            'aggregate': [self.aggregate_observe, self.aggregate_action],
        }

        self.cur_state = self.master_control['from']
        self.time_step = 0

        self.slow_flag = 0

        self.bad_flag = 0

        self.slow_time = 0


        self.test_time = 0

    def _build_allowed_table_tokens(self):
        if self.path_constraint is None:
            return set(self.tables)
        allowed = {
            self.word_num_map[table_name]
            for table_name in self.path_constraint.allowed_tables
            if table_name in self.word_num_map
        }
        return allowed or set(self.tables)

    def _build_allowed_join_edges(self):
        if self.path_constraint is None:
            return set()
        return {tuple(sorted(edge)) for edge in self.path_constraint.allowed_join_paths}

    def _filter_table_candidates(self, candidates, current_table=None):
        filtered = [table for table in candidates if table in self.allowed_table_tokens]
        if current_table is not None and self.allowed_join_edges:
            current_name = self.num_word_map[current_table]
            filtered = [
                table
                for table in filtered
                if tuple(sorted((current_name, self.num_word_map[table]))) in self.allowed_join_edges
            ]
        return filtered

    def get_from_table_candidates(self, observation):
        if self.from_mode == "single":
            return []
        if self.from_mode == "join":
            relation_tables = self.relation_graph.get_relation(observation)
            filtered = list(relation_tables.difference(self.from_space))
            return self._filter_table_candidates(filtered, current_table=observation)
        if self.from_mode == "cartesian":
            candidates = [table for table in self.tables if table not in self.from_space]
            return self._filter_table_candidates(candidates)
        raise ValueError(f"Unsupported from mode: {self.from_mode}")

    def get_from_join_candidates(self):
        if self.from_mode == "single":
            return []
        if self.from_mode == "join":
            return [self.word_num_map['join']]
        if self.from_mode == "cartesian":
            return [self.word_num_map['cartesian']]
        raise ValueError(f"Unsupported from mode: {self.from_mode}")

    def _build_relation_env(self):
        schema = get_table_structure(self.cursor)
        sample_data = get_tables_sample_data(self.dbname, schema)

        tree = Tree()
        tree.create_node("root", 0, None, data=DataNode(0))

        word_num_map = dict()
        num_word_map = dict()

        word_num_map[self.terminal_word] = 0
        num_word_map[0] = self.terminal_word


        count = 1
        for table_name in schema.keys():
            tree.create_node(table_name, count, parent=0, data=DataNode(count, datatype="table_name"))
            word_num_map[table_name] = count
            num_word_map[count] = table_name
            count += 1


        for table_name in schema.keys():
            for field in schema[table_name].keys():
                attribute = '{0}.{1}'.format(table_name, field)
                tree.create_node(attribute, count, parent=word_num_map[table_name],
                                 data=DataNode(count, datatype=schema[table_name][field][0]))

                word_num_map[attribute] = count
                num_word_map[count] = attribute
                count += 1


        relation_graph = RelationGraph()
        foreign_key_table = set()
        foreign_key_column = set()
        for table_name in schema.keys():
            sql = ""

            sql = '''
            SELECT
                kcu.constraint_name,
                kcu.table_name,
                kcu.column_name,
                ccu.table_name AS referenced_table_name,
                ccu.column_name AS referenced_column_name
            FROM
                information_schema.referential_constraints AS rc
            JOIN
                information_schema.constraint_column_usage AS ccu
                ON rc.unique_constraint_name = ccu.constraint_name
            JOIN
                information_schema.key_column_usage AS kcu
                ON rc.constraint_name = kcu.constraint_name
            WHERE
                kcu.table_schema = 'public' AND
                kcu.constraint_name != 'PRIMARY' AND
                kcu.table_name = '{}';
            '''.format(table_name)

            self.cursor.execute(sql)
            relations = self.cursor.fetchall()
            relation_edges = [
                (relation[2], relation[3], relation[4])
                for relation in relations
            ]

            for column_name, referenced_table, referenced_column in relation_edges:
                if referenced_table not in word_num_map:
                    continue
                relation_from = '{0}.{1}'.format(table_name, column_name)
                relation_to = '{0}.{1}'.format(referenced_table, referenced_column)
                relation_graph.add_relation(word_num_map[table_name], word_num_map[referenced_table], (relation_from, relation_to))
                relation_graph.add_relation(word_num_map[referenced_table], word_num_map[table_name], (relation_to, relation_from))
                foreign_key_table.add(referenced_table)
                foreign_key_column.add(relation_from)
                foreign_key_column.add(relation_to)


        for table_name in schema.keys():
            for field in schema[table_name].keys():
                for data in sample_data[table_name][field]:
                    if data in word_num_map.keys():
                        pass
                    else:
                        word_num_map[data] = len(num_word_map)
                        num_word_map[len(num_word_map)] = data
                    field_name = '{0}.{1}'.format(table_name, field)
                    tree.create_node(data, count, parent=word_num_map[field_name], data=DataNode(word_num_map[data]))
                    count += 1

        self.add_map(operator, word_num_map, num_word_map)

        self.add_map(predicate_type, word_num_map, num_word_map)
        self.add_map(conjunction, word_num_map, num_word_map)

        self.add_map(keyword, word_num_map, num_word_map)

        self.add_map(join, word_num_map, num_word_map)
        return word_num_map, num_word_map, tree, relation_graph, foreign_key_table, foreign_key_column




















    def reset(self):

        self.cur_state = self.master_control['from']
        self.select_clause = self.from_clause = self.where_clause = self.group_by_clause = self.having_clause =\
            self.order_by_clause = self.aggregate_clause = self.delete_clause = self.update_clause = self.set_clause = ""
        self.where_space.clear()
        self.from_space.clear()
        self.select_space.clear()
        self.aggregate_space.clear()
        self.group_by_space.clear()
        self.order_by_space.clear()
        self.having_space.clear()
        self.time_step = 0
        self.step_reward = 0
        self.group_key = False
        self.last_query_success = False
        self.last_runtime_status = ""
        self.last_runtime = 0.0

        return self.word_num_map['from']

    def activate_space(self, cur_space, keyword):

        cur_space[keyword] = 1

    def activate_ternminal(self, cur_space):
        cur_space[0] = 1

    def select_observe(self, observation):

        candidate_word = np.zeros((self.action_space,), dtype=int)
        if self.num_word_map[observation] == 'select' or observation in self.join:
            self.need_select_table = self.from_space.copy()
            for table_index in self.need_select_table:
                candidate_word[[field.identifier for field in self.relation_tree.children(table_index)]] = 1
            return candidate_word
        else:
            if self.need_select_table:
                for table_index in self.need_select_table:
                    candidate_word[[field.identifier for field in self.relation_tree.children(table_index)]] = 1
                return candidate_word
            else:
                self.activate_space(candidate_word, self.word_num_map['aggregate'])
                self.activate_space(candidate_word, self.word_num_map['where'])
                self.activate_space(candidate_word, self.word_num_map['order by'])
                if len(self.from_space) > 1 or self.where_clause or self.order_by_clause:
                    self.activate_ternminal(candidate_word)
                return candidate_word

    def select_action(self, action):

        if self.num_word_map[action] == 'select' or action in self.join:
            self.select_clause = 'select'
        elif action in self.keyword:
            self.cur_state = self.master_control[self.num_word_map[action]]
            self.cur_state[1](action)
        else:
            self.select_space.append(action)
            self.group_by_space.append(action)
            self.order_by_space.append(action)
            table_name_index = self.relation_tree.parent(action).identifier
            self.need_select_table.remove(table_name_index)
            self.select_clause = self.select_clause + ' ' + self.num_word_map[action] + ','
        return self.step_reward, 0

    def aggregate_observe(self, observation=None):
        candidate_word = np.zeros((self.action_space,), dtype=int)
        if self.group_key is False:
            self.group_by_generate()
            self.group_key = True
        self.activate_space(candidate_word, self.word_num_map['aggregate'])
        self.activate_space(candidate_word, self.word_num_map['where'])
        self.activate_space(candidate_word, self.word_num_map['order by'])
        self.activate_space(candidate_word, self.word_num_map['having'])
        self.activate_ternminal(candidate_word)
        return candidate_word

    def aggregate_action(self, action):
        if action == self.word_num_map['aggregate']:
            while True:
                table = np.random.choice(self.from_space)
                attributes = [node.identifier for node in self.relation_tree.children(table)]
                choose_attribute = np.random.choice(attributes)
                choose_aggregate_type = np.random.choice(AGGREGATE_CONSTRAINTS[self.relation_tree.get_node(choose_attribute).data.datatype])
                if (choose_aggregate_type, choose_attribute) not in self.aggregate_space:
                    break
            self.aggregate_space.append((choose_aggregate_type, choose_attribute))
            self.aggregate_clause = self.aggregate_clause + ' ' + '{aggregate_type}({aggregate_attribute})'.format(
                aggregate_type=choose_aggregate_type, aggregate_attribute=self.num_word_map[choose_attribute]) + ','
        else:
            self.cur_state = self.master_control[self.num_word_map[action]]
            self.cur_state[1](action)
        return self.step_reward, 0

    def from_observe(self, observation=None):
        if observation == self.word_num_map['from']:
            self.from_clause = 'from'
            candidate_tables = np.zeros((self.action_space,), dtype=int)
            candidate_tables[list(self.allowed_table_tokens)] = 1
            return candidate_tables
        else:
            next_tables = self.get_from_table_candidates(observation)
            candidate_tables = np.zeros((self.action_space,), dtype=int)

            if len(self.from_space) > 1:
                join_candidates = self.get_from_join_candidates()
                if join_candidates:
                    candidate_tables[join_candidates] = 1
                if next_tables:
                    candidate_tables[next_tables] = 1
            else:
                if next_tables:
                    candidate_tables[next_tables] = 1
                if self.from_mode != "join":
                    candidate_tables[self.word_num_map['select']] = 1
            return candidate_tables

    def from_action(self, action):

        if action in self.tables:
            self.from_space.append(action)
        elif action == self.word_num_map['select']:
            self.from_clause = self.from_clause + ' ' + self.num_word_map[self.from_space[0]]
            self.cur_state = self.master_control[ 'select' ]
            self.cur_state[1](action)
        else:
            if action == self.word_num_map['cartesian']:
                for table_index in self.from_space:
                    self.from_clause = self.from_clause + ' ' + self.num_word_map[table_index] + ','
                self.from_clause = self.from_clause[: -1]
            else:
                join_type = self.num_word_map[action]


                if len(self.from_space) > 0:
                    self.from_clause = self.from_clause + ' ' + self.num_word_map[self.from_space[0]]
                else:
                    self.from_clause += " <missing_table>"

                for i in range(1, len(self.from_space)):
                    relation_key = self.relation_graph.get_relation_key(self.from_space[i], self.from_space[i - 1])
                    self.from_clause = self.from_clause + ' ' + join_type + ' ' +\
                                       self.num_word_map[self.from_space[i]] + ' on ' + relation_key[0] + '=' + relation_key[1]
            self.cur_state = self.master_control['select']
            self.cur_state[1](action)

        return self.step_reward, 0

    def where_observe(self, observation):

        candidate_word = np.zeros((self.action_space,), dtype=int)
        if observation == self.word_num_map['where']:
            self.where_attributes = []
            for table_index in self.from_space:
                for field in self.relation_tree.children(table_index):
                    self.where_attributes.append(field.identifier)
            candidate_word[self.where_attributes] = 1
            return candidate_word
        elif observation in self.attributes:
            attribute_type = self.relation_tree.get_node(observation).data.datatype
            candidate_word[allowed_operator_tokens(self.word_num_map, attribute_type)] = 1

            return candidate_word
        elif observation in self.operator:
            candidate_word[self.operation_data(self.cur_attribtue)] = 1
            return candidate_word
        elif observation in self.conjunction:
            candidate_word[self.where_attributes] = 1
            return candidate_word
        else:
            candidate_word[self.conjunction] = 1
            self.activate_ternminal(candidate_word)
            self.activate_space(candidate_word, self.word_num_map['order by'])
            if self.group_key:
                self.activate_space(candidate_word, self.word_num_map['having'])
            return candidate_word



    def where_action(self, action):


        if action == self.word_num_map['where']:
            self.where_clause = 'where '
        elif action in self.attributes:
            self.cur_attribtue = action
            self.where_clause = self.where_clause + self.num_word_map[action]
        elif action in self.operator:
            self.where_clause = self.where_clause + ' ' + self.num_word_map[action] + ' '
        elif action in self.conjunction:
            self.where_clause = self.where_clause + ' {} '.format(self.num_word_map[action])
        elif action in self.keyword:
            self.cur_state = self.master_control[self.num_word_map[action]]
            self.cur_state[1](action)
        else:
            attribute_type = self.relation_tree.get_node(self.cur_attribtue).data.datatype




            value = self.num_word_map[action]
            if attribute_type == DataType.VALUE.value:
                self.where_clause += value
            else:
                self.where_clause += sql_literal(value)


        return self.step_reward, 0

    def operation_data(self, attributes):
        data = [node.data.action_index for node in self.relation_tree.children(attributes)]
        return data

    def group_by_generate(self):
        self.group_by_clause = 'group by'
        for attribute in self.group_by_space:
            self.group_by_clause = self.group_by_clause + ' ' + self.num_word_map[attribute] + ','
        self.group_by_clause = self.group_by_clause[: -1]

    def having_observe(self, observation):

        candidate_word = np.zeros((self.action_space,), dtype=int)
        cur_word = self.num_word_map[observation]
        if cur_word == 'having':
            aggregate_attributes = [record[1] for record in self.having_space]
            candidate_word[aggregate_attributes] = 1
            return candidate_word
        elif observation in self.attributes:
            self.activate_ternminal(candidate_word)
            self.activate_space(candidate_word, self.word_num_map['order by'])
            if len(self.having_space) != 0:
                candidate_word[self.conjunction] = 1
            return candidate_word
        else:
            aggregate_attributes = [record[1] for record in self.having_space]
            candidate_word[aggregate_attributes] = 1
            return candidate_word

    def having_action(self, action):
        if action == self.word_num_map['having']:
            self.having_clause = 'having'
            self.having_space = self.aggregate_space.copy()
        elif action in self.attributes:
            chosen_item = -1
            for item in self.having_space:
                if item[1] == action:
                    chosen_item = item
                    break
            self.having_space.remove(chosen_item)
            assert chosen_item[1] == action
            chosen_attribute_type = self.relation_tree.get_node(chosen_item[1]).data.datatype
            if chosen_item[0] == 'count':
                chosen_operator = np.random.choice(TYPE_OPERATOR_CONSTRAINTS[DataType.VALUE.value])
            else:
                chosen_operator = np.random.choice(TYPE_OPERATOR_CONSTRAINTS[chosen_attribute_type])

            if chosen_item[0] == 'count':
                chosen_data = np.random.choice(10)
            else:
                if self.operation_data(action):
                    chosen_data = np.random.choice(self.operation_data(action))
                else:
                    chosen_data = np.random.choice(10)

            self.having_clause = self.having_clause + ' ' + '{aggregate_type}({attribute})'.format(
                aggregate_type=chosen_item[0], attribute=self.num_word_map[chosen_item[1]]) + ' ' + chosen_operator + ' '










            value = self.num_word_map[chosen_data]
            if chosen_item[0] == 'count':
                self.having_clause = self.having_clause + str(chosen_data)
            else:
                if chosen_attribute_type == DataType.CHAR.value:
                    self.having_clause += sql_literal(str(value))
                else:
                    self.having_clause += value

        elif action in self.conjunction:
            self.having_clause = self.having_clause + ' ' + self.num_word_map[action]
        else:
            self.cur_state = self.master_control[self.num_word_map[action]]
            self.cur_state[1](action)
        return self.step_reward, 0

    def order_by_observe(self, observation):
        candidate_word = np.zeros((self.action_space,), dtype=int)
        if observation == self.word_num_map['order by']:
            self.activate_space(candidate_word, self.word_num_map['select'])
            if self.group_key:
                self.activate_space(candidate_word, self.word_num_map['aggregate'])
        else:
            self.activate_ternminal(candidate_word)
        return candidate_word

    def order_by_action(self, action):
        if action == self.word_num_map['order by']:
            self.order_by_clause = 'order by'
        elif action == self.word_num_map['select']:
            number = np.random.randint(1, len(self.select_space) + 1)
            attributes = np.random.choice(self.select_space, size=number, replace=False)
            for attribute in attributes:
                choose_order = np.random.choice(order_by_key)
                self.order_by_clause = self.order_by_clause + ' ' + self.num_word_map[attribute] + ' ' + choose_order + ','
        else:
            number = np.random.randint(1, len(self.aggregate_space) + 1)
            tuple_indexes = np.random.choice(range(0, len(self.aggregate_space)), size=number, replace=False)
            for index in tuple_indexes:
                aggregate_tuple = self.aggregate_space[index]
                choose_order = np.random.choice(order_by_key)
                self.order_by_clause = self.order_by_clause + ' ' + '{}({})'.format(aggregate_tuple[0], self.num_word_map[aggregate_tuple[1]]) + ' ' + choose_order + ','
        return self.step_reward, 0

    def add_map(self, series, word_num_map, num_word_map):
        count = len(word_num_map)
        for word in series:
            if word not in word_num_map.keys():
                word_num_map[word] = count
                num_word_map[count] = word
                count += 1

    def observe(self, observation):


        return self.cur_state[0](observation)

    def step(self, action):
        self.time_step += 1


        if action == 0:
            return self.final_reward(), 1


        else:
            return self.cur_state[1](action)

    def encode_one_hot(self, action):
        one_hot = np.zeros(self.action_space)
        one_hot[action] = 1
        return one_hot

    def get_sql(self):
        final_sql = ''
        if self.select_clause:
            final_sql = final_sql + self.select_clause[:-1]
        if self.aggregate_clause:
            final_sql = final_sql + ', ' + self.aggregate_clause[1:-1]
        final_sql = final_sql + ' ' + self.from_clause
        if self.where_clause:
            final_sql = final_sql + ' ' + self.where_clause
        if self.group_by_clause:
            final_sql = final_sql + ' ' + self.group_by_clause
        if self.having_clause:
            final_sql = final_sql + ' ' + self.having_clause
        if self.order_by_clause:
            final_sql = final_sql + ' ' + self.order_by_clause[:-1]
        final_sql = final_sql + ';'
        return final_sql





































    def reward_modify(self, reward):
        log_reward = 0
        if reward != 0:
            log_cost = math.log(reward, 1.5)
        dist = abs(log_reward - self.log_target)

        return 1 / dist - 0.45

    @staticmethod
    def extract_execution_time_ms(plan_payload):
        if isinstance(plan_payload, list) and plan_payload:
            root = plan_payload[0]
        else:
            root = plan_payload
        if isinstance(root, dict) and isinstance(root.get("Execution Time"), (int, float)):
            return float(root["Execution Time"])
        raise ValueError("EXPLAIN ANALYZE output does not contain Execution Time")

    def get_run_time(self):
        inject_sql = self.get_sql()
        try:
            self.db.rollback()
            self.db.autocommit = False
            timeout_ms = int(self.need_target * self.timeout_ratio * 1000)
            with self.db.cursor() as cursor:
                cursor.execute(f"SET LOCAL statement_timeout = {timeout_ms};")
                cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + inject_sql)
                row = cursor.fetchone()
            runtime = self.extract_execution_time_ms(row[0]) / 1000.0
            self.db.commit()
            return {
                "status": "ok",
                "runtime": runtime,
            }
        except Exception as result:
            self.db.rollback()
            if "canceling statement due to statement timeout" in str(result):
                self.slow_flag = 1
                s_time = time.time()
                e_time = time.time()
                self.test_time += e_time - s_time
                return {
                    "status": "timeout",
                    "runtime": self.need_target * self.timeout_ratio,
                }
            else:
                return {
                    "status": "error",
                    "runtime": 0,
                    "message": str(result),
                }


    def get_explain(self):
        db, cursor = connect_server(self.dbname)
        try:
            cursor.execute('EXPLAIN (FORMAT JSON) ' + self.get_sql())
            res = cursor.fetchall()
            res = res[0][0]

            plan_rows = extract_key(res, 'Plan Rows')

            rows = sum(plan_rows)

            is_possible_key = extract_key(res, 'Node Type')

            possible_key = 0
            for s in is_possible_key:
                if "Index" in s:
                    possible_key = 1
                    break

            return rows, possible_key
        except Exception as result:
            cursor.close()
            db.close()
            self.db, self.cursor = connect_server(self.dbname)
            return None, None

    def get_sql_tree(self, final_reward):
        select_dict = defaultdict(list)
        delete_dict = defaultdict(list)
        update_dict = defaultdict(list)

        if self.select_clause:

            select_dict["select_clause"] = [self.select_clause[: -1].split(" ")[0]]

            if self.aggregate_clause:
                select_dict["statistic_clause"] = self.aggregate_clause[1: -1].split(
                    ",")

            select_dict["column_name"] = self.select_clause[: -1].split(" ")[1:]

            if len(parse_node) == 0:
                parse_node.append(select_dict["select_clause"])
                if self.aggregate_clause:
                    self.statistic_flag = 1
                    statistic = []
                    column = []
                    for statistic_word in select_dict["statistic_clause"]:
                        if [statistic_word.split("(")[0], select_dict["select_clause"][0]] not in parse_connect:
                            parse_connect.append([statistic_word.split("(")[0], select_dict["select_clause"][0]])
                        if [statistic_word.split("(")[1][:-1], statistic_word.split("(")[0]] not in parse_connect:
                            parse_connect.append([statistic_word.split("(")[1][:-1], statistic_word.split("(")[0]])
                        if ["from", statistic_word.split("(")[1][:-1]] not in parse_connect:
                            parse_connect.append(["from", statistic_word.split("(")[1][:-1]])

                        if statistic_word.split("(")[0] not in statistic:
                            statistic.append(statistic_word.split("(")[0])
                        if statistic_word.split("(")[1][:-1] not in column:
                            column.append(statistic_word.split("(")[1][:-1])
                    parse_node.append(statistic)
                    parse_node.append(column)
                for column_name in select_dict["column_name"]:
                    if [column_name, select_dict["select_clause"][0]] not in parse_connect:
                        parse_connect.append([column_name, select_dict["select_clause"][0]])
                    if ["from", column_name] not in parse_connect:
                        parse_connect.append(["from", column_name])

                    if self.aggregate_clause:
                        if column_name not in parse_node[2]:
                            parse_node[2].append(column_name)
                if not self.aggregate_clause:
                    parse_node.append(select_dict["column_name"])
            else:
                if self.aggregate_clause:
                    if self.statistic_flag == 1:
                        select = select_dict["select_clause"][0]
                        for statistic_word in select_dict["statistic_clause"]:
                            the_statistic = statistic_word.split("(")[0]
                            column_name = statistic_word.split("(")[1][:-1]
                            if [the_statistic, select] not in parse_connect:
                                parse_connect.append([the_statistic, select])
                            if [column_name, the_statistic] not in parse_connect:
                                parse_connect.append([column_name, the_statistic])
                            if ["from", column_name] not in parse_connect:
                                parse_connect.append(["from", column_name])

                            if the_statistic not in parse_node[1]:
                                parse_node[1].append(the_statistic)
                            if column_name not in parse_node[2]:
                                parse_node[2].append(column_name)

                        for column_name in select_dict["column_name"]:
                            if [column_name, select] not in parse_connect:
                                parse_connect.append([column_name, select])
                            if ["from", column_name] not in parse_connect:
                                parse_connect.append(["from", column_name])

                            if column_name not in parse_node[2]:
                                parse_node[2].append(column_name)

                    else:
                        statistic = []
                        select = select_dict["select_clause"][0]
                        for statistic_word in select_dict["statistic_clause"]:
                            the_statistic = statistic_word.split("(")[0]
                            column_name = statistic_word.split("(")[1][:-1]
                            if [the_statistic, select] not in parse_connect:
                                parse_connect.append([the_statistic, select])
                            if [column_name, the_statistic] not in parse_connect:
                                parse_connect.append([column_name, the_statistic])
                            if ["from", column_name] not in parse_connect:
                                parse_connect.append(["from", column_name])

                            statistic.append(the_statistic)
                            if column_name not in parse_node[1]:
                                parse_node[1].append(column_name)
                        parse_node.insert(1, statistic)

                        for column_name in select_dict["column_name"]:
                            if [column_name, select] not in parse_connect:
                                parse_connect.append([column_name, select])
                            if ["from", column_name] not in parse_connect:
                                parse_connect.append(["from", column_name])

                            if column_name not in parse_node[2]:
                                parse_node[2].append(column_name)
                    self.statistic_flag = 1
                else:
                    if self.statistic_flag == 1:
                        select = select_dict["select_clause"]

                        for column_name in select_dict["column_name"]:
                            if [column_name, select] not in parse_connect:
                                parse_connect.append([column_name, select])
                            if ["from", column_name] not in parse_connect:
                                parse_connect.append(["from", column_name])

                            if column_name not in parse_node[2]:
                                parse_node[2].append(column_name)
                    else:
                        select = select_dict["select_clause"][0]

                        for column_name in select_dict["column_name"]:
                            if [column_name, select] not in parse_connect:
                                parse_connect.append([column_name, select])
                            if ["from", column_name] not in parse_connect:
                                parse_connect.append(["from", column_name])

                            if column_name not in parse_node[1]:
                                parse_node[1].append(column_name)

        if self.delete_clause:
            delete_dict["delete_clause"] = [self.delete_clause.split(" ")[0]]

            if len(parse_node) == 0:
                parse_node.append(delete_dict["delete_clause"])
                parse_connect.append(["from", delete_dict["delete_clause"][0]])
            else:
                if delete_dict["delete_clause"][0] not in parse_node[0]:
                    parse_node[0].append(delete_dict["delete_clause"][0])
                if ["from", delete_dict["delete_clause"][0]] not in parse_connect:
                    parse_connect.append(["from", delete_dict["delete_clause"][0]])

        if self.update_clause:
            update_dict["update_clause"] = [self.update_clause]
            update_dict["table_name"] = [self.from_clause.split(" ")[1]]

            if len(parse_node) == 0:
                parse_node.append(update_dict["update_clause"])
                parse_node.append(update_dict["table_name"])
                parse_connect.append([update_dict["table_name"][0], update_dict["update_clause"][0]])
                parse_connect.append(["set", update_dict["table_name"][0]])
            elif len(parse_node) == 1:
                if update_dict["update_clause"][0] not in parse_node[0]:
                    parse_node[0].append(update_dict["table_name"][0])
                parse_node.append(update_dict["table_name"])
                if [update_dict["table_name"][0], update_dict["update_clause"][0]] not in parse_connect:
                    parse_connect.append([update_dict["table_name"][0], update_dict["update_clause"][0]])
                if ["set", update_dict["table_name"][0]] not in parse_connect:
                    parse_connect.append(["set", update_dict["table_name"][0]])
            else:
                if update_dict["update_clause"][0] not in parse_node[0]:
                    parse_node[0].append(update_dict["update_clause"][0])
                if update_dict["table_name"][0] not in parse_node[1]:
                    parse_node[1].append(update_dict["table_name"][0])
                if [update_dict["table_name"][0], update_dict["update_clause"][0]] not in parse_connect:
                    parse_connect.append([update_dict["table_name"][0], update_dict["update_clause"][0]])
                if ["set", update_dict["table_name"][0]] not in parse_connect:
                    parse_connect.append(["set", update_dict["table_name"][0]])


        from_dict = defaultdict(list)
        from_dict["from_clause"] = [self.from_clause.split(" ")[0]]
        from_dict["table_name"] = [self.from_clause.split(" ")[-1]]

        if from_dict["from_clause"] not in parse_node:
            parse_node.append(from_dict["from_clause"])
            parse_node.append(from_dict["table_name"])
        else:
            the_index = parse_node.index(from_dict["from_clause"])
            if from_dict["table_name"][0] not in parse_node[the_index + 1]:
                parse_node[the_index + 1].append(from_dict["table_name"][0])
        if [from_dict["table_name"][0], from_dict["from_clause"][0]] not in parse_connect:
            parse_connect.append([from_dict["table_name"][0], from_dict["from_clause"][0]])
        if self.where_clause:
            if ["where", from_dict["table_name"][0]] not in parse_connect:
                parse_connect.append(["where", from_dict["table_name"][0]])
        else:
            if self.group_by_clause:
                if ["group by", from_dict["table_name"][0]] not in parse_connect:
                    parse_connect.append(["group by", from_dict["table_name"][0]])
            else:
                if self.order_by_clause:
                    if ["order by", from_dict["table_name"][0]] not in parse_connect:
                        parse_connect.append(["order by", from_dict["table_name"][0]])
                else:
                    for one_token in from_dict["table_name"]:
                        if one_token not in leaf_token_name:
                            leaf_token_name.append(one_token)
                            leaf_token_reward.append(final_reward)
                            leaf_token_count.append(1)
                        else:
                            leaf_token_index = leaf_token_name.index(one_token)
                            tempt_reward = leaf_token_reward[leaf_token_index]
                            tempt_count = leaf_token_count[leaf_token_index]
                            leaf_token_reward[leaf_token_index] = tempt_reward + final_reward
                            leaf_token_count[leaf_token_index] = tempt_count + 1

        if self.set_clause:
            set_dict = defaultdict(list)

            set_dict["set_clause"] = [self.set_clause.split(" ")[0]]

            set_dict["column_name"] = [self.set_clause.split(" ")[1]]

            set_dict["operator_clause"] = [self.set_clause.split(" ")[2]]

            set_dict["sampled_cell_value"] = [self.set_clause.split(" ")[3]]

            if set_dict["set_clause"] not in parse_node:
                parse_node.append(set_dict["set_clause"])
                parse_node.append(set_dict["column_name"])
                parse_node.append(set_dict["operator_clause"])
                parse_node.append(set_dict["operator_clause"])
            else:
                the_index = parse_node.index(set_dict["set_clause"])
                if set_dict["column_name"][0] not in parse_node[the_index + 1]:
                    parse_node[the_index + 1].append(set_dict["column_name"][0])
                if set_dict["operator_clause"][0] not in parse_node[the_index + 1]:
                    parse_node[the_index + 1].append(set_dict["operator_clause"][0])
                if set_dict["sampled_cell_value"][0] not in parse_node[the_index + 1]:
                    parse_node[the_index + 1].append(set_dict["sampled_cell_value"][0])
            if [set_dict["column_name"][0], set_dict["set_clause"][0]] not in parse_connect:
                parse_connect.append([set_dict["column_name"][0], set_dict["set_clause"][0]])
            if [set_dict["operator_clause"][0], set_dict["column_name"][0]] not in parse_connect:
                parse_connect.append([set_dict["operator_clause"][0], set_dict["column_name"][0]])
            if [set_dict["sampled_cell_value"][0], set_dict["operator_clause"][0]] not in parse_connect:
                parse_connect.append([set_dict["sampled_cell_value"][0], set_dict["operator_clause"][0]])
            if self.where_clause:
                if ["where", set_dict["sampled_cell_value"][0]] not in parse_connect:
                    parse_connect.append(["where", set_dict["sampled_cell_value"][0]])
            else:
                for one_token in set_dict["sampled_cell_value"]:
                    if one_token not in leaf_token_name:
                        leaf_token_name.append(one_token)
                        leaf_token_reward.append(final_reward)
                        leaf_token_count.append(1)
                    else:
                        leaf_token_index = leaf_token_name.index(one_token)
                        tempt_reward = leaf_token_reward[leaf_token_index]
                        tempt_count = leaf_token_count[leaf_token_index]
                        leaf_token_reward[leaf_token_index] = tempt_reward + final_reward
                        leaf_token_count[leaf_token_index] = tempt_count + 1

        if self.where_clause:
            parts = re.split(r'\b(and|or)\b', self.where_clause)
            parts = [part.strip('; ') for part in parts]

            where_part = parts[0]
            where_dict = defaultdict(list)

            where_dict["where_clause"] = [where_part.split(" ")[0]]

            where_dict["column_name"] = [where_part.split(" ")[1]]

            where_dict["operator_clause"] = [where_part.split(" ")[2]]

            where_dict["sampled_cell_value"] = [where_part.split(" ")[3]]


            if where_dict["where_clause"] not in parse_node:
                parse_node.append(where_dict["where_clause"])
                parse_node.append(where_dict["column_name"])
                parse_node.append(where_dict["operator_clause"])
                parse_node.append(where_dict["sampled_cell_value"])
            else:
                the_index = parse_node.index(where_dict["where_clause"])
                if where_dict["column_name"][0] not in parse_node[the_index + 1]:
                    parse_node[the_index + 1].append(where_dict["column_name"][0])
                if where_dict["operator_clause"][0] not in parse_node[the_index + 2]:
                    parse_node[the_index + 2].append(where_dict["operator_clause"][0])
                if where_dict["sampled_cell_value"][0] not in parse_node[the_index + 3]:
                    parse_node[the_index + 3].append(where_dict["sampled_cell_value"][0])

            if len(parts) == 1:
                if self.group_by_clause:
                    if [where_dict["column_name"][0], where_dict["where_clause"][0]] not in parse_connect:
                        parse_connect.append([where_dict["column_name"][0], where_dict["where_clause"][0]])
                    if [where_dict["operator_clause"][0], where_dict["column_name"][0]] not in parse_connect:
                        parse_connect.append([where_dict["operator_clause"][0], where_dict["column_name"][0]])
                    if [where_dict["sampled_cell_value"][0], where_dict["operator_clause"][0]] not in parse_connect:
                        parse_connect.append([where_dict["sampled_cell_value"][0], where_dict["operator_clause"][0]])
                    if ["group by", where_dict["sampled_cell_value"][0]] not in parse_connect:
                        parse_connect.append(["group by", where_dict["sampled_cell_value"][0]])
                else:
                    if self.order_by_clause:
                        if [where_dict["column_name"][0], where_dict["where_clause"][0]] not in parse_connect:
                            parse_connect.append([where_dict["column_name"][0], where_dict["where_clause"][0]])
                        if [where_dict["operator_clause"][0], where_dict["column_name"][0]] not in parse_connect:
                            parse_connect.append([where_dict["operator_clause"][0], where_dict["column_name"][0]])
                        if [where_dict["sampled_cell_value"][0], where_dict["operator_clause"][0]] not in parse_connect:
                            parse_connect.append(
                                [where_dict["sampled_cell_value"][0], where_dict["operator_clause"][0]])
                        if ["order by", where_dict["sampled_cell_value"][0]] not in parse_connect:
                            parse_connect.append(["order by", where_dict["sampled_cell_value"][0]])
                    else:
                        if [where_dict["column_name"][0], where_dict["where_clause"][0]] not in parse_connect:
                            parse_connect.append([where_dict["column_name"][0], where_dict["where_clause"][0]])
                        if [where_dict["operator_clause"][0], where_dict["column_name"][0]] not in parse_connect:
                            parse_connect.append([where_dict["operator_clause"][0], where_dict["column_name"][0]])
                        if [where_dict["sampled_cell_value"][0], where_dict["operator_clause"][0]] not in parse_connect:
                            parse_connect.append(
                                [where_dict["sampled_cell_value"][0], where_dict["operator_clause"][0]])
                        for one_token in where_dict["sampled_cell_value"]:
                            if one_token not in leaf_token_name:
                                leaf_token_name.append(one_token)
                                leaf_token_reward.append(final_reward)
                                leaf_token_count.append(1)
                            else:
                                leaf_token_index = leaf_token_name.index(one_token)
                                tempt_reward = leaf_token_reward[leaf_token_index]
                                tempt_count = leaf_token_count[leaf_token_index]
                                leaf_token_reward[leaf_token_index] = tempt_reward + final_reward
                                leaf_token_count[leaf_token_index] = tempt_count + 1
            else:
                if [where_dict["column_name"][0], where_dict["where_clause"][0]] not in parse_connect:
                    parse_connect.append([where_dict["column_name"][0], where_dict["where_clause"][0]])
                if [where_dict["operator_clause"][0], where_dict["column_name"][0]] not in parse_connect:
                    parse_connect.append([where_dict["operator_clause"][0], where_dict["column_name"][0]])
                if [where_dict["sampled_cell_value"][0], where_dict["operator_clause"][0]] not in parse_connect:
                    parse_connect.append([where_dict["sampled_cell_value"][0], where_dict["operator_clause"][0]])
                if [parts[1] + str(0), where_dict["sampled_cell_value"][0]] not in parse_connect:
                    parse_connect.append([parts[1] + str(0), where_dict["sampled_cell_value"][0]])

            and_or_dict_list = []
            and_or_dict = defaultdict(list)
            if len(parts) > 1:
                index = 1
                count = 0
                while index < len(parts):

                    and_or_dict["and_or_clause"] = [parts[index] + str(count)]
                    and_or_part = parts[index + 1]

                    and_or_dict["column_name"] = [and_or_part.split(" ")[0]]

                    and_or_dict["operator_clause"] = [and_or_part.split(" ")[1]]

                    and_or_dict["sampled_cell_value"] = [and_or_part.split(" ")[2]]


                    where_index = parse_node.index(where_dict["where_clause"])
                    and_or_index = where_index + 3 + 1 + count * 4
                    if and_or_index >= len(parse_node):
                        parse_node.append(and_or_dict["and_or_clause"])
                        parse_node.append(and_or_dict["column_name"])
                        parse_node.append(and_or_dict["operator_clause"])
                        parse_node.append(and_or_dict["sampled_cell_value"])
                    elif "order by" in parse_node[and_or_index] or "group by" in parse_node[and_or_index]:
                        parse_node.insert(and_or_index, and_or_dict["and_or_clause"])
                        parse_node.insert(and_or_index + 1, and_or_dict["column_name"])
                        parse_node.insert(and_or_index + 2, and_or_dict["operator_clause"])
                        parse_node.insert(and_or_index + 3, and_or_dict["sampled_cell_value"])

                    else:
                        if and_or_dict["and_or_clause"][0] not in parse_node[and_or_index]:
                            parse_node[and_or_index].append(and_or_dict["and_or_clause"][0])
                        if and_or_dict["column_name"][0] not in parse_node[and_or_index + 1]:
                            parse_node[and_or_index + 1].append(and_or_dict["column_name"][0])
                        if and_or_dict["operator_clause"][0] not in parse_node[and_or_index + 2]:
                            parse_node[and_or_index + 2].append(and_or_dict["operator_clause"][0])
                        if and_or_dict["sampled_cell_value"][0] not in parse_node[and_or_index + 3]:
                            parse_node[and_or_index + 3].append(and_or_dict["sampled_cell_value"][0])
                    if index + 2 < len(parts):
                        if [and_or_dict["column_name"][0], parts[index] + str(count)] not in parse_connect:
                            parse_connect.append([and_or_dict["column_name"][0], parts[index] + str(count)])
                        if [and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]] not in parse_connect:
                            parse_connect.append([and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]])
                        if [and_or_dict["sampled_cell_value"][0],
                            and_or_dict["operator_clause"][0]] not in parse_connect:
                            parse_connect.append(
                                [and_or_dict["sampled_cell_value"][0], and_or_dict["operator_clause"][0]])
                        if [parts[index + 2] + str(count + 1),
                            and_or_dict["sampled_cell_value"][0]] not in parse_connect:
                            parse_connect.append(
                                [parts[index + 2] + str(count + 1), and_or_dict["sampled_cell_value"][0]])
                    else:
                        if self.group_by_clause:
                            if [and_or_dict["column_name"][0], parts[index] + str(count)] not in parse_connect:
                                parse_connect.append([and_or_dict["column_name"][0], parts[index] + str(count)])
                            if [and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]] not in parse_connect:
                                parse_connect.append([and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]])
                            if [and_or_dict["sampled_cell_value"][0],
                                and_or_dict["operator_clause"][0]] not in parse_connect:
                                parse_connect.append(
                                    [and_or_dict["sampled_cell_value"][0], and_or_dict["operator_clause"][0]])
                            if ["group by", and_or_dict["sampled_cell_value"][0]] not in parse_connect:
                                parse_connect.append(["group by", and_or_dict["sampled_cell_value"][0]])
                        else:
                            if self.order_by_clause:
                                if [and_or_dict["column_name"][0], parts[index] + str(count)] not in parse_connect:
                                    parse_connect.append([and_or_dict["column_name"][0], parts[index] + str(count)])
                                if [and_or_dict["operator_clause"][0],
                                    and_or_dict["column_name"][0]] not in parse_connect:
                                    parse_connect.append(
                                        [and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]])
                                if [and_or_dict["sampled_cell_value"][0],
                                    and_or_dict["operator_clause"][0]] not in parse_connect:
                                    parse_connect.append(
                                        [and_or_dict["sampled_cell_value"][0], and_or_dict["operator_clause"][0]])
                                if ["order by", and_or_dict["sampled_cell_value"][0]] not in parse_connect:
                                    parse_connect.append(["order by", and_or_dict["sampled_cell_value"][0]])
                            else:
                                if [and_or_dict["column_name"][0], parts[index] + str(count)] not in parse_connect:
                                    parse_connect.append([and_or_dict["column_name"][0], parts[index] + str(count)])
                                if [and_or_dict["operator_clause"][0],
                                    and_or_dict["column_name"][0]] not in parse_connect:
                                    parse_connect.append(
                                        [and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]])
                                if [and_or_dict["sampled_cell_value"][0],
                                    and_or_dict["operator_clause"][0]] not in parse_connect:
                                    parse_connect.append(
                                        [and_or_dict["sampled_cell_value"][0], and_or_dict["operator_clause"][0]])
                                for one_token in and_or_dict["sampled_cell_value"]:
                                    if one_token not in leaf_token_name:
                                        leaf_token_name.append(one_token)
                                        leaf_token_reward.append(final_reward)
                                        leaf_token_count.append(1)
                                    else:
                                        leaf_token_index = leaf_token_name.index(one_token)
                                        tempt_reward = leaf_token_reward[leaf_token_index]
                                        tempt_count = leaf_token_count[leaf_token_index]
                                        leaf_token_reward[leaf_token_index] = tempt_reward + final_reward
                                        leaf_token_count[leaf_token_index] = tempt_count + 1
                    and_or_dict_list.append(and_or_dict)
                    count = count + 1
                    index = index + 2

        if self.group_by_clause:

            group_by_dict = defaultdict(list)

            group_by_dict["group_by_clause"] = ["group by"]

            group_by_dict["column_name"] = self.group_by_clause.replace("group by ", "").split(",")


            if group_by_dict["group_by_clause"] not in parse_node:
                parse_node.append(group_by_dict["group_by_clause"])
                parse_node.append(group_by_dict["column_name"])
            else:
                group_by_index = parse_node.index(group_by_dict["group_by_clause"])
                for one_column_name in group_by_dict["column_name"]:
                    if one_column_name not in parse_node[group_by_index + 1]:
                        parse_node[group_by_index + 1].append(one_column_name)
            if self.having_clause:

                for one_column_name in group_by_dict["column_name"]:
                    if [one_column_name, group_by_dict["group_by_clause"][0]] not in parse_connect:
                        parse_connect.append([one_column_name, group_by_dict["group_by_clause"][0]])
                    if ["having", one_column_name] not in parse_connect:
                        parse_connect.append(["having", one_column_name])
            else:
                if self.order_by_clause:

                    for one_column_name in group_by_dict["column_name"]:
                        if [one_column_name, group_by_dict["group_by_clause"][0]] not in parse_connect:
                            parse_connect.append([one_column_name, group_by_dict["group_by_clause"][0]])
                        if ["order by", one_column_name] not in parse_connect:
                            parse_connect.append(["order by", one_column_name])
                else:

                    for one_column_name in group_by_dict["column_name"]:
                        if [one_column_name, group_by_dict["group_by_clause"][0]] not in parse_connect:
                            parse_connect.append([one_column_name, group_by_dict["group_by_clause"][0]])

                    for one_token in group_by_dict["column_name"]:
                        if one_token not in leaf_token_name:
                            leaf_token_name.append(one_token)
                            leaf_token_reward.append(final_reward)
                            leaf_token_count.append(1)
                        else:
                            leaf_token_index = leaf_token_name.index(one_token)
                            tempt_reward = leaf_token_reward[leaf_token_index]
                            tempt_count = leaf_token_count[leaf_token_index]
                            leaf_token_reward[leaf_token_index] = tempt_reward + final_reward
                            leaf_token_count[leaf_token_index] = tempt_count + 1

        if self.having_clause:
            parts = re.split(r'\b(and|or)\b', self.having_clause)
            parts = [part.strip('; ') for part in parts]

            having_part = parts[0]
            having_dict = defaultdict(list)

            having_dict["having_clause"] = [having_part.split(" ")[0]]

            having_dict["statistic_clause"] = [having_part.split(" ")[1].split("(")[0]]

            having_dict["column_name"] = [having_part.split(" ")[1].split("(")[1][:-1]]

            having_dict["operator_clause"] = [having_part.split(" ")[2]]
            operator_index = having_part.find(having_dict["operator_clause"][0]) + len(
                having_dict["operator_clause"][0]) + 1

            having_dict["sampled_cell_value"] = [having_part[operator_index:]]


            if having_dict["having_clause"] not in parse_node:
                parse_node.append(having_dict["having_clause"])
                parse_node.append(having_dict["statistic_clause"])
                parse_node.append(having_dict["column_name"])
                parse_node.append(having_dict["operator_clause"])
                parse_node.append(having_dict["sampled_cell_value"])
            else:
                the_index = parse_node.index(having_dict["having_clause"])
                if having_dict["statistic_clause"][0] not in parse_node[the_index + 1]:
                    parse_node[the_index + 1].append(having_dict["statistic_clause"][0])
                if having_dict["column_name"][0] not in parse_node[the_index + 2]:
                    parse_node[the_index + 2].append(having_dict["column_name"][0])
                if having_dict["operator_clause"][0] not in parse_node[the_index + 3]:
                    parse_node[the_index + 3].append(having_dict["operator_clause"][0])
                if having_dict["sampled_cell_value"][0] not in parse_node[the_index + 4]:
                    parse_node[the_index + 4].append(having_dict["sampled_cell_value"][0])

            if len(parts) == 1:
                if self.order_by_clause:
                    if [having_dict["statistic_clause"][0], having_dict["having_clause"][0]] not in parse_connect:
                        parse_connect.append([having_dict["statistic_clause"][0], having_dict["having_clause"][0]])
                    if [having_dict["column_name"][0], having_dict["statistic_clause"][0]] not in parse_connect:
                        parse_connect.append([having_dict["column_name"][0], having_dict["statistic_clause"][0]])
                    if [having_dict["operator_clause"][0], having_dict["column_name"][0]] not in parse_connect:
                        parse_connect.append([having_dict["operator_clause"][0], having_dict["column_name"][0]])
                    if [having_dict["sampled_cell_value"][0], having_dict["operator_clause"][0]] not in parse_connect:
                        parse_connect.append([having_dict["sampled_cell_value"][0], having_dict["operator_clause"][0]])
                    if ["order by", having_dict["sampled_cell_value"][0]] not in parse_connect:
                        parse_connect.append(["order by", having_dict["sampled_cell_value"][0]])
                else:
                    if [having_dict["statistic_clause"][0], having_dict["having_clause"][0]] not in parse_connect:
                        parse_connect.append([having_dict["statistic_clause"][0], having_dict["having_clause"][0]])
                    if [having_dict["column_name"][0], having_dict["statistic_clause"][0]] not in parse_connect:
                        parse_connect.append([having_dict["column_name"][0], having_dict["statistic_clause"][0]])
                    if [having_dict["operator_clause"][0], having_dict["column_name"][0]] not in parse_connect:
                        parse_connect.append([having_dict["operator_clause"][0], having_dict["column_name"][0]])
                    if [having_dict["sampled_cell_value"][0], having_dict["operator_clause"][0]] not in parse_connect:
                        parse_connect.append([having_dict["sampled_cell_value"][0], having_dict["operator_clause"][0]])
                    for one_token in having_dict["sampled_cell_value"]:
                        if one_token not in leaf_token_name:
                            leaf_token_name.append(one_token)
                            leaf_token_reward.append(final_reward)
                            leaf_token_count.append(1)
                        else:
                            leaf_token_index = leaf_token_name.index(one_token)
                            tempt_reward = leaf_token_reward[leaf_token_index]
                            tempt_count = leaf_token_count[leaf_token_index]
                            leaf_token_reward[leaf_token_index] = tempt_reward + final_reward
                            leaf_token_count[leaf_token_index] = tempt_count + 1
            else:
                if [having_dict["statistic_clause"][0], having_dict["having_clause"][0]] not in parse_connect:
                    parse_connect.append([having_dict["statistic_clause"][0], having_dict["having_clause"][0]])
                if [having_dict["column_name"][0], having_dict["statistic_clause"][0]] not in parse_connect:
                    parse_connect.append([having_dict["column_name"][0], having_dict["statistic_clause"][0]])
                if [having_dict["operator_clause"][0], having_dict["column_name"][0]] not in parse_connect:
                    parse_connect.append([having_dict["operator_clause"][0], having_dict["column_name"][0]])
                if [having_dict["sampled_cell_value"][0], having_dict["operator_clause"][0]] not in parse_connect:
                    parse_connect.append([having_dict["sampled_cell_value"][0], having_dict["operator_clause"][0]])
                if [having_dict["having_clause"][0] + "_" + parts[1] + str(0),
                    having_dict["sampled_cell_value"][0]] not in parse_connect:
                    parse_connect.append([having_dict["having_clause"][0] + "_" + parts[1] + str(0),
                                          having_dict["sampled_cell_value"][0]])

            and_or_dict_list = []
            and_or_dict = defaultdict(list)
            if len(parts) > 1:
                index = 1
                count = 0
                while index < len(parts):
                    and_or_dict["and_or_clause"] = [having_dict["having_clause"][0] + "_" + parts[index]]
                    and_or_part = parts[index + 1]

                    and_or_dict["statistic_clause"] = [and_or_part.split(" ")[0].split("(")[0]]

                    and_or_dict["column_name"] = [and_or_part.split(" ")[0].split("(")[1][:-1]]

                    and_or_dict["operator_clause"] = [and_or_part.split(" ")[1]]

                    and_or_dict["sampled_cell_value"] = [and_or_part.split(" ")[2]]


                    having_index = parse_node.index(having_dict["having_clause"])
                    and_or_index = having_index + 4 + 1 + count * 5
                    if and_or_index >= len(parse_node):
                        parse_node.append(and_or_dict["and_or_clause"])
                        parse_node.append(and_or_dict["statistic_clause"])
                        parse_node.append(and_or_dict["column_name"])
                        parse_node.append(and_or_dict["operator_clause"])
                        parse_node.append(and_or_dict["sampled_cell_value"])
                    else:
                        if and_or_dict["and_or_clause"] not in parse_node[and_or_index]:
                            parse_node[and_or_index].append(and_or_dict["and_or_clause"])
                        if and_or_dict["statistic_clause"] not in parse_node[and_or_index + 1]:
                            parse_node[and_or_index + 1].append(and_or_dict["statistic_clause"])
                        if and_or_dict["column_name"] not in parse_node[and_or_index + 2]:
                            parse_node[and_or_index + 2].append(and_or_dict["column_name"])
                        if and_or_dict["operator_clause"] not in parse_node[and_or_index + 3]:
                            parse_node[and_or_index + 3].append(and_or_dict["operator_clause"])
                        if and_or_dict["sampled_cell_value"] not in parse_node[and_or_index + 4]:
                            parse_node[and_or_index + 4].append(and_or_dict["sampled_cell_value"])
                    if index + 2 < len(parts):
                        if [and_or_dict["statistic_clause"][0],
                            having_dict["having_clause"][0] + "_" + parts[index] + str(count)] not in parse_connect:
                            parse_connect.append([and_or_dict["statistic_clause"][0],
                                                  having_dict["having_clause"][0] + "_" + parts[index] + str(count)])
                        if [and_or_dict["column_name"][0], and_or_dict["statistic_clause"][0]] not in parse_connect:
                            parse_connect.append([and_or_dict["column_name"][0], and_or_dict["statistic_clause"][0]])
                        if [and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]] not in parse_connect:
                            parse_connect.append([and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]])
                        if [and_or_dict["sampled_cell_value"][0],
                            and_or_dict["operator_clause"][0]] not in parse_connect:
                            parse_connect.append(
                                [and_or_dict["sampled_cell_value"][0], and_or_dict["operator_clause"][0]])
                        if [having_dict["having_clause"][0] + "_" + parts[index + 2] + str(count + 1),
                            and_or_dict["sampled_cell_value"][0]] not in parse_connect:
                            parse_connect.append(
                                [having_dict["having_clause"][0] + "_" + parts[index + 2] + str(count + 1),
                                 and_or_dict["sampled_cell_value"][0]])
                    else:
                        if self.order_by_clause:
                            if [and_or_dict["statistic_clause"][0],
                                and_or_dict["and_or_clause"][0] + str(count)] not in parse_connect:
                                parse_connect.append(
                                    [and_or_dict["statistic_clause"][0], and_or_dict["and_or_clause"][0] + str(count)])
                            if [and_or_dict["column_name"][0], and_or_dict["statistic_clause"][0]] not in parse_connect:
                                parse_connect.append(
                                    [and_or_dict["column_name"][0], and_or_dict["statistic_clause"][0]])
                            if [and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]] not in parse_connect:
                                parse_connect.append([and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]])
                            if [and_or_dict["sampled_cell_value"][0],
                                and_or_dict["operator_clause"][0]] not in parse_connect:
                                parse_connect.append(
                                    [and_or_dict["sampled_cell_value"][0], and_or_dict["operator_clause"][0]])
                            if ["order by", and_or_dict["sampled_cell_value"][0]] not in parse_connect:
                                parse_connect.append(["order by", and_or_dict["sampled_cell_value"][0]])
                        else:
                            if [and_or_dict["statistic_clause"][0],
                                and_or_dict["and_or_clause"][0] + str(count)] not in parse_connect:
                                parse_connect.append(
                                    [and_or_dict["statistic_clause"][0], and_or_dict["and_or_clause"][0] + str(count)])
                            if [and_or_dict["column_name"][0], and_or_dict["statistic_clause"][0]] not in parse_connect:
                                parse_connect.append(
                                    [and_or_dict["column_name"][0], and_or_dict["statistic_clause"][0]])
                            if [and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]] not in parse_connect:
                                parse_connect.append([and_or_dict["operator_clause"][0], and_or_dict["column_name"][0]])
                            if [and_or_dict["sampled_cell_value"][0],
                                and_or_dict["operator_clause"][0]] not in parse_connect:
                                parse_connect.append(
                                    [and_or_dict["sampled_cell_value"][0], and_or_dict["operator_clause"][0]])
                            for one_token in and_or_dict["sampled_cell_value"]:
                                if one_token not in leaf_token_name:
                                    leaf_token_name.append(one_token)
                                    leaf_token_reward.append(final_reward)
                                    leaf_token_count.append(1)
                                else:
                                    leaf_token_index = leaf_token_name.index(one_token)
                                    tempt_reward = leaf_token_reward[leaf_token_index]
                                    tempt_count = leaf_token_count[leaf_token_index]
                                    leaf_token_reward[leaf_token_index] = tempt_reward + final_reward
                                    leaf_token_count[leaf_token_index] = tempt_count + 1
                    and_or_dict_list.append(and_or_dict)
                    count = count + 1
                    index = index + 2

        if self.order_by_clause:

            order_by_dict = defaultdict(list)

            order_by_dict["order_by_clause"] = ["order by"]
            order_by_part = self.order_by_clause[:-1].replace("order by ", "").split(",")
            total_statistic_clause = []
            total_column_name = []
            total_sorting_words = []

            for one_order_by_part in order_by_part:
                if "(" in one_order_by_part:
                    statistic_clause = one_order_by_part.split("(")[0]
                    column_name = one_order_by_part.split("(")[1].split(" ")[0][:-1]
                    sorting_words = one_order_by_part.split("(")[1].split(" ")[-1]
                    total_statistic_clause.append(statistic_clause)
                    total_column_name.append(column_name)
                    total_sorting_words.append(sorting_words)
                    if [statistic_clause, order_by_dict["order_by_clause"][0]] not in parse_connect:
                        parse_connect.append([statistic_clause, order_by_dict["order_by_clause"][0]])
                    if [column_name, statistic_clause] not in parse_connect:
                        parse_connect.append([column_name, statistic_clause])
                    if [sorting_words, column_name] not in parse_connect:
                        parse_connect.append([sorting_words, column_name])
                else:
                    column_name = one_order_by_part.split(" ")[0]
                    sorting_words = one_order_by_part.split(" ")[1]
                    total_column_name.append(column_name)
                    total_sorting_words.append(sorting_words)
                    if [column_name, order_by_dict["order_by_clause"][0]] not in parse_connect:
                        parse_connect.append([column_name, order_by_dict["order_by_clause"][0]])
                    if [sorting_words, column_name] not in parse_connect:
                        parse_connect.append([sorting_words, column_name])

            if order_by_dict["order_by_clause"] not in parse_node:
                parse_node.append(order_by_dict["order_by_clause"])
                if len(total_statistic_clause) != 0:
                    self.order_by_statistic_flag = 1
                    parse_node.append(total_statistic_clause)
                parse_node.append(total_column_name)
                parse_node.append(total_sorting_words)
            else:
                if self.order_by_statistic_flag == 1:
                    order_by_index = parse_node.index(order_by_dict["order_by_clause"])
                    if len(total_statistic_clause) != 0:
                        for one_statistic_clause in total_statistic_clause:
                            if one_statistic_clause not in parse_node[order_by_index + 1]:
                                parse_node[order_by_index + 1].append(one_statistic_clause)
                        for one_column_name in total_column_name:
                            if one_column_name not in parse_node[order_by_index + 2]:
                                parse_node[order_by_index + 2].append(one_column_name)
                        for one_sorting_words in total_sorting_words:
                            if one_sorting_words not in parse_node[order_by_index + 3]:
                                parse_node[order_by_index + 3].append(one_sorting_words)
                else:
                    order_by_index = parse_node.index(order_by_dict["order_by_clause"])
                    if len(total_statistic_clause) != 0:
                        self.order_by_statistic_flag = 1
                        parse_node.insert(order_by_index + 1, total_statistic_clause)
                        for one_column_name in total_column_name:
                            if one_column_name not in parse_node[order_by_index + 2]:
                                parse_node[order_by_index + 2].append(one_column_name)
                        for one_sorting_words in total_sorting_words:
                            if one_sorting_words not in parse_node[order_by_index + 3]:
                                parse_node[order_by_index + 3].append(one_sorting_words)
                    else:
                        for one_column_name in total_column_name:
                            if one_column_name not in parse_node[order_by_index + 1]:
                                parse_node[order_by_index + 1].append(one_column_name)
                        for one_sorting_words in total_sorting_words:
                            if one_sorting_words not in parse_node[order_by_index + 2]:
                                parse_node[order_by_index + 2].append(one_sorting_words)
            order_by_dict["statistic_clause"] = total_statistic_clause
            order_by_dict["column_name"] = total_column_name
            order_by_dict["sorting_words"] = total_sorting_words

            for one_token in total_sorting_words:
                if one_token not in leaf_token_name:
                    leaf_token_name.append(one_token)
                    leaf_token_reward.append(final_reward)
                    leaf_token_count.append(1)
                else:
                    leaf_token_index = leaf_token_name.index(one_token)
                    tempt_reward = leaf_token_reward[leaf_token_index]
                    tempt_count = leaf_token_count[leaf_token_index]
                    leaf_token_reward[leaf_token_index] = tempt_reward + final_reward
                    leaf_token_count[leaf_token_index] = tempt_count + 1

    def get_slow_reward(self):

        inject_sql = self.get_sql()
        is_trivial_seed, trivial_reason = is_trivial_single_table_scan_seed(
            inject_sql,
            getattr(self, "summary", None),
        )






        s_t = time.time()


        res = self.get_run_time()

        e_t = time.time()

        self.slow_time = self.slow_time + e_t - s_t

        runtime = 0

        if res is None:
            self.unused_episode = self.unused_episode + 1
            self.last_query_success = False
            self.last_runtime_status = "error"
            self.last_runtime = 0.0
            return 0, self.bug_reward

        if self.slow_flag == 1 or res["status"] == "timeout":
            self.slow_flag = 0
            runtime = float(res["runtime"])
            self.last_query_success = False
            self.last_runtime_status = "timeout"
            self.last_runtime = runtime
            sql_path = os.path.abspath('.')
            with open(sql_path + '/sql_templates_tempt.log', 'a', encoding='utf-8') as f:
                f.write(inject_sql + '\t' + str(runtime) + '\n')
            timeout_penalty = -8.0 - max(0.0, runtime - self.accept_max) / max(self.need_target, 1e-6)
            self.unused_episode = self.unused_episode + 1
            return 0, timeout_penalty
        elif res["status"] == "error":
            self.unused_episode = self.unused_episode + 1
            self.last_query_success = False
            self.last_runtime_status = "error"
            self.last_runtime = 0.0
            return 0, self.bug_reward
        else:
            runtime = float(res["runtime"])
            self.last_runtime_status = "ok"
            self.last_runtime = runtime


        sql_path = os.path.abspath('.')
        with open(sql_path + '/sql_templates_tempt.log', 'a', encoding='utf-8') as f:
            f.write(inject_sql + '\t' + str(runtime) + '\n')




        final_reward = 0
        gap = abs(runtime - self.need_target)
        if self.accept_min <= runtime <= self.accept_max:
            radius = max(self.need_target - self.accept_min, self.accept_max - self.need_target, 1e-6)
            normalized_gap = min(1.0, gap / radius)
            final_reward = 5.0 - 4.0 * (normalized_gap ** 2)
            self.last_query_success = True
        else:
            self.unused_episode = self.unused_episode + 1
            if runtime < self.accept_min:
                bad_gap = self.accept_min - runtime
                normalized_gap = bad_gap / max(self.accept_min, 1e-6)
                final_reward = -1.0 - 4.0 * normalized_gap
            else:
                bad_gap = runtime - self.accept_max
                normalized_gap = bad_gap / max(self.need_target, 1e-6)
                final_reward = -3.0 - 6.0 * normalized_gap
            final_reward = max(-10.0, final_reward)
            self.last_query_success = False

        if is_trivial_seed:
            self.unused_episode = self.unused_episode + 1
            self.last_query_success = False
            final_reward = min(final_reward, -6.0)
            self.last_runtime_status = f"trivial_seed:{trivial_reason}"


        try:
            self.get_sql_tree(final_reward)
        except Exception as e:
            pass

        if self.last_query_success:
            return 1, final_reward
        return 0, final_reward


    def final_reward(self):
        _, reward = self.get_slow_reward()
        return reward

    def __del__(self):
        self.cursor.close()
        self.db.close()

    def reset_token_reward(self):
        for i in range(len(leaf_token_name)):
            self.total_token_reward[leaf_token_name[i]] = 1.0 * leaf_token_reward[i] / leaf_token_count[i]

        for j in range(len(parse_node) - 1, -1, -1):
            for k in range(len(parse_node[j])):
                count = 0.0000001
                reward = 0

                node_value = parse_node[j][k][0] if isinstance(parse_node[j][k], list) else parse_node[j][k]
                if node_value not in self.total_token_reward.keys():
                    for one_connect in parse_connect:
                        if one_connect[1] == node_value and one_connect[0] in self.total_token_reward.keys():
                            reward = reward + self.total_token_reward[one_connect[0]]
                            count = count + 1




                    self.total_token_reward[node_value] = 1.0 * reward / count


class ActorCriticModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, x, hidden=None):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        gru_out, hidden = self.gru(x, hidden)

        logits = self.actor(gru_out)

        value = self.critic(gru_out)

        return logits.squeeze(), value.squeeze(), hidden


class ActorCriticAgent:
    def __init__(self, input_size, hidden_size, output_size,
                 lr=0.00001, gamma=0.5, grad_clip=1.5):
        self.model = ActorCriticModel(input_size, hidden_size, output_size).to(device)
        self.gamma = gamma
        self.grad_clip = grad_clip

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        self.hidden = None

        self.entropy_coef = 0.35

    def reset_hidden(self):
        self.hidden = None

    def build_valid_mask(self, observation):
        candidate_list = np.argwhere(observation == np.max(observation)).flatten()
        valid_mask = np.zeros_like(observation, dtype=bool)
        valid_mask[candidate_list] = True
        return candidate_list, valid_mask

    def choose_action(self, observation):

        observation = np.array(observation, dtype=np.float32) if not isinstance(observation,
                                                                                np.ndarray) else observation

        candidate_list, valid_mask = self.build_valid_mask(observation)

        if np.all(observation == 0):
            return None

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits, _, self.hidden = self.model(obs_tensor, self.hidden)

        masked_logits = logits.clone()
        masked_logits[..., ~valid_mask] = -float('inf')

        probs = F.softmax(masked_logits, dim=-1)

        if len(candidate_list) == 0:
            return None

        try:
            valid_probs = probs[..., candidate_list]
            valid_probs /= valid_probs.sum()
            selected_index = torch.multinomial(valid_probs, 1).item()
            action = candidate_list[selected_index]
        except Exception as e:
            action = candidate_list[torch.argmax(probs[candidate_list]).item()]

        if action not in candidate_list:
            action = candidate_list[probs[candidate_list].argmax().item()]

        return action

    def update(self, states, actions, rewards, next_states, action_mask):

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        action_mask = np.array(action_mask, dtype=bool)

        if states.ndim == 1:
            states = states.reshape(1, -1)
            next_states = next_states.reshape(1, -1)
        if action_mask.ndim == 1:
            action_mask = action_mask.reshape(1, -1)
        if isinstance(actions, int):
            actions = [actions]

        states_tensor = torch.as_tensor(states, device=device, dtype=torch.float32)
        next_states_tensor = torch.as_tensor(next_states, device=device, dtype=torch.float32)
        rewards_tensor = torch.as_tensor(rewards, device=device, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, device=device, dtype=torch.long)
        action_mask_tensor = torch.as_tensor(action_mask, device=device, dtype=torch.bool)

        logits, values, _ = self.model(states_tensor)
        _, next_values, _ = self.model(next_states_tensor)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.squeeze(1)

        masked_logits = logits.masked_fill(~action_mask_tensor, -float('inf'))
        probs = F.softmax(masked_logits, dim=-1)
        actions_tensor = actions_tensor.view(-1, 1)

        try:
            selected_probs = probs.gather(1, actions_tensor).squeeze()
        except Exception as e:
            selected_probs = probs.mean(dim=-1) + 1e-8

        advantages = rewards_tensor + self.gamma * next_values.detach() - values.detach()
        policy_loss = (-torch.log(selected_probs + 1e-5) * advantages).mean()
        value_loss = F.mse_loss(values, rewards_tensor + self.gamma * next_values.detach())
        safe_probs = torch.where(action_mask_tensor, probs, torch.zeros_like(probs))
        entropy = -(safe_probs * torch.log(safe_probs + 1e-5)).sum(dim=-1).mean()
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, foreach=True)
        self.optimizer.step()

        return total_loss.item()

class SeedGeneratorSession:

    def __init__(
        self,
        dbname,
        target,
        accept_min_ratio,
        accept_max_ratio,
        timeout_ratio,
        from_mode,
        max_steps,
        *,
        emit_output=True,
        write_output=True,
        path_constraint: PathConstraint | None = None,
        summary: dict[str, Any] | None = None,
        shared_agent=None,
    ):
        self.dbname = dbname
        self.target = target
        self.emit_output = emit_output
        self.write_output = write_output
        self.path_constraint = path_constraint
        self.summary = summary
        self.env = GenSqlEnv(
            100000,
            dbname,
            target,
            accept_min_ratio,
            accept_max_ratio,
            timeout_ratio,
            from_mode,
            max_steps,
            path_constraint=path_constraint,
            summary=summary,
        )
        input_size = self.env.action_space
        hidden_size = 128
        output_size = self.env.action_space
        self.agent = shared_agent or ActorCriticAgent(input_size, hidden_size, output_size)
        self.episode = 0
        self.agent_time = 0.0
        self.gen_sql_time = 0.0
        self.rejected_by_constraint = 0
        self.total_generation_attempts = 0

    def _write_success_sql(self, sql: str) -> None:

        if not self.write_output:
            return
        sql_path = os.path.abspath('.')
        with open(sql_path + f'/RLSQL_{self.dbname}_pg_{self.target}.log', 'a', encoding='utf-8') as file_obj:
            file_obj.write(sql + '\n')

    def _print_success(self, sql: str) -> None:

        if not self.emit_output:
            return
        print(sql)
        print(f"runtime={self.env.last_runtime:.6f}s")
        print(f"episode={self.episode}")
        print(f"unused_episode={self.env.unused_episode}")

    def _print_rejected(self, sql: str, flag: int) -> None:

        if not self.emit_output or flag == 1:
            return
        print(sql)
        print(f"runtime={self.env.last_runtime:.6f}s")

    def _matches_path_constraint(self, sql: str) -> tuple[bool, str]:

        if self.path_constraint is None or self.summary is None:
            return True, "ok"
        return sql_matches_constraint(sql, self.summary, self.path_constraint)

    def _matches_seed_quality(self, sql: str) -> tuple[bool, str]:

        is_trivial_seed, reason = is_trivial_single_table_scan_seed(sql, self.summary)
        if is_trivial_seed:
            return False, reason
        return True, "ok"

    def generate_next_seed(self, max_attempts: int | None = None):

        attempts = 0
        while max_attempts is None or attempts < max_attempts:
            attempts += 1
            self.total_generation_attempts += 1
            flag = 0
            env = self.env
            agent = self.agent

            current_action = env.reset()
            current_state = env.encode_one_hot(current_action)
            reward, done = env.bug_reward, False
            ep_steps = 0
            total_reward = 0
            and_or_count = 0
            having_and_or_count = 0
            having_and_or_flag = 0
            env.reset_token_reward()

            while not done:
                if ep_steps >= env.max_steps:
                    flag = 1
                    break

                current_observation = env.observe(current_action)
                action = agent.choose_action(current_observation)

                if action is None:
                    flag = 1
                    break

                tempt_token = env.num_word_map[action]
                tempt_reward = 0.0
                if env.total_token_reward:
                    if env.num_word_map[action] == "having":
                        having_and_or_flag = 1
                    if env.num_word_map[action] == "and" or env.num_word_map[action] == "or":
                        if having_and_or_flag == 1:
                            tempt_token = "having_" + env.num_word_map[action] + str(having_and_or_count)
                            having_and_or_count = having_and_or_count + 1
                        else:
                            tempt_token = env.num_word_map[action] + str(and_or_count)
                            and_or_count = and_or_count + 1
                    if tempt_token in env.total_token_reward.keys():
                        tempt_reward = env.total_token_reward[tempt_token]
                    env.total_reward = env.total_reward + tempt_reward
                    env.reward_count = env.reward_count + 1
                    env.step_reward = 1.0 * env.total_reward / env.reward_count

                g_s_t = time.time()
                reward, done = env.step(action)
                g_e_t = time.time()
                self.gen_sql_time = self.gen_sql_time + g_e_t - g_s_t

                next_state = current_state.copy()
                next_state[action] = 1

                s_t = time.time()
                agent.update(current_state, action, reward, next_state, current_observation)
                e_t = time.time()
                self.agent_time = self.agent_time + e_t - s_t

                current_state = next_state
                current_action = action
                total_reward += reward
                ep_steps += 1

            sql = env.get_sql()
            if (not env.last_query_success) or flag == 1:
                sql_path = os.path.abspath('.')
                self._print_rejected(sql, flag)
                with open(sql_path + '/bad_sql.log', 'a', encoding='utf-8') as file_obj:
                    file_obj.write(sql + '\n')
                continue

            self.episode += 1
            matches_constraint, constraint_reason = self._matches_path_constraint(sql)
            if not matches_constraint:
                self.rejected_by_constraint += 1
                if self.emit_output:
                    print(sql)
                    print(f"runtime={self.env.last_runtime:.6f}s")
                    print(f"rejected_by_constraint={constraint_reason}")
                continue
            matches_quality, quality_reason = self._matches_seed_quality(sql)
            if not matches_quality:
                self.rejected_by_constraint += 1
                if self.emit_output:
                    print(sql)
                    print(f"runtime={self.env.last_runtime:.6f}s")
                    print(f"rejected_by_seed_quality={quality_reason}")
                continue
            self._write_success_sql(sql)
            self._print_success(sql)
            return {
                "sql": sql,
                "runtime": env.last_runtime,
                "episode": self.episode,
                "unused_episode": env.unused_episode,
                "attempts": attempts,
                "constraint_rejections": self.rejected_by_constraint,
                "path_id": self.path_constraint.path_id if self.path_constraint else "",
                "path_name": self.path_constraint.path_name if self.path_constraint else "",
            }
        return None

    def generate_many(self, max_episodes):

        results = []
        while len(results) < max_episodes:
            result = self.generate_next_seed()
            if result is None:
                break
            results.append(result)
        return results


def test_generate(
    dbname,
    max_episodes,
    target,
    accept_min_ratio,
    accept_max_ratio,
    timeout_ratio,
    from_mode,
    max_steps,
    collect_results=False,
    emit_output=True,
    write_output=True,
):

    session = SeedGeneratorSession(
        dbname=dbname,
        target=target,
        accept_min_ratio=accept_min_ratio,
        accept_max_ratio=accept_max_ratio,
        timeout_ratio=timeout_ratio,
        from_mode=from_mode,
        max_steps=max_steps,
        emit_output=emit_output,
        write_output=write_output,
    )
    results = session.generate_many(max_episodes)
    if collect_results:
        return results


if __name__ == '__main__':
    seed_settings = load_settings().seed_generator
    start_time = time.perf_counter()
    reset_output_files(seed_settings.dbname, seed_settings.target_seconds)
    test_generate(
        seed_settings.dbname,
        seed_settings.sql_count,
        seed_settings.target_seconds,
        seed_settings.accept_min_ratio,
        seed_settings.accept_max_ratio,
        seed_settings.timeout_ratio,
        seed_settings.from_mode,
        seed_settings.max_steps,
    )
    total_runtime = time.perf_counter() - start_time
    print(f"total_runtime={total_runtime:.6f}s")
