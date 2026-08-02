"""
Построение и визуализация байесовской сети из JSON-ответа LLM.

LLM возвращает описание сети (variables/edges/cpd) — см. промпт
app/prompts/bayesian_network_building. Здесь это описание превращается
в настоящую модель pgmpy (DiscreteBayesianNetwork), считается апостериорная
вероятность целевой переменной и рисуется интерактивный граф (plotly).

Требование к JSON для сборки модели:
- у каждой переменной ровно 3 состояния;
- CPD полные: без родителей — 1 строка (априорная вероятность),
  с k родителями — ровно 3^k строк, по одной на комбинацию состояний
  родителей; первый родитель меняется медленнее всех, последний — быстрее
  (такой же порядок колонок у pgmpy).
"""

import itertools
import logging
from typing import Optional

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

logger = logging.getLogger(__name__)

# Палитра по умолчанию (используется, если фронт не передал свою).
_DEFAULT_PALETTE = {
    "surface": "#ffffff", "grid": "#e1e0d9", "axis": "#898781",
    "ink": "#52514e", "sma20": "#2a78d6", "sma50": "#eb6834",
    "rsi": "#1baf7a", "up": "#0ca30c", "down": "#d03b3b", "muted": "#c3c2b7",
}


# --- Сборка модели pgmpy ---

def _normalize_cpd(name: str, states: list, parents: list, raw_cpd) -> tuple[list, list[str]]:
    """
    Приводит сырой cpd от LLM к полной таблице (n_states, expected_rows).

    LLM часто даёт неполный CPD: лишние/недостающие строки, строки не той
    длины, суммы != 1. Всё это чинится здесь, чтобы pgmpy смог построить модель.
    Если CPD идеальный — нормализация ничего не меняет.

    Returns:
        (values: list[list[float]], warnings: list[str])
    """
    n_states = len(states)
    expected_rows = 1 if not parents else 3 ** len(parents)
    warnings: list[str] = []

    rows: list[list[float]] = []
    for row in raw_cpd:
        if isinstance(row, (list, tuple)):
            nums = [
                float(x) for x in row
                if isinstance(x, (int, float)) and not isinstance(x, bool)
            ]
            if nums:
                rows.append(nums)

    if not rows:
        warnings.append(f"'{name}': cpd пуст — использовано равномерное распределение")
        rows = [[1.0 / n_states] * n_states]

    # Нормализуем длину и сумму каждой строки
    fixed: list[list[float]] = []
    for row in rows:
        if len(row) > n_states:
            row = row[:n_states]
        elif len(row) < n_states:
            row = list(row) + [0.0] * (n_states - len(row))
        s = sum(row)
        if s > 0:
            row = [x / s for x in row]
        else:
            row = [1.0 / n_states] * n_states
        fixed.append(row)

    # Доводим число строк до expected_rows (полная таблица для pgmpy)
    if len(fixed) > expected_rows:
        if len(fixed) != expected_rows:
            warnings.append(
                f"'{name}': лишние строки CPD — оставлены первые {expected_rows} из {len(fixed)}"
            )
        fixed = fixed[:expected_rows]
    elif len(fixed) < expected_rows:
        warnings.append(
            f"'{name}': CPD неполный ({len(fixed)} из {expected_rows} строк) — "
            f"недостающие комбинации заполнены повтором последней строки"
        )
        last = fixed[-1]
        while len(fixed) < expected_rows:
            fixed.append(last[:])

    # Транспонируем: LLM даёт строки = комбинации родителей, pgmpy ждёт
    # колонки (значения раскладываются в тензор var_card x cards...).
    return [
        [fixed[combo][state] for combo in range(expected_rows)]
        for state in range(n_states)
    ], warnings


def build_model_from_json(data: dict):
    """
    Собирает pgmpy DiscreteBayesianNetwork из JSON-описания сети.

    Терпим к «грязному» выводу LLM: неполные CPD нормализуются (см.
    _normalize_cpd), рёбра и родители на неизвестные переменные отбрасываются.
    Обо всех исправлениях сообщается в warnings.

    Args:
        data: словарь {variables, edges, target_variable, explanation}.

    Returns:
        (model, warnings)

    Raises:
        ValueError: если данные вообще не похожи на сеть (нет variables,
            нет имён, cpd не является списком, нет 3 состояний).
    """
    warnings: list[str] = []

    if not isinstance(data, dict):
        raise ValueError("JSON сети не является объектом")
    raw_variables = data.get("variables")
    if not isinstance(raw_variables, list) or not raw_variables:
        raise ValueError("JSON не содержит списка variables")

    variables: dict[str, dict] = {}
    for v in raw_variables:
        if isinstance(v, dict) and v.get("name"):
            variables[v["name"]] = v

    if not variables:
        raise ValueError("Список variables пуст или не содержит имён")

    known = set(variables.keys())

    # Рёбра из edges + из полей parents (страховка от расхождения LLM).
    # Ссылки на неизвестные переменные отбрасываем и предупреждаем.
    edge_set: set[tuple[str, str]] = set()
    for e in data.get("edges", []):
        if isinstance(e, (list, tuple)) and len(e) == 2:
            if str(e[0]) in known and str(e[1]) in known:
                edge_set.add((str(e[0]), str(e[1])))
            else:
                warnings.append(f"Ребро {e} ссылается на неизвестные переменные — исключено")
    for name, var in variables.items():
        for p in var.get("parents", []):
            if isinstance(p, str) and p in known:
                edge_set.add((p, name))

    model = DiscreteBayesianNetwork(list(edge_set))
    # Добавляем все переменные как узлы (в т.ч. изолированные, без рёбер),
    # иначе pgmpy отвергнет их CPD в add_cpds.
    model.add_nodes_from(variables.keys())

    # Строим CPD. Родителей берём из графа (предшественники узла), чтобы
    # evidence CPD был консистентен с рёбрами и check_model() прошёл.
    cpds = []
    for name in model.nodes():
        var = variables[name]
        states = var.get("states")
        cpd = var.get("cpd")

        if not isinstance(states, list) or len(states) != 3:
            raise ValueError(f"'{name}': нужно ровно 3 состояния, получено {states!r}")
        if not isinstance(cpd, list) or not cpd:
            raise ValueError(f"'{name}': cpd не задан")

        parents = list(model.predecessors(name))
        values, cpd_warnings = _normalize_cpd(name, states, parents, cpd)
        warnings.extend(cpd_warnings)

        state_names = {name: states}
        evidence_card = []
        for p in parents:
            state_names[p] = variables[p].get("states")
            evidence_card.append(len(state_names[p]))

        kwargs: dict = {
            "variable": name,
            "variable_card": len(states),
            "values": values,
            "state_names": state_names,
        }
        if parents:
            kwargs["evidence"] = parents
            kwargs["evidence_card"] = evidence_card

        cpds.append(TabularCPD(**kwargs))

    model.add_cpds(*cpds)
    if not model.check_model():
        warnings.append("Модель не прошла внутреннюю проверку pgmpy — вероятности могут быть рассогласованы")

    return model, warnings


# --- Инференс ---

def infer_target(model, target: str = "Price_Change"):
    """
    Апостериорная вероятность целевой переменной (VariableElimination).

    Returns:
        (states: list[str], probabilities: dict[str, float])
    """
    from pgmpy.inference import VariableElimination

    inf = VariableElimination(model)
    factor = inf.query([target])
    states = list(factor.state_names[target])
    values = factor.values.tolist()
    return states, {s: round(float(v), 4) for s, v in zip(states, values)}


def action_from_probs(probs: dict) -> tuple[str, str]:
    """
    Торговое действие по апостериорным вероятностям целевой переменной.

    Берём наиболее вероятное состояние (argmax), а не попарное сравнение
    Up vs Down. Иначе при почти нейтральном распределении (Neutral — мода)
    выдаётся ложный направленный сигнал, хотя сеть «не уверена» в сторону.

    Returns:
        (action, best_state): ('LONG'|'NEUTRAL'|'SHORT', 'Up'|'Neutral'|'Down')
    """
    best_state = max(
        ("Down", probs.get("Down", 0.0)),
        ("Neutral", probs.get("Neutral", 0.0)),
        ("Up", probs.get("Up", 0.0)),
        key=lambda kv: kv[1],
    )[0]
    action = {"Down": "SHORT", "Neutral": "NEUTRAL", "Up": "LONG"}[best_state]
    return action, best_state


# --- Таблицы CPD ---

def cpd_tables(model):
    """
    CPD всех переменных модели в виде (имя_переменной, pandas.DataFrame).

    Строки DataFrame — комбинации состояний родителей (для переменных без
    родителей — одна строка), колонки — состояния переменной.
    """
    tables = []
    for cpd in model.get_cpds():
        var = cpd.variable
        var_states = list(cpd.state_names[var])
        evidence = [v for v in cpd.variables if v != var]

        # Колонки CPD идут в порядке product(evidence): первый родитель — самый
        # медленный. Переставляем тензор в (var_card, n_combos) и транспонируем.
        vals = cpd.values.reshape(len(var_states), -1).T

        if evidence:
            combos = itertools.product(*[cpd.state_names[e] for e in evidence])
            index = [
                ", ".join(f"{e}={s}" for e, s in zip(evidence, combo))
                for combo in combos
            ]
            index_name = " / ".join(evidence)
        else:
            index = ["—"]
            index_name = "prior"

        df = pd.DataFrame(vals, index=index, columns=var_states)
        df.index.name = index_name
        tables.append((var, df))
    return tables


# --- Граф (plotly) ---

def _layered_layout(G):
    """
    Слоистый layout для DAG: слой узла = длина самого длинного пути из корней.
    Родители выше детей, направления читаются сверху вниз.
    """
    try:
        order = list(nx.topological_sort(G))
    except Exception:
        return None
    layer: dict = {}
    for n in order:
        parents = list(G.predecessors(n))
        layer[n] = 0 if not parents else max(layer[p] for p in parents) + 1
    groups: dict[int, list] = {}
    for n, l in layer.items():
        groups.setdefault(l, []).append(n)
    max_layer = max(groups.keys(), default=0)
    pos = {}
    for l, nodes in groups.items():
        y = float(max_layer - l)
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i + 1) / (n + 1)
            pos[node] = (x, y)
    return pos


def build_figure(data: dict, palette: Optional[dict] = None) -> go.Figure:
    """
    Интерактивный граф сети: узлы — переменные, стрелки — рёбра.
    Целевая переменная подсвечена.

    Для layout используем networkx (слоистый для DAG, иначе spring).
    """
    p = {**_DEFAULT_PALETTE, **(palette or {})}

    variables = data.get("variables") or []
    edges = [
        (str(e[0]), str(e[1]))
        for e in data.get("edges", [])
        if isinstance(e, (list, tuple)) and len(e) == 2
    ]
    target = data.get("target_variable", "Price_Change")

    G = nx.DiGraph()
    G.add_nodes_from(v["name"] for v in variables if isinstance(v, dict) and v.get("name"))
    G.add_edges_from(edges)

    pos = _layered_layout(G)
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=0.6)

    fig = go.Figure()

    # Рёбра: стрелки через annotations (arrowhead рисует направление)
    for p_src, p_dst in edges:
        if p_src not in pos or p_dst not in pos:
            continue
        x0, y0 = pos[p_src]
        x1, y1 = pos[p_dst]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.4,
            arrowcolor=p["muted"], text="", opacity=0.8,
        )

    # Узлы
    node_colors = [
        p["up"] if name == target else p["sma20"]
        for name in G.nodes()
    ]
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=list(G.nodes()),
        textposition="bottom center",
        textfont=dict(color=p["ink"], size=13),
        marker=dict(size=38, color=node_colors, line=dict(color=p["ink"], width=1.2),
                    opacity=0.9),
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Байесовская сеть (целевая: {target})",
                   font=dict(color=p["ink"])),
        height=520,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["ink"]),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, range=[-0.08, 1.08])
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False, range=[-0.35, None])
    return fig
