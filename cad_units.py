"""CAD 参数尺寸图的单位迁移组件（仅依赖 Python 标准库）。

模型为 JSON 可序列化的 dict：
- "units": {"length": "mm"|"m", "angle": "deg"|"rad"}  基础单位
- "nodes": {name: node}  具名节点组成的有向无环图，允许前向引用
- "features": {slot: node_name}  特征尺寸槽引用节点

节点类型：
- {"type": "input", "value": float, "dim": [a, b]?}   独立输入，dim 可缺失
- {"type": "const", "value": float, "dim": [a, b]}     常量，dim 必须标注
- {"type": "add"|"sub"|"mul"|"div", "args": [l, r]}    二元运算
- {"type": "pow", "arg": name, "exp": int}             整数常量次幂
- {"type": "sin"|"cos", "arg": name}                   三角函数

量纲 dim 为整数二元组 [长度指数, 角度指数]。
"""

from __future__ import annotations

import json
import math
from fractions import Fraction

LENGTH_SLOTS = ("extrude.depth", "fillet.radius")
ANGLE_SLOTS = ("revolve.angle",)
SLOT_DIMS = {
    "extrude.depth": (1, 0),
    "fillet.radius": (1, 0),
    "revolve.angle": (0, 1),
}

SOURCE_UNITS = {"length": "mm", "angle": "deg"}
TARGET_UNITS = {"length": "m", "angle": "rad"}


class ModelError(ValueError):
    """模型结构非法或量纲约束不可解。"""


def _check_dim(dim, ctx):
    if (
        not isinstance(dim, (list, tuple))
        or len(dim) != 2
        or any(not isinstance(d, int) or isinstance(d, bool) for d in dim)
    ):
        raise ModelError(f"{ctx}: dim 必须是整数二元组，得到 {dim!r}")
    return (dim[0], dim[1])


def _node_refs(node):
    """返回节点直接引用的其它节点名。"""
    t = node["type"]
    if t in ("add", "sub", "mul", "div"):
        return list(node["args"])
    if t in ("pow", "sin", "cos"):
        return [node["arg"]]
    return []


def validate_model(model: dict) -> None:
    """校验模型结构：节点类型、引用存在、无环、所有节点可达尺寸槽。"""
    if not isinstance(model, dict):
        raise ModelError("模型必须是 dict")
    units = model.get("units")
    if not isinstance(units, dict) or "length" not in units or "angle" not in units:
        raise ModelError("模型缺少 units.length / units.angle")
    nodes = model.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ModelError("模型缺少非空 nodes")
    features = model.get("features")
    if not isinstance(features, dict) or not features:
        raise ModelError("模型缺少非空 features")

    for name, node in nodes.items():
        if not isinstance(name, str) or not name:
            raise ModelError(f"非法节点名 {name!r}")
        if not isinstance(node, dict):
            raise ModelError(f"节点 {name}: 必须是 dict")
        t = node.get("type")
        if t == "input":
            if not isinstance(node.get("value"), (int, float)):
                raise ModelError(f"节点 {name}: input 需要数值 value")
            if "dim" in node:
                _check_dim(node["dim"], f"节点 {name}")
        elif t == "const":
            if not isinstance(node.get("value"), (int, float)):
                raise ModelError(f"节点 {name}: const 需要数值 value")
            if "dim" not in node:
                raise ModelError(f"节点 {name}: const 必须标注 dim")
            _check_dim(node["dim"], f"节点 {name}")
        elif t in ("add", "sub", "mul", "div"):
            args = node.get("args")
            if not isinstance(args, (list, tuple)) or len(args) != 2:
                raise ModelError(f"节点 {name}: {t} 需要两个参数")
        elif t == "pow":
            exp = node.get("exp")
            if not isinstance(exp, int) or isinstance(exp, bool):
                raise ModelError(f"节点 {name}: pow 的 exp 必须是整数")
            if "arg" not in node:
                raise ModelError(f"节点 {name}: pow 缺少 arg")
        elif t in ("sin", "cos"):
            if "arg" not in node:
                raise ModelError(f"节点 {name}: {t} 缺少 arg")
        else:
            raise ModelError(f"节点 {name}: 未知类型 {t!r}")
        for ref in _node_refs(node):
            if ref not in nodes:
                raise ModelError(f"节点 {name}: 引用了不存在的节点 {ref!r}")

    for slot, target in features.items():
        if slot not in SLOT_DIMS:
            raise ModelError(f"未知尺寸槽 {slot!r}")
        if target not in nodes:
            raise ModelError(f"尺寸槽 {slot}: 引用了不存在的节点 {target!r}")

    topo_order(model)

    # 所有节点必须能沿依赖方向到达某个尺寸槽，
    # 即每个节点都是某个尺寸槽节点的祖先（经 args 边可达）。
    reachable = set()
    stack = list(features.values())
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        stack.extend(_node_refs(nodes[cur]))
    unreachable = set(nodes) - reachable
    if unreachable:
        raise ModelError(f"节点不可达任何尺寸槽: {sorted(unreachable)}")


def topo_order(model: dict) -> list:
    """按依赖先序返回节点名列表；有环时抛 ModelError。"""
    nodes = model["nodes"]
    order = []
    state = {}  # 0=未访问 1=访问中 2=完成

    def visit(name):
        s = state.get(name, 0)
        if s == 1:
            raise ModelError(f"节点图存在环，涉及 {name!r}")
        if s == 2:
            return
        state[name] = 1
        for ref in _node_refs(nodes[name]):
            visit(ref)
        state[name] = 2
        order.append(name)

    for name in nodes:
        visit(name)
    return order


def _solve_linear(equations, nvars, ctx):
    """高斯消元求解 Fraction 线性方程组。

    equations: [({var_index: Fraction}, rhs Fraction)]
    返回唯一解 list[Fraction]；无解或多解时抛 ModelError。
    """
    rows = []
    for coeffs, rhs in equations:
        row = [Fraction(0)] * (nvars + 1)
        for j, c in coeffs.items():
            row[j] = c
        row[nvars] = rhs
        rows.append(row)

    pivot_of_col = {}
    r = 0
    for col in range(nvars):
        piv = None
        for i in range(r, len(rows)):
            if rows[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = 1 / rows[r][col]
        rows[r] = [x * inv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col] != 0:
                factor = rows[i][col]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r])]
        pivot_of_col[col] = r
        r += 1

    for row in rows:
        if all(row[j] == 0 for j in range(nvars)) and row[nvars] != 0:
            raise ModelError(f"{ctx}: 量纲约束矛盾")
    if len(pivot_of_col) < nvars:
        raise ModelError(f"{ctx}: 量纲约束不足，存在多解")

    solution = [Fraction(0)] * nvars
    for col, ri in pivot_of_col.items():
        solution[col] = rows[ri][nvars]
    return solution


def infer_dims(model: dict) -> dict:
    """联合表达式约束、量纲标注与尺寸槽要求，推断每个节点的整数量纲。

    只有整个模型存在唯一整数量纲解时才返回 {节点名: (长度指数, 角度指数)}。
    """
    validate_model(model)
    nodes = model["nodes"]
    names = list(nodes)
    index = {name: i for i, name in enumerate(names)}

    # 长度分量与角度分量是两个独立的线性方程组
    systems = ([], [])  # equations per component

    def eq(coeffs, rhs, comp=None):
        targets = comp if comp is not None else (0, 1)
        for c in targets:
            systems[c].append(({k: Fraction(v) for k, v in coeffs.items()}, Fraction(rhs)))

    for name, node in nodes.items():
        i = index[name]
        t = node["type"]
        if t in ("input", "const") and "dim" in node:
            dim = _check_dim(node["dim"], f"节点 {name}")
            eq({i: 1}, dim[0], comp=(0,))
            eq({i: 1}, dim[1], comp=(1,))
        elif t in ("add", "sub"):
            a, b = (index[x] for x in node["args"])
            eq({i: 1, a: -1}, 0)
            eq({i: 1, b: -1}, 0)
        elif t == "mul":
            a, b = (index[x] for x in node["args"])
            eq({i: 1, a: -1, b: -1}, 0)
        elif t == "div":
            a, b = (index[x] for x in node["args"])
            eq({i: 1, a: -1, b: 1}, 0)
        elif t == "pow":
            a = index[node["arg"]]
            eq({i: 1, a: -node["exp"]}, 0)
        elif t in ("sin", "cos"):
            a = index[node["arg"]]
            eq({i: 1}, 0)
            eq({a: 1}, 0, comp=(0,))
            eq({a: 1}, 1, comp=(1,))

    for slot, target in model["features"].items():
        dim = SLOT_DIMS[slot]
        i = index[target]
        eq({i: 1}, dim[0], comp=(0,))
        eq({i: 1}, dim[1], comp=(1,))

    dims = {}
    for comp in (0, 1):
        sol = _solve_linear(systems[comp], len(names), "量纲推断")
        for name, v in zip(names, sol):
            if v.denominator != 1:
                raise ModelError(f"量纲推断: 节点 {name} 的量纲为分数 {v}，非整数解")
            dims.setdefault(name, [0, 0])[comp] = int(v)
    return {name: tuple(d) for name, d in dims.items()}


def normalize(model: dict) -> dict:
    """推断每个节点的量纲；唯一整数解时返回目标单位（m/rad）JSON，否则抛 ModelError。"""
    dims = infer_dims(model)
    out_nodes = {}
    for name, node in model["nodes"].items():
        new_node = {k: v for k, v in node.items() if k != "dim"}
        a, b = dims[name]
        factor = (0.001 ** a) * ((math.pi / 180.0) ** b)
        if node["type"] in ("input", "const"):
            new_node["value"] = node["value"] * factor
        new_node["dim"] = [a, b]
        out_nodes[name] = new_node
    return {
        "units": dict(TARGET_UNITS),
        "nodes": out_nodes,
        "features": dict(model["features"]),
    }


def evaluate(model: dict, overrides: dict | None = None) -> dict:
    """求出全部特征尺寸；overrides 替换独立输入，数值采用模型自身基础单位。"""
    validate_model(model)
    overrides = overrides or {}
    nodes = model["nodes"]
    for key in overrides:
        if key not in nodes or nodes[key]["type"] != "input":
            raise ModelError(f"overrides 引用了不存在的独立输入 {key!r}")

    angle_unit = model["units"]["angle"]
    if angle_unit not in ("deg", "rad"):
        raise ModelError(f"未知角度单位 {angle_unit!r}")
    to_rad = math.radians if angle_unit == "deg" else (lambda x: x)

    values = {}
    for name in topo_order(model):
        node = nodes[name]
        t = node["type"]
        if t == "input":
            v = overrides.get(name, node["value"])
        elif t == "const":
            v = node["value"]
        elif t == "add":
            v = values[node["args"][0]] + values[node["args"][1]]
        elif t == "sub":
            v = values[node["args"][0]] - values[node["args"][1]]
        elif t == "mul":
            v = values[node["args"][0]] * values[node["args"][1]]
        elif t == "div":
            denom = values[node["args"][1]]
            if denom == 0:
                raise ModelError(f"节点 {name}: 除数为零")
            v = values[node["args"][0]] / denom
        elif t == "pow":
            v = values[node["arg"]] ** node["exp"]
        elif t == "sin":
            v = math.sin(to_rad(values[node["arg"]]))
        elif t == "cos":
            v = math.cos(to_rad(values[node["arg"]]))
        else:  # pragma: no cover - validate_model 已拦截
            raise ModelError(f"节点 {name}: 未知类型 {t!r}")
        if not math.isfinite(v):
            raise ModelError(f"节点 {name}: 计算结果非有限值")
        values[name] = v

    return {slot: values[target] for slot, target in model["features"].items()}
