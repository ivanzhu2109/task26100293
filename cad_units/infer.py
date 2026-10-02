"""量纲推断: 联合表达式约束、量纲标注与尺寸槽要求求解整数量纲。"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple

from .model import (
    BINARY_OPS,
    FEATURE_SLOTS,
    TRIG_OPS,
    Dims,
    Model,
    ModelError,
    validate_model,
)

Var = Tuple[str, int]  # (节点名, 0=长度指数 / 1=角度指数)
Equation = Tuple[Dict[Var, Fraction], Fraction]


def _eq(terms: Dict[Var, int], rhs: int = 0) -> Equation:
    return ({v: Fraction(c) for v, c in terms.items() if c}, Fraction(rhs))


def _build_equations(model: Model) -> Tuple[List[Var], List[Equation]]:
    nodes = model["nodes"]
    variables: List[Var] = [(name, axis) for name in nodes for axis in (0, 1)]
    equations: List[Equation] = []

    for name, node in nodes.items():
        op = node["op"]
        na = (name, 0)
        nb = (name, 1)
        if op in ("input", "const") and "dims" in node:
            da, db = node["dims"]
            equations.append(_eq({na: 1}, da))
            equations.append(_eq({nb: 1}, db))
        elif op in ("add", "sub"):
            lhs, rhs = node["args"]
            for axis in (0, 1):
                n, l, r = (name, axis), (lhs, axis), (rhs, axis)
                equations.append(_eq({n: 1, l: -1}))
                equations.append(_eq({n: 1, r: -1}))
        elif op == "mul":
            lhs, rhs = node["args"]
            for axis in (0, 1):
                equations.append(_eq({(lhs, axis): 1, (rhs, axis): 1, (name, axis): -1}))
        elif op == "div":
            lhs, rhs = node["args"]
            for axis in (0, 1):
                equations.append(_eq({(lhs, axis): 1, (rhs, axis): -1, (name, axis): -1}))
        elif op == "pow":
            k = node["exp"]
            arg = node["arg"]
            for axis in (0, 1):
                equations.append(_eq({(arg, axis): k, (name, axis): -1}))
        elif op in TRIG_OPS:
            arg = node["arg"]
            equations.append(_eq({(arg, 0): 1}, 0))
            equations.append(_eq({(arg, 1): 1}, 1))
            equations.append(_eq({na: 1}, 0))
            equations.append(_eq({nb: 1}, 0))

    for slot, (da, db) in FEATURE_SLOTS.items():
        ref = model["features"][slot]
        equations.append(_eq({(ref, 0): 1}, da))
        equations.append(_eq({(ref, 1): 1}, db))

    return variables, equations


def _solve_unique(variables: List[Var], equations: List[Equation]) -> Dict[Var, Fraction]:
    """稀疏高斯消元; 要求每个变量都被唯一确定, 否则抛 ModelError。"""
    pivots: Dict[Var, Tuple[Dict[Var, Fraction], Fraction]] = {}
    for eq, rhs in equations:
        eq = {v: c for v, c in eq.items() if c}
        while any(v in pivots for v in eq):
            for v in list(eq):
                if v in pivots:
                    peq, prhs = pivots[v]
                    factor = eq.pop(v)
                    for vv, cc in peq.items():
                        eq[vv] = eq.get(vv, Fraction(0)) - factor * cc
                    rhs -= factor * prhs
            eq = {v: c for v, c in eq.items() if c}
        if not eq:
            if rhs != 0:
                raise ModelError("量纲约束矛盾: 同一节点被要求具有不一致的量纲")
            continue
        pv = min(eq)
        coeff = eq.pop(pv)
        pivots[pv] = ({v: c / coeff for v, c in eq.items()}, rhs / coeff)

    free = [v for v in variables if v not in pivots]
    if free:
        names = sorted({v[0] for v in free})
        raise ModelError(f"量纲约束不足, 存在多解; 未确定的节点: {names}")

    values: Dict[Var, Fraction] = {}
    for pv in reversed(list(pivots)):
        peq, prhs = pivots[pv]
        values[pv] = prhs - sum(c * values[v] for v, c in peq.items())
    return values


def infer_dims(model: Model) -> Dict[str, Dims]:
    """推断每个节点的量纲。

    只有整个模型存在唯一整数量纲解时返回 {节点名: (a, b)},
    否则(多解 / 矛盾 / 仅分数解)抛 ModelError。
    """
    validate_model(model)
    variables, equations = _build_equations(model)
    values = _solve_unique(variables, equations)
    result: Dict[str, Dims] = {}
    for name in model["nodes"]:
        da, db = values[(name, 0)], values[(name, 1)]
        if da.denominator != 1 or db.denominator != 1:
            raise ModelError(
                f"节点 {name!r} 的量纲为分数 ({da}, {db}), 不存在整数量纲解"
            )
        result[name] = (int(da), int(db))
    return result
