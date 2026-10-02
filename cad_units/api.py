"""对外操作: normalize(单位迁移) 与 evaluate(特征尺寸求值)。"""
from __future__ import annotations

import copy
import math
from typing import Dict, Mapping

from .infer import infer_dims
from .model import (
    FEATURE_SLOTS,
    SOURCE_UNITS,
    TARGET_UNITS,
    Model,
    ModelError,
    topo_order,
    validate_model,
)


def conversion_factor(a: int, b: int) -> float:
    """长度指数 a、角度指数 b 对应的 源->目标 换算因子。"""
    return (0.001 ** a) * ((math.pi / 180.0) ** b)


def normalize(model: Model) -> Model:
    """推断量纲并把源模型(mm/deg)迁移为目标模型(m/rad)的 JSON。

    只有整个模型存在唯一整数量纲解时才返回目标 JSON, 否则抛 ModelError。
    目标 JSON 仍使用相同节点类型: 独立输入与常量按推断量纲换算数值,
    表达式结构(参数依赖)原样保留; 三角函数在目标模型按弧度计算,
    角度量纲的数值已乘以 pi/180, 故节点本身无需改动。
    """
    if not isinstance(model, dict) or model.get("units") != SOURCE_UNITS:
        raise ModelError(f"normalize 只接受源模型(units={SOURCE_UNITS!r})")
    dims = infer_dims(model)

    out = copy.deepcopy(model)
    out["units"] = TARGET_UNITS
    for name, node in out["nodes"].items():
        if node["op"] == "input":
            a, b = dims[name]
            node["value"] = node["value"] * conversion_factor(a, b)
            node["dims"] = [a, b]
        elif node["op"] == "const":
            a, b = node["dims"]
            node["value"] = node["value"] * conversion_factor(a, b)
    return out


def evaluate(model: Model, overrides: Mapping[str, float] | None = None) -> Dict[str, float]:
    """求出全部特征尺寸; overrides 替换独立输入(采用模型自身基础单位)。

    源模型(units="mm_deg")按毫米/度计算, 目标模型(units="m_rad")按米/弧度计算。
    """
    validate_model(model)
    overrides = overrides or {}
    for key in overrides:
        node = model["nodes"].get(key)
        if node is None or node["op"] != "input":
            raise ModelError(f"overrides 只能替换独立输入节点, 收到 {key!r}")

    in_degrees = model["units"] == SOURCE_UNITS
    values: Dict[str, float] = {}
    for name in topo_order(model):
        node = model["nodes"][name]
        op = node["op"]
        if op == "input":
            values[name] = float(overrides.get(name, node["value"]))
        elif op == "const":
            values[name] = float(node["value"])
        elif op == "add":
            values[name] = values[node["args"][0]] + values[node["args"][1]]
        elif op == "sub":
            values[name] = values[node["args"][0]] - values[node["args"][1]]
        elif op == "mul":
            values[name] = values[node["args"][0]] * values[node["args"][1]]
        elif op == "div":
            values[name] = values[node["args"][0]] / values[node["args"][1]]
        elif op == "pow":
            values[name] = values[node["arg"]] ** node["exp"]
        elif op in ("sin", "cos"):
            x = values[node["arg"]]
            if in_degrees:
                x = math.radians(x)
            values[name] = math.sin(x) if op == "sin" else math.cos(x)
        else:  # pragma: no cover - validate_model 已拦截
            raise ModelError(f"未知 op: {op!r}")

    return {slot: values[ref] for slot, ref in model["features"].items()}
