"""CAD 参数尺寸模型的 JSON 结构定义与校验。

模型 JSON 结构:

    {
      "units": "mm_deg",            # "mm_deg"(源) 或 "m_rad"(目标)
      "nodes": {"<name>": <node>, ...},
      "features": {
        "extrude.depth": "<node名>",
        "fillet.radius": "<node名>",
        "revolve.angle": "<node名>"
      }
    }

节点类型(node.op):
  - "input": 独立输入, {"op": "input", "value": 数, "dims": [a, b](可选)}
  - "const": 常量, {"op": "const", "value": 数, "dims": [a, b](必须)}
  - "add"/"sub"/"mul"/"div": {"op": ..., "args": [左, 右]}
  - "pow": 整数常量次幂, {"op": "pow", "arg": 节点名, "exp": 整数}
  - "sin"/"cos": {"op": ..., "arg": 节点名}

量纲 dims 为整数二元组 [长度指数, 角度指数]。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

Dims = Tuple[int, int]
Node = Dict[str, Any]
Model = Dict[str, Any]

SOURCE_UNITS = "mm_deg"
TARGET_UNITS = "m_rad"

FEATURE_SLOTS: Dict[str, Dims] = {
    "extrude.depth": (1, 0),
    "fillet.radius": (1, 0),
    "revolve.angle": (0, 1),
}

BINARY_OPS = ("add", "sub", "mul", "div")
TRIG_OPS = ("sin", "cos")


class ModelError(ValueError):
    """模型结构或量纲约束不合法。"""


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_dims(dims: Any, where: str) -> None:
    if (
        not isinstance(dims, (list, tuple))
        or len(dims) != 2
        or any(isinstance(d, bool) or not isinstance(d, int) for d in dims)
    ):
        raise ModelError(f"{where}: dims 必须是整数二元组 [长度指数, 角度指数]")


def node_deps(node: Node) -> List[str]:
    """返回一个节点直接依赖的节点名列表。"""
    op = node.get("op")
    if op in BINARY_OPS:
        return list(node["args"])
    if op in ("pow",) + TRIG_OPS:
        return [node["arg"]]
    return []


def validate_model(model: Model) -> None:
    """校验模型 JSON 结构(节点类型、引用、DAG、特征槽齐全)。"""
    if not isinstance(model, dict):
        raise ModelError("模型必须是 JSON 对象")
    units = model.get("units")
    if units not in (SOURCE_UNITS, TARGET_UNITS):
        raise ModelError(f"units 必须是 {SOURCE_UNITS!r} 或 {TARGET_UNITS!r}")
    nodes = model.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ModelError("nodes 必须是非空对象")
    features = model.get("features")
    if not isinstance(features, dict) or set(features) != set(FEATURE_SLOTS):
        raise ModelError(f"features 必须恰好包含 {sorted(FEATURE_SLOTS)} 三个尺寸槽")

    for name, node in nodes.items():
        if not isinstance(name, str) or not name:
            raise ModelError("节点名必须是非空字符串")
        if not isinstance(node, dict):
            raise ModelError(f"节点 {name!r} 必须是对象")
        op = node.get("op")
        if op == "input":
            if not _is_number(node.get("value")):
                raise ModelError(f"input 节点 {name!r} 缺少数值 value")
            if "dims" in node:
                _check_dims(node["dims"], f"input 节点 {name!r}")
        elif op == "const":
            if not _is_number(node.get("value")):
                raise ModelError(f"const 节点 {name!r} 缺少数值 value")
            if "dims" not in node:
                raise ModelError(f"const 节点 {name!r} 必须标注 dims")
            _check_dims(node["dims"], f"const 节点 {name!r}")
        elif op in BINARY_OPS:
            args = node.get("args")
            if (
                not isinstance(args, list)
                or len(args) != 2
                or not all(isinstance(a, str) for a in args)
            ):
                raise ModelError(f"{op} 节点 {name!r} 需要两个节点名 args")
        elif op == "pow":
            if not isinstance(node.get("arg"), str):
                raise ModelError(f"pow 节点 {name!r} 需要节点名 arg")
            exp = node.get("exp")
            if isinstance(exp, bool) or not isinstance(exp, int):
                raise ModelError(f"pow 节点 {name!r} 的 exp 必须是整数")
        elif op in TRIG_OPS:
            if not isinstance(node.get("arg"), str):
                raise ModelError(f"{op} 节点 {name!r} 需要节点名 arg")
        else:
            raise ModelError(f"节点 {name!r} 的 op 未知: {op!r}")

    for name, node in nodes.items():
        for dep in node_deps(node):
            if dep not in nodes:
                raise ModelError(f"节点 {name!r} 引用了不存在的节点 {dep!r}")
    for slot, ref in features.items():
        if not isinstance(ref, str) or ref not in nodes:
            raise ModelError(f"尺寸槽 {slot!r} 引用了不存在的节点 {ref!r}")

    order = topo_order(model)

    reachable = set()
    for ref in features.values():
        i = order.index(ref)
        reachable.update(order[: i + 1])
    unreachable = set(nodes) - reachable
    if unreachable:
        raise ModelError(f"以下节点无法沿依赖方向到达任何尺寸槽: {sorted(unreachable)}")


def topo_order(model: Model) -> List[str]:
    """返回按依赖先后的节点名拓扑序;有环则抛 ModelError。"""
    nodes = model["nodes"]
    order: List[str] = []
    state: Dict[str, int] = {}  # 1=访问中 2=完成

    def visit(name: str, stack: Tuple[str, ...]) -> None:
        s = state.get(name, 0)
        if s == 2:
            return
        if s == 1:
            raise ModelError(f"依赖存在环: {' -> '.join(stack + (name,))}")
        state[name] = 1
        for dep in node_deps(nodes[name]):
            visit(dep, stack + (name,))
        state[name] = 2
        order.append(name)

    for name in nodes:
        visit(name, ())
    return order
