# CAD 参数尺寸图单位迁移组件

仅依赖 Python 标准库（`json` / `math` / `fractions`），在源模型（毫米/度）与目标模型（米/弧度）之间迁移参数化尺寸图，保留完整参数依赖而非仅换算当前值。

## 模型 JSON 结构

```json
{
  "units": {"length": "mm", "angle": "deg"},
  "nodes": {
    "p":     {"type": "input", "value": 12.0},
    "theta": {"type": "input", "value": 30.0},
    "half":  {"type": "const", "value": 0.5, "dim": [0, 0]},
    "q2":    {"type": "pow", "arg": "q", "exp": 2},
    "depth": {"type": "mul", "args": ["p", "q"]},
    "s":     {"type": "sin", "arg": "theta"}
  },
  "features": {
    "extrude.depth": "depth",
    "fillet.radius": "radius",
    "revolve.angle": "angle_out"
  }
}
```

- 节点类型：`input`（独立输入，`dim` 可缺省）、`const`（必须标注 `dim`）、`add`/`sub`/`mul`/`div`、`pow`（整数次幂）、`sin`/`cos`。引用允许前向出现，多个尺寸可共享子图。
- 量纲 `dim` 为整数二元组 `[长度指数, 角度指数]`，允许负数。
- 尺寸槽：`extrude.depth`、`fillet.radius` 要求长度量纲；`revolve.angle` 要求角度量纲。所有节点必须可达某个尺寸槽。

## API

```python
from cad_units import normalize, evaluate, infer_dims, ModelError

# 推断每个节点的量纲；仅当存在唯一整数解时返回目标单位（m/rad）JSON，
# 否则抛 ModelError（多解 / 矛盾 / 分数量纲）。
target = normalize(source_model)

# 求全部特征尺寸；overrides 替换独立输入，数值采用模型自身基础单位。
sizes = evaluate(model, {"p": 20.0, "theta": 45.0})
```

迁移规则：量纲 `[a, b]` 的输入/常量默认值乘以 `0.001**a * (pi/180)**b`，图结构不变；源模型按度、目标模型按弧度计算三角函数，角度输入换算为弧度后二者结果一致。`normalize` 输出的 JSON 可直接 `json.dumps`/`json.loads` 往返，之后仍支持 `evaluate(..., overrides)` 修改独立输入。

## 运行测试

```sh
python3 -m unittest test_cad_units -v
```
