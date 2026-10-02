# CAD 参数尺寸图单位迁移组件

纯 Python 标准库实现, 无第三方依赖。输入为 JSON 模型(具名节点构成的有向无环图 +
特征尺寸槽引用), 迁移保留参数依赖关系, 而非只换算当前尺寸值。

## 模型 JSON 结构

```json
{
  "units": "mm_deg",
  "nodes": {
    "p":     {"op": "input", "value": 2.0},
    "q":     {"op": "input", "value": 3.0},
    "theta": {"op": "input", "value": 30.0, "dims": [0, 1]},
    "pq":    {"op": "mul", "args": ["p", "q"]},
    "q2":    {"op": "pow", "arg": "q", "exp": 2},
    "s":     {"op": "sin", "arg": "theta"},
    "depth": {"op": "mul", "args": ["pq", "s"]}
  },
  "features": {
    "extrude.depth": "depth",
    "fillet.radius": "pq",
    "revolve.angle": "theta"
  }
}
```

- `units`: `"mm_deg"`(源, 毫米/度) 或 `"m_rad"`(目标, 米/弧度)。
- 节点 `op`: `input`(独立输入, `dims` 可缺省)、`const`(常量, `dims` 必须)、
  `add`/`sub`/`mul`/`div`(`args` 两个节点名)、`pow`(`arg` + 整数 `exp`)、
  `sin`/`cos`(`arg`)。引用允许前向出现, 多个尺寸可共享同一子图。
- `dims` 为整数二元组 `[长度指数, 角度指数]`, 允许负数。
- `features` 固定三个尺寸槽: `extrude.depth`、`fillet.radius`(长度量纲)、
  `revolve.angle`(角度量纲); 所有节点都必须能沿依赖方向到达某个尺寸槽。

## API

```python
import cad_units

# 量纲推断: 联合表达式约束 + 已有标注 + 尺寸槽要求,
# 仅当整个模型存在唯一整数量纲解时成功, 否则抛 ModelError
dims = cad_units.infer_dims(model)          # {"p": (1, 0), "q": (0, 0), ...}

# 单位迁移: 源模型(mm/deg) -> 目标模型(m/rad), 保留参数依赖
target = cad_units.normalize(model)

# 求全部特征尺寸; overrides 替换独立输入, 数值采用模型自身基础单位
cad_units.evaluate(model, {"p": 5.0})       # 源模型: 毫米/度
cad_units.evaluate(target, {"p": 0.005})    # 目标模型: 米/弧度
```

换算规则: 长度指数 `a`、角度指数 `b` 时, 源->目标换算因子为
`0.001**a * (pi/180)**b`。三角函数源模型按度、目标模型按弧度计算;
角度量纲数值迁移时已乘以 `pi/180`, 因此 `sin`/`cos` 节点结构保持不变。

模型为纯 JSON, `json.dumps`/`json.loads` 序列化往返后仍可正常推断、
迁移与修改独立输入求值。

## 运行测试

```sh
python3 tests/test_cad_units.py
```
