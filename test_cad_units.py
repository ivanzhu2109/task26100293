import json
import math
import unittest

from cad_units import ModelError, evaluate, infer_dims, normalize


def joint_model():
    """p、q 未标注量纲，通过 p*q 与 p*q^2 联合约束；含角度与三角函数依赖。"""
    return {
        "units": {"length": "mm", "angle": "deg"},
        "nodes": {
            "p": {"type": "input", "value": 12.0},
            "q": {"type": "input", "value": 2.0},
            "theta": {"type": "input", "value": 30.0},
            "half": {"type": "const", "value": 0.5, "dim": [0, 0]},
            "depth": {"type": "mul", "args": ["p", "q"]},
            "q2": {"type": "pow", "arg": "q", "exp": 2},
            "pq2": {"type": "mul", "args": ["p", "q2"]},
            "s": {"type": "sin", "arg": "theta"},
            "radius": {"type": "mul", "args": ["pq2", "s"]},
            "angle_out": {"type": "mul", "args": ["theta", "half"]},
        },
        "features": {
            "extrude.depth": "depth",
            "fillet.radius": "radius",
            "revolve.angle": "angle_out",
        },
    }


class NormalizeTest(unittest.TestCase):
    def test_joint_constraints_infer_dims(self):
        dims = infer_dims(joint_model())
        self.assertEqual(dims["p"], (1, 0))   # 长度
        self.assertEqual(dims["q"], (0, 0))   # 无量纲
        self.assertEqual(dims["theta"], (0, 1))
        self.assertEqual(dims["s"], (0, 0))
        self.assertEqual(dims["angle_out"], (0, 1))

    def test_normalize_rescales_inputs_and_consts(self):
        target = normalize(joint_model())
        self.assertEqual(target["units"], {"length": "m", "angle": "rad"})
        self.assertAlmostEqual(target["nodes"]["p"]["value"], 0.012)
        self.assertAlmostEqual(target["nodes"]["q"]["value"], 2.0)
        self.assertAlmostEqual(target["nodes"]["theta"]["value"], math.pi / 6)
        self.assertAlmostEqual(target["nodes"]["half"]["value"], 0.5)
        # 结构保留：目标 JSON 仍使用相同节点类型
        self.assertEqual(target["nodes"]["s"]["type"], "sin")
        self.assertEqual(target["nodes"]["q2"]["type"], "pow")
        # 所有节点标注了推断量纲
        for node in target["nodes"].values():
            self.assertIn("dim", node)

    def test_migrated_model_matches_source_under_overrides(self):
        source = joint_model()
        target = normalize(source)
        overrides = {"p": 20.0, "q": 3.0, "theta": 45.0}
        src = evaluate(source, overrides)
        # overrides 采用目标模型基础单位（m/rad）
        tgt = evaluate(target, {
            "p": 20.0 * 0.001,
            "q": 3.0,
            "theta": 45.0 * math.pi / 180.0,
        })
        self.assertAlmostEqual(tgt["extrude.depth"], src["extrude.depth"] * 0.001)
        self.assertAlmostEqual(tgt["fillet.radius"], src["fillet.radius"] * 0.001)
        self.assertAlmostEqual(tgt["revolve.angle"], src["revolve.angle"] * math.pi / 180.0)
        # 具体数值核对：sin 在源模型按度、目标模型按弧度
        self.assertAlmostEqual(src["fillet.radius"], 20.0 * 9.0 * math.sin(math.radians(45.0)))
        self.assertAlmostEqual(src["revolve.angle"], 22.5)

    def test_json_roundtrip_supports_overrides(self):
        target = normalize(joint_model())
        reloaded = json.loads(json.dumps(target))
        result = evaluate(reloaded, {"p": 0.05, "theta": math.pi / 2})
        self.assertAlmostEqual(result["extrude.depth"], 0.05 * 2.0)
        self.assertAlmostEqual(result["fillet.radius"], 0.05 * 4.0 * 1.0)
        self.assertAlmostEqual(result["revolve.angle"], math.pi / 4)


class RejectTest(unittest.TestCase):
    def test_underdetermined_model_rejected(self):
        model = {
            "units": {"length": "mm", "angle": "deg"},
            "nodes": {
                "p": {"type": "input", "value": 1.0},
                "q": {"type": "input", "value": 2.0},
                "depth": {"type": "mul", "args": ["p", "q"]},
            },
            "features": {"extrude.depth": "depth"},
        }
        with self.assertRaises(ModelError):
            normalize(model)

    def test_contradictory_slots_rejected(self):
        model = {
            "units": {"length": "mm", "angle": "deg"},
            "nodes": {"x": {"type": "input", "value": 1.0}},
            "features": {"extrude.depth": "x", "revolve.angle": "x"},
        }
        with self.assertRaises(ModelError):
            normalize(model)

    def test_fractional_dim_rejected(self):
        model = {
            "units": {"length": "mm", "angle": "deg"},
            "nodes": {
                "p": {"type": "input", "value": 4.0},
                "p2": {"type": "pow", "arg": "p", "exp": 2},
            },
            "features": {"extrude.depth": "p2"},
        }
        with self.assertRaises(ModelError):
            normalize(model)


class EvaluateTest(unittest.TestCase):
    def test_forward_reference_and_shared_subgraph(self):
        model = {
            "units": {"length": "mm", "angle": "deg"},
            "nodes": {
                # 前向引用：shared 在使用之后才定义
                "d1": {"type": "add", "args": ["shared", "c"]},
                "d2": {"type": "sub", "args": ["shared", "c"]},
                "shared": {"type": "mul", "args": ["a", "b"]},
                "a": {"type": "input", "value": 3.0, "dim": [1, 0]},
                "b": {"type": "input", "value": 4.0},
                "c": {"type": "const", "value": 1.0, "dim": [1, 0]},
            },
            "features": {"extrude.depth": "d1", "fillet.radius": "d2"},
        }
        result = evaluate(model, {"a": 5.0})
        self.assertAlmostEqual(result["extrude.depth"], 5.0 * 4.0 + 1.0)
        self.assertAlmostEqual(result["fillet.radius"], 5.0 * 4.0 - 1.0)
        dims = infer_dims(model)
        self.assertEqual(dims["b"], (0, 0))

    def test_div_and_cos(self):
        model = {
            "units": {"length": "mm", "angle": "deg"},
            "nodes": {
                "x": {"type": "input", "value": 10.0, "dim": [1, 0]},
                "theta": {"type": "input", "value": 60.0},
                "cs": {"type": "cos", "arg": "theta"},
                "depth": {"type": "div", "args": ["x", "cs"]},
            },
            "features": {"extrude.depth": "depth"},
        }
        self.assertAlmostEqual(evaluate(model)["extrude.depth"], 20.0)
        target = normalize(model)
        self.assertAlmostEqual(
            evaluate(target, {"x": 0.01, "theta": math.pi / 3})["extrude.depth"], 0.02
        )

    def test_invalid_models_rejected(self):
        base = joint_model()
        # 环
        cyclic = json.loads(json.dumps(base))
        cyclic["nodes"]["p"] = {"type": "mul", "args": ["depth", "q"]}
        with self.assertRaises(ModelError):
            evaluate(cyclic)
        # 不可达节点
        stray = json.loads(json.dumps(base))
        stray["nodes"]["stray"] = {"type": "input", "value": 1.0}
        with self.assertRaises(ModelError):
            evaluate(stray)
        # const 未标注量纲
        nodim = json.loads(json.dumps(base))
        del nodim["nodes"]["half"]["dim"]
        with self.assertRaises(ModelError):
            evaluate(nodim)
        # 未知尺寸槽
        badslot = json.loads(json.dumps(base))
        badslot["features"]["loft.width"] = "depth"
        with self.assertRaises(ModelError):
            evaluate(badslot)
        # overrides 引用非输入节点
        with self.assertRaises(ModelError):
            evaluate(base, {"depth": 1.0})

    def test_non_finite_result_rejected(self):
        model = {
            "units": {"length": "mm", "angle": "deg"},
            "nodes": {
                "x": {"type": "input", "value": 1.0, "dim": [1, 0]},
                "z": {"type": "const", "value": 0.0, "dim": [0, 0]},
                "depth": {"type": "div", "args": ["x", "z"]},
            },
            "features": {"extrude.depth": "depth"},
        }
        with self.assertRaises(ModelError):
            evaluate(model)


if __name__ == "__main__":
    unittest.main()
