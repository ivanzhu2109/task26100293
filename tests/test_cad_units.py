"""cad_units 自动化测试: python3 tests/test_cad_units.py"""
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cad_units
from cad_units import ModelError

DEG = math.pi / 180.0


def joint_model():
    """p、q 未标注量纲, 由 p*q 与 p*q^2 接入两个长度槽联合确定;
    另含角度输入与 sin/cos 三角函数依赖, 且多个尺寸共享子图。"""
    return {
        "units": "mm_deg",
        "nodes": {
            "p": {"op": "input", "value": 2.0},
            "q": {"op": "input", "value": 3.0},
            "pq": {"op": "mul", "args": ["p", "q"]},
            "q2": {"op": "pow", "arg": "q", "exp": 2},
            "pqq": {"op": "mul", "args": ["p", "q2"]},
            "theta": {"op": "input", "value": 30.0, "dims": [0, 1]},
            "half": {"op": "const", "value": 0.5, "dims": [0, 0]},
            "double_theta": {"op": "div", "args": ["theta", "half"]},
            "s": {"op": "sin", "arg": "theta"},
            "c": {"op": "cos", "arg": "double_theta"},
            "sc": {"op": "add", "args": ["s", "c"]},
            "depth": {"op": "mul", "args": ["pq", "sc"]},
            "pi_const": {"op": "const", "value": 180.0, "dims": [0, 1]},
            "angle": {"op": "sub", "args": ["pi_const", "theta"]},
        },
        "features": {
            "extrude.depth": "depth",
            "fillet.radius": "pqq",
            "revolve.angle": "angle",
        },
    }


class TestJointInference(unittest.TestCase):
    def test_dims_inferred(self):
        dims = cad_units.infer_dims(joint_model())
        self.assertEqual(dims["p"], (1, 0))   # 长度
        self.assertEqual(dims["q"], (0, 0))   # 无量纲
        self.assertEqual(dims["pq"], (1, 0))
        self.assertEqual(dims["pqq"], (1, 0))
        self.assertEqual(dims["theta"], (0, 1))
        self.assertEqual(dims["s"], (0, 0))
        self.assertEqual(dims["depth"], (1, 0))
        self.assertEqual(dims["angle"], (0, 1))

    def test_normalize_returns_target_json(self):
        target = cad_units.normalize(joint_model())
        self.assertEqual(target["units"], "m_rad")
        self.assertAlmostEqual(target["nodes"]["p"]["value"], 0.002)
        self.assertAlmostEqual(target["nodes"]["q"]["value"], 3.0)
        self.assertAlmostEqual(target["nodes"]["theta"]["value"], 30.0 * DEG)
        self.assertEqual(target["nodes"]["p"]["dims"], [1, 0])
        self.assertEqual(target["nodes"]["q"]["dims"], [0, 0])
        # 表达式结构(参数依赖)保留
        self.assertEqual(target["nodes"]["depth"]["op"], "mul")
        self.assertEqual(target["nodes"]["depth"]["args"], ["pq", "sc"])
        # 归一化结果自身仍可通过量纲校验
        cad_units.infer_dims(target)

    def test_source_target_consistency_with_overrides(self):
        model = joint_model()
        target = cad_units.normalize(model)
        for ov_src in (
            {},
            {"p": 5.0, "q": 1.5, "theta": 45.0},
            {"p": 0.1, "q": 7.25, "theta": 123.0},
        ):
            src = cad_units.evaluate(model, ov_src)
            ov_tgt = {}
            dims = cad_units.infer_dims(model)
            for name, value in ov_src.items():
                a, b = dims[name]
                ov_tgt[name] = value * cad_units.conversion_factor(a, b)
            tgt = cad_units.evaluate(target, ov_tgt)
            self.assertAlmostEqual(tgt["extrude.depth"], src["extrude.depth"] * 0.001)
            self.assertAlmostEqual(tgt["fillet.radius"], src["fillet.radius"] * 0.001)
            self.assertAlmostEqual(tgt["revolve.angle"], src["revolve.angle"] * DEG)

    def test_serialization_roundtrip(self):
        target = cad_units.normalize(joint_model())
        loaded = json.loads(json.dumps(target))
        result = cad_units.evaluate(loaded, {"p": 0.004, "q": 2.0, "theta": 10.0 * DEG})
        src = cad_units.evaluate(joint_model(), {"p": 4.0, "q": 2.0, "theta": 10.0})
        self.assertAlmostEqual(result["extrude.depth"], src["extrude.depth"] * 0.001)
        self.assertAlmostEqual(result["fillet.radius"], src["fillet.radius"] * 0.001)
        self.assertAlmostEqual(result["revolve.angle"], src["revolve.angle"] * DEG)

    def test_forward_reference_and_shared_subgraph(self):
        model = joint_model()
        # 前向引用: 在 nodes 字典序中, 引用出现在被引用节点之前
        nodes = model["nodes"]
        reordered = {"depth": nodes["depth"], "sc": nodes["sc"]}
        for name, node in nodes.items():
            reordered.setdefault(name, node)
        model["nodes"] = reordered
        # pq 子图同时被 depth 与 pqq 共享 -> 推断与求值不受影响
        self.assertEqual(cad_units.infer_dims(model)["pq"], (1, 0))
        target = cad_units.normalize(model)
        self.assertAlmostEqual(
            cad_units.evaluate(target)["fillet.radius"], 18.0 * 0.001
        )


class TestRejections(unittest.TestCase):
    def test_underdetermined_model_rejected(self):
        # 只有 p*q 接入长度槽: p、q 量纲有无穷多解
        model = {
            "units": "mm_deg",
            "nodes": {
                "p": {"op": "input", "value": 2.0},
                "q": {"op": "input", "value": 3.0},
                "pq": {"op": "mul", "args": ["p", "q"]},
                "r": {"op": "input", "value": 1.0, "dims": [1, 0]},
                "a": {"op": "input", "value": 90.0, "dims": [0, 1]},
            },
            "features": {
                "extrude.depth": "pq",
                "fillet.radius": "r",
                "revolve.angle": "a",
            },
        }
        with self.assertRaises(ModelError):
            cad_units.normalize(model)

    def test_contradictory_model_rejected(self):
        # 同一节点同时接入长度槽与角度槽
        model = {
            "units": "mm_deg",
            "nodes": {
                "x": {"op": "input", "value": 1.0},
                "r": {"op": "input", "value": 1.0, "dims": [1, 0]},
            },
            "features": {
                "extrude.depth": "x",
                "fillet.radius": "r",
                "revolve.angle": "x",
            },
        }
        with self.assertRaises(ModelError):
            cad_units.normalize(model)

    def test_fractional_dims_rejected(self):
        # p^2 接入长度槽 -> p 的量纲只能是 (1/2, 0), 无整数解
        model = {
            "units": "mm_deg",
            "nodes": {
                "p": {"op": "input", "value": 4.0},
                "p2": {"op": "pow", "arg": "p", "exp": 2},
                "r": {"op": "input", "value": 1.0, "dims": [1, 0]},
                "a": {"op": "input", "value": 90.0, "dims": [0, 1]},
            },
            "features": {
                "extrude.depth": "p2",
                "fillet.radius": "r",
                "revolve.angle": "a",
            },
        }
        with self.assertRaises(ModelError):
            cad_units.normalize(model)


class TestModelValidation(unittest.TestCase):
    def test_const_requires_dims(self):
        model = joint_model()
        del model["nodes"]["half"]["dims"]
        with self.assertRaises(ModelError):
            cad_units.evaluate(model)

    def test_cycle_rejected(self):
        model = joint_model()
        model["nodes"]["pq"]["args"] = ["depth", "q"]
        with self.assertRaises(ModelError):
            cad_units.evaluate(model)

    def test_unknown_reference_rejected(self):
        model = joint_model()
        model["nodes"]["pq"]["args"] = ["p", "nope"]
        with self.assertRaises(ModelError):
            cad_units.evaluate(model)

    def test_orphan_node_rejected(self):
        model = joint_model()
        model["nodes"]["orphan"] = {"op": "input", "value": 1.0}
        with self.assertRaises(ModelError):
            cad_units.evaluate(model)

    def test_override_must_be_input(self):
        with self.assertRaises(ModelError):
            cad_units.evaluate(joint_model(), {"pq": 1.0})

    def test_evaluate_both_unit_systems(self):
        model = joint_model()
        src = cad_units.evaluate(model)
        tgt = cad_units.evaluate(cad_units.normalize(model))
        self.assertAlmostEqual(tgt["extrude.depth"], src["extrude.depth"] * 0.001)
        self.assertAlmostEqual(tgt["revolve.angle"], src["revolve.angle"] * DEG)


if __name__ == "__main__":
    unittest.main(verbosity=2)
