"""Regression tests for the released checkpoint's encoder output contract."""

import ast
import math
from pathlib import Path
import unittest

import torch


def load_forward_encode_image():
    source_path = Path("LHM/models/modeling_hand_lrm.py")
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelHandLRM":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward_encode_image":
                    isolated = ast.Module(body=[item], type_ignores=[])
                    ast.fix_missing_locations(isolated)
                    namespace = {"torch": torch, "math": math}
                    exec(compile(isolated, str(source_path), "exec"), namespace)
                    return namespace["forward_encode_image"]
    raise AssertionError("forward_encode_image was not found")


class EncoderOutputContractTest(unittest.TestCase):
    def setUp(self):
        self.forward_encode_image = load_forward_encode_image()

    def call(self, output):
        model = type("Model", (), {"encoder": lambda _self, _image: output})()
        return self.forward_encode_image(model, torch.empty(1))

    def test_current_three_item_encoder_output_is_preserved(self):
        local = torch.randn(1, 4, 3)
        feature = torch.randn(1, 3, 2, 2)
        cls = torch.randn(1, 3)
        self.assertEqual(self.call((local, feature, cls)), (local, feature, cls))

    def test_legacy_single_token_tensor_with_global_token_is_normalized(self):
        # 2x2 local tokens followed by the legacy global token.
        tokens = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
        local, feature, cls = self.call(tokens)
        self.assertTrue(torch.equal(local, tokens[:, :4]))
        self.assertTrue(torch.equal(cls, tokens[:, -1]))
        self.assertEqual(tuple(feature.shape), (1, 3, 2, 2))

    def test_legacy_single_feature_map_is_normalized(self):
        feature = torch.randn(1, 3, 2, 2)
        local, returned_feature, cls = self.call(feature)
        self.assertEqual(tuple(local.shape), (1, 4, 3))
        self.assertIs(returned_feature, feature)
        self.assertTrue(torch.equal(cls, local.mean(dim=1)))

    def test_feature_map_dict_is_normalized(self):
        feature = torch.randn(1, 3, 2, 2)
        local, returned_feature, cls = self.call({"feature_map": feature})
        self.assertEqual(tuple(local.shape), (1, 4, 3))
        self.assertIs(returned_feature, feature)
        self.assertTrue(torch.equal(cls, local.mean(dim=1)))


if __name__ == "__main__":
    unittest.main()
