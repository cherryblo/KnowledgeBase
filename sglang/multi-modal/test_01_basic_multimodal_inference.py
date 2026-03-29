"""
用例1：基础多模态推理测试（图像+文本）

测试目的：验证Multi-Modal特性在NPU环境下能正常处理图像+文本输入并生成预期输出
测试环境：使用1张NPU卡，单机部署
"""

import base64
import io
import json
import os
import time
import unittest
from pathlib import Path

import requests
from PIL import Image


class TestBasicMultimodalInference(unittest.TestCase):
    """基础多模态推理测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv(
            "MODEL_PATH", "microsoft/Phi-4-multimodal-instruct"
        )
        cls.timeout = 300

        # 创建测试图像
        cls.test_image_path = cls._create_test_image()

    @staticmethod
    def _create_test_image():
        """创建一个简单的测试图像"""
        image = Image.new("RGB", (640, 480), color="red")
        image_path = Path("/tmp/test_image.jpg")
        image.save(image_path)
        return str(image_path)

    def _encode_image(self, image_path):
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def test_01_server_health_check(self):
        """测试1：服务器健康检查"""
        print("\n=== 测试1：服务器健康检查 ===")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            self.assertEqual(response.status_code, 200)
            print("✓ 服务器健康检查通过")
        except Exception as e:
            self.fail(f"服务器健康检查失败: {e}")

    def test_02_model_info(self):
        """测试2：获取模型信息"""
        print("\n=== 测试2：获取模型信息 ===")

        try:
            response = requests.get(f"{self.base_url}/get_model_info", timeout=10)
            self.assertEqual(response.status_code, 200)
            model_info = response.json()
            print(f"✓ 模型信息: {model_info.get('model_path', 'N/A')}")
        except Exception as e:
            self.fail(f"获取模型信息失败: {e}")

    def test_03_single_image_inference(self):
        """测试3：单图像推理"""
        print("\n=== 测试3：单图像推理 ===")

        # 准备请求
        image_base64 = self._encode_image(self.test_image_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "What color is this image? Please answer with just the color name."},
                    ],
                }
            ],
            "max_tokens": 50,
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            elapsed_time = time.time() - start_time

            self.assertEqual(response.status_code, 200)
            result = response.json()

            # 验证响应
            self.assertIn("choices", result)
            self.assertTrue(len(result["choices"]) > 0)
            self.assertIn("message", result["choices"][0])
            self.assertIn("content", result["choices"][0]["message"])

            content = result["choices"][0]["message"]["content"]
            print(f"✓ 推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

            # 验证回答中包含颜色信息
            self.assertTrue(
                any(color in content.lower() for color in ["red", "color"]),
                f"回答中未检测到颜色信息: {content}",
            )

        except Exception as e:
            self.fail(f"单图像推理失败: {e}")

    def test_04_multiple_images_inference(self):
        """测试4：多图像推理"""
        print("\n=== 测试4：多图像推理 ===")

        # 创建不同颜色的图像
        images = []
        colors = ["red", "blue", "green"]

        for color in colors:
            image = Image.new("RGB", (640, 480), color=color)
            image_path = Path(f"/tmp/test_image_{color}.jpg")
            image.save(image_path)
            images.append((str(image_path), color))

        # 准备请求
        content = [{"type": "text", "text": "Describe the colors of these images in order."}]
        for image_path, _ in images:
            image_base64 = self._encode_image(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        payload = {
            "model": self.model_path,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 100,
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            elapsed_time = time.time() - start_time

            self.assertEqual(response.status_code, 200)
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            print(f"✓ 多图像推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

        except Exception as e:
            self.fail(f"多图像推理失败: {e}")

    def test_05_streaming_inference(self):
        """测试5：流式推理"""
        print("\n=== 测试5：流式推理 ===")

        image_base64 = self._encode_image(self.test_image_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "What do you see in this image?"},
                    ],
                }
            ],
            "max_tokens": 50,
            "stream": True,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )

            self.assertEqual(response.status_code, 200)

            full_content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                full_content += content
                        except json.JSONDecodeError:
                            pass

            print(f"✓ 流式推理成功")
            print(f"✓ 完整回答: {full_content}")
            self.assertTrue(len(full_content) > 0, "流式推理未返回任何内容")

        except Exception as e:
            self.fail(f"流式推理失败: {e}")

    @classmethod
    def tearDownClass(cls):
        """清理测试文件"""
        try:
            if os.path.exists(cls.test_image_path):
                os.remove(cls.test_image_path)
            for color in ["red", "blue", "green"]:
                image_path = f"/tmp/test_image_{color}.jpg"
                if os.path.exists(image_path):
                    os.remove(image_path)
        except Exception as e:
            print(f"清理测试文件时出错: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
