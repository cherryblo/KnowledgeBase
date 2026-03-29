"""
用例4：多模态数据限制测试

测试目的：验证--limit-mm-data-per-request参数在NPU环境下生效，能够限制每个请求的多模态数据数量
测试环境：使用1张NPU卡，单机部署
"""

import base64
import json
import os
import unittest
from pathlib import Path

import requests
from PIL import Image


class TestMultimodalDataLimit(unittest.TestCase):
    """多模态数据限制测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv(
            "MODEL_PATH", "microsoft/Phi-4-multimodal-instruct"
        )
        cls.timeout = 300

        # 从环境变量读取限制配置
        limit_config = os.getenv("LIMIT_MM_DATA", '{"image": 2}')
        cls.limit_config = json.loads(limit_config)
        cls.image_limit = cls.limit_config.get("image", 2)

        print(f"图像限制: {cls.image_limit}")

    def _create_test_images(self, count):
        """创建多个测试图像"""
        images = []
        colors = ["red", "blue", "green", "yellow", "purple"]

        for i in range(count):
            color = colors[i % len(colors)]
            image = Image.new("RGB", (640, 480), color=color)
            image_path = Path(f"/tmp/test_image_limit_{i}.jpg")
            image.save(image_path)
            images.append(str(image_path))

        return images

    def _encode_image(self, image_path):
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _send_request_with_images(self, num_images):
        """发送包含指定数量图像的请求"""
        images = self._create_test_images(num_images)

        content = [{"type": "text", "text": "Describe these images."}]
        for image_path in images:
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
            "max_tokens": 50,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )

            # 清理图像
            for image_path in images:
                if os.path.exists(image_path):
                    os.remove(image_path)

            return {
                "num_images": num_images,
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text,
            }

        except Exception as e:
            # 清理图像
            for image_path in images:
                if os.path.exists(image_path):
                    os.remove(image_path)

            return {
                "num_images": num_images,
                "success": False,
                "error": str(e),
            }

    def test_01_request_within_limit(self):
        """测试1：请求数量在限制范围内"""
        print("\n=== 测试1：请求数量在限制范围内 ===")

        num_images = self.image_limit
        print(f"发送包含 {num_images} 张图像的请求（在限制范围内）")

        result = self._send_request_with_images(num_images)

        self.assertTrue(
            result["success"],
            f"请求失败: {result.get('error', result.get('response'))}",
        )
        print(f"✓ 请求成功，包含 {num_images} 张图像")

    def test_02_request_exceeds_limit(self):
        """测试2：请求数量超过限制"""
        print("\n=== 测试2：请求数量超过限制 ===")

        num_images = self.image_limit + 1
        print(f"发送包含 {num_images} 张图像的请求（超过限制）")

        result = self._send_request_with_images(num_images)

        # 请求应该被拒绝
        self.assertFalse(
            result["success"],
            "超过限制的请求应该被拒绝",
        )
        print(f"✓ 请求被正确拒绝，状态码: {result.get('status_code')}")

        # 验证错误信息中包含限制相关信息
        response_text = str(result.get("response", ""))
        self.assertTrue(
            "limit" in response_text.lower() or "exceed" in response_text.lower() or "too many" in response_text.lower(),
            f"错误信息中未包含限制相关信息: {response_text}",
        )

    def test_03_request_at_limit_boundary(self):
        """测试3：请求在限制边界"""
        print("\n=== 测试3：请求在限制边界 ===")

        # 测试刚好等于限制
        num_images = self.image_limit
        result = self._send_request_with_images(num_images)
        self.assertTrue(result["success"], f"边界请求失败: {result}")
        print(f"✓ 刚好等于限制 ({num_images} 张) 的请求成功")

        # 测试刚好超过限制
        num_images = self.image_limit + 1
        result = self._send_request_with_images(num_images)
        self.assertFalse(result["success"], "超过限制的请求应该被拒绝")
        print(f"✓ 刚好超过限制 ({num_images} 张) 的请求被拒绝")

    def test_04_single_image_request(self):
        """测试4：单图像请求（最小边界）"""
        print("\n=== 测试4：单图像请求 ===")

        num_images = 1
        print(f"发送包含 {num_images} 张图像的请求")

        result = self._send_request_with_images(num_images)

        self.assertTrue(
            result["success"],
            f"单图像请求失败: {result.get('error', result.get('response'))}",
        )
        print(f"✓ 单图像请求成功")

    def test_05_multiple_requests_within_limit(self):
        """测试5：多个请求都在限制范围内"""
        print("\n=== 测试5：多个请求都在限制范围内 ===")

        num_requests = 3
        num_images_per_request = self.image_limit

        print(f"发送 {num_requests} 个请求，每个包含 {num_images_per_request} 张图像")

        results = []
        for i in range(num_requests):
            result = self._send_request_with_images(num_images_per_request)
            results.append(result)

        successful = sum(1 for r in results if r["success"])
        print(f"✓ 成功: {successful}/{num_requests}")

        self.assertEqual(
            successful,
            num_requests,
            f"部分请求失败: {num_requests - successful}",
        )

    def test_06_zero_image_request(self):
        """测试6：零图像请求（纯文本）"""
        print("\n=== 测试6：零图像请求 ===")

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                }
            ],
            "max_tokens": 20,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )

            self.assertEqual(response.status_code, 200)
            print("✓ 纯文本请求成功")

        except Exception as e:
            self.fail(f"纯文本请求失败: {e}")

    def test_07_various_image_counts(self):
        """测试7：测试不同数量的图像"""
        print("\n=== 测试7：测试不同数量的图像 ===")

        test_counts = [1, self.image_limit, self.image_limit + 1, self.image_limit + 2]
        results = {}

        for count in test_counts:
            result = self._send_request_with_images(count)
            results[count] = result["success"]

            status = "成功" if result["success"] else "被拒绝"
            print(f"✓ {count} 张图像: {status}")

        # 验证结果
        for count, success in results.items():
            if count <= self.image_limit:
                self.assertTrue(success, f"{count} 张图像的请求应该成功")
            else:
                self.assertFalse(success, f"{count} 张图像的请求应该被拒绝")


if __name__ == "__main__":
    unittest.main(verbosity=2)
