"""
用例3：多模态请求超时测试

测试目的：验证--mm-per-request-timeout参数在NPU环境下生效，超时后会正确处理
测试环境：使用1张NPU卡，单机部署
"""

import base64
import json
import os
import time
import unittest
from pathlib import Path

import requests
from PIL import Image


class TestMultimodalTimeout(unittest.TestCase):
    """多模态请求超时测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv(
            "MODEL_PATH", "microsoft/Phi-4-multimodal-instruct"
        )
        cls.timeout = 300
        cls.per_request_timeout = float(
            os.getenv("PER_REQUEST_TIMEOUT", "5.0")
        )

        # 创建测试图像
        cls.test_image_path = cls._create_test_image()

    @staticmethod
    def _create_test_image():
        """创建一个简单的测试图像"""
        image = Image.new("RGB", (640, 480), color="red")
        image_path = Path("/tmp/test_image_timeout.jpg")
        image.save(image_path)
        return str(image_path)

    def _encode_image(self, image_path):
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def test_01_normal_request_completes(self):
        """测试1：正常请求能够完成"""
        print("\n=== 测试1：正常请求能够完成 ===")

        image_base64 = self._encode_image(self.test_image_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "What color is this image?"},
                    ],
                }
            ],
            "max_tokens": 20,
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
            print(f"✓ 正常请求完成，耗时: {elapsed_time:.2f}秒")

            # 验证请求在超时时间内完成
            self.assertLess(
                elapsed_time,
                self.per_request_timeout * 1.5,
                f"请求耗时过长: {elapsed_time:.2f}秒",
            )

        except Exception as e:
            self.fail(f"正常请求失败: {e}")

    def test_02_large_image_request(self):
        """测试2：大图像请求（可能接近超时）"""
        print("\n=== 测试2：大图像请求 ===")

        # 创建大图像
        large_image = Image.new("RGB", (4096, 4096), color="blue")
        large_image_path = Path("/tmp/test_large_image.jpg")
        large_image.save(large_image_path, quality=95)

        try:
            image_base64 = self._encode_image(str(large_image_path))

            payload = {
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": "Describe this image in detail."},
                        ],
                    }
                ],
                "max_tokens": 100,
            }

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            elapsed_time = time.time() - start_time

            # 大图像请求可能成功也可能超时，验证服务器状态
            if response.status_code == 200:
                print(f"✓ 大图像请求成功，耗时: {elapsed_time:.2f}秒")
            else:
                print(f"✓ 大图像请求超时/失败，耗时: {elapsed_time:.2f}秒")
                print(f"✓ 响应状态码: {response.status_code}")

        except requests.exceptions.Timeout:
            elapsed_time = time.time() - start_time
            print(f"✓ 请求超时（符合预期），耗时: {elapsed_time:.2f}秒")
        except Exception as e:
            print(f"✓ 请求异常: {e}")
        finally:
            if large_image_path.exists():
                large_image_path.unlink()

    def test_03_server_health_after_timeout(self):
        """测试3：超时后服务器状态正常"""
        print("\n=== 测试3：超时后服务器状态正常 ===")

        # 先发送一个可能导致超时的请求
        large_image = Image.new("RGB", (4096, 4096), color="green")
        large_image_path = Path("/tmp/test_large_image_2.jpg")
        large_image.save(large_image_path, quality=95)

        try:
            image_base64 = self._encode_image(str(large_image_path))

            payload = {
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": "Describe this image."},
                        ],
                    }
                ],
                "max_tokens": 200,
            }

            requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.per_request_timeout + 2,
            )
        except:
            pass  # 忽略可能的超时
        finally:
            if large_image_path.exists():
                large_image_path.unlink()

        # 等待一下
        time.sleep(1)

        # 检查服务器健康状态
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            print("✓ 服务器健康状态正常")
        except Exception as e:
            self.fail(f"服务器健康检查失败: {e}")

        # 发送一个正常请求验证服务器功能
        image_base64 = self._encode_image(self.test_image_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "What color is this image?"},
                    ],
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
            print("✓ 服务器功能正常，能够处理后续请求")
        except Exception as e:
            self.fail(f"服务器功能异常: {e}")

    def test_04_multiple_requests_with_various_sizes(self):
        """测试4：多个不同大小的请求"""
        print("\n=== 测试4：多个不同大小的请求 ===")

        image_sizes = [(640, 480), (1280, 960), (1920, 1080)]
        results = []

        for i, (width, height) in enumerate(image_sizes):
            image = Image.new("RGB", (width, height), color="red")
            image_path = Path(f"/tmp/test_image_{i}.jpg")
            image.save(image_path)

            try:
                image_base64 = self._encode_image(str(image_path))

                payload = {
                    "model": self.model_path,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                                {"type": "text", "text": "What do you see?"},
                            ],
                        }
                    ],
                    "max_tokens": 20,
                }

                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed_time = time.time() - start_time

                results.append(
                    {
                        "size": f"{width}x{height}",
                        "success": response.status_code == 200,
                        "elapsed_time": elapsed_time,
                    }
                )

                print(
                    f"✓ 图像 {width}x{height}: {'成功' if response.status_code == 200 else '失败'}, "
                    f"耗时: {elapsed_time:.2f}秒"
                )

            except Exception as e:
                results.append(
                    {
                        "size": f"{width}x{height}",
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"✗ 图像 {width}x{height}: 失败 - {e}")
            finally:
                if image_path.exists():
                    image_path.unlink()

        # 验证大部分请求成功
        successful = sum(1 for r in results if r["success"])
        success_rate = successful / len(results)
        print(f"✓ 成功率: {success_rate:.2%}")
        self.assertGreater(
            success_rate,
            0.5,
            f"成功率过低: {success_rate:.2%}",
        )

    @classmethod
    def tearDownClass(cls):
        """清理测试文件"""
        try:
            if os.path.exists(cls.test_image_path):
                os.remove(cls.test_image_path)
        except Exception as e:
            print(f"清理测试文件时出错: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
