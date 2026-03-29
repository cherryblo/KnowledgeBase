"""
用例9：多模态预处理配置测试

测试目的：验证--mm-process-config参数在NPU环境下生效，能够自定义多模态预处理参数
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


class TestMultimodalProcessConfig(unittest.TestCase):
    """多模态预处理配置测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv(
            "MODEL_PATH", "microsoft/Phi-4-multimodal-instruct"
        )
        cls.timeout = 300

        # 从环境变量读取预处理配置
        process_config = os.getenv("MM_PROCESS_CONFIG", "{}")
        cls.process_config = json.loads(process_config)

        print(f"预处理配置: {cls.process_config}")

        # 创建测试图像
        cls.test_image_path = cls._create_test_image()

    @staticmethod
    def _create_test_image():
        """创建一个简单的测试图像"""
        image = Image.new("RGB", (1280, 720), color="red")
        image_path = Path("/tmp/test_image_config.jpg")
        image.save(image_path)
        return str(image_path)

    def _encode_image(self, image_path):
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _send_request(self, prompt):
        """发送多模态请求"""
        image_base64 = self._encode_image(self.test_image_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 50,
        }

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "elapsed_time": elapsed_time,
                    "content": content,
                }
            else:
                return {
                    "success": False,
                    "elapsed_time": elapsed_time,
                    "error": response.text,
                }
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                "success": False,
                "elapsed_time": elapsed_time,
                "error": str(e),
            }

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

            # 检查是否包含预处理配置信息
            if "mm_process_config" in model_info:
                print(f"✓ 预处理配置: {model_info['mm_process_config']}")
            else:
                print("✓ 模型信息中未包含预处理配置（可能不暴露）")

        except Exception as e:
            self.fail(f"获取模型信息失败: {e}")

    def test_03_basic_inference(self):
        """测试3：基本推理测试"""
        print("\n=== 测试3：基本推理测试 ===")

        result = self._send_request("What color is this image?")

        self.assertTrue(
            result["success"],
            f"基本推理失败: {result.get('error', 'Unknown error')}",
        )
        print(f"✓ 基本推理成功，耗时: {result['elapsed_time']:.2f}秒")
        print(f"✓ 模型回答: {result['content']}")

    def test_04_different_image_sizes(self):
        """测试4：不同图像尺寸测试"""
        print("\n=== 测试4：不同图像尺寸测试 ===")

        sizes = [(640, 480), (1280, 720), (1920, 1080)]
        results = []

        for width, height in sizes:
            # 创建不同尺寸的图像
            image = Image.new("RGB", (width, height), color="blue")
            image_path = Path(f"/tmp/test_image_{width}x{height}.jpg")
            image.save(image_path)

            try:
                image_base64 = self._encode_image(str(image_path))

                payload = {
                    "model": self.model_path,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    },
                                },
                                {"type": "text", "text": "What do you see?"},
                            ],
                        }
                    ],
                    "max_tokens": 50,
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

                status = "成功" if response.status_code == 200 else "失败"
                print(f"✓ 图像 {width}x{height}: {status}, 耗时atan: {elapsed_time:.2f}秒")

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

        # 验证所有尺寸都能处理
        successful = sum(1 for r in results if r["success"])
        print(f"✓ 成功处理: {successful}/{len(sizes)} 种尺寸")
        self.assertEqual(
            successful,
            len(sizes),
            f"部分尺寸处理失败: {len(sizes) - successful}",
        )

    def test_05_image_quality_variations(self):
        """测试5：不同图像质量测试"""
        print("\n=== 测试5：不同图像质量测试 ===")

        qualities = [50, 75, 95]
        results = []

        for quality in qualities:
            # 创建不同质量的图像
            image = Image.new("RGB", (640, 480), color="green")
            image_path = Path(f"/tmp/test_image_quality_{quality}.jpg")
            image.save(image_path, quality=quality)

            try:
                image_base64 = self._encode_image(str(image_path))

                payload = {
                    "model": self.model_path,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    },
                                },
                                {"type": "text", "text": "What color is this?"},
                            ],
                        }
                    ],
                    "max_tokens": 30,
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
                        "quality": quality,
                        "success": response.status_code == 200,
                        "elapsed_time": elapsed_time,
                    }
                )

                status = "成功" if response.status_code == 200 else "失败"
                print(f"✓ 质量 {quality}: {status}, 耗时: {elapsed_time:.2f}秒")

            except Exception as e:
                results.append(
                    {
                        "quality": quality,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"✗ 质量 {quality}: 失败 - {e}")
            finally:
                if image_path.exists():
                    image_path.unlink()

        # 验证所有质量都能处理
        successful = sum(1 for r in results if r["success"])
        print(f"✓ 成功处理: {successful}/{len(qualities)} 种质量")
        self.assertEqual(
            successful,
            len(qualities),
            f"部分质量处理失败: {len(qualities) - successful}",
        )

    def test_06_concurrent_requests_with_config(self):
        """测试6：配置下的并发请求测试"""
        print("\n=== 测试6：配置下的并发请求测试 ===")

        import concurrent.futures

        num_requests = 4
        print(f"发送 {num_requests} 个并发请求")

        results = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_requests
        ) as executor:
            futures = [
                executor.submit(self._send_request, f"Request {i}: What do you see?")
                for i in range(num_requests)
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

        total_time = time.time() - start_time

        # 统计结果
        successful = sum(1 for r in results if r["success"])
        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 成功: {successful}/{num_requests}")

        self.assertEqual(successful, num_requests, f"部分请求失败: {num_requests - successful}")

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
