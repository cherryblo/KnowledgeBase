"""
用例2：多模态并发处理测试

测试目的：验证--mm-max-concurrent-calls参数在NPU环境下生效，能够控制并发处理的多模态请求数量
测试环境：使用1张NPU卡，单机部署
"""

import base64
import concurrent.futures
import json
import os
import time
import unittest
from pathlib import Path

import requests
from PIL import Image


class TestConcurrentMultimodal(unittest.TestCase):
    """多模态并发处理测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv(
            "MODEL_PATH", "microsoft/Phi-4-multimodal-instruct"
        )
        cls.timeout = 300
        cls.max_concurrent_calls = int(
            os.getenv("MAX_CONCURRENT_CALLS", "4")
        )

        # 创建测试图像
        cls.test_image_path = cls._create_test_image()

    @staticmethod
    def _create_test_image():
        """创建一个简单的测试图像"""
        image = Image.new("RGB", (640, 480), color="red")
        image_path = Path("/tmp/test_image_concurrent.jpg")
        image.save(image_path)
        return str(image_path)

    def _encode_image(self, image_path):
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _send_request(self, request_id):
        """发送单个多模态请求"""
        image_base64 = self._encode_image(self.test_image_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": f"Request {request_id}: What color is this image?"},
                    ],
                }
            ],
            "max_tokens": 20,
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
                    "request_id": request_id,
                    "success": True,
                    "elapsed_time": elapsed_time,
                    "content": content,
                }
            else:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": response.text,
                }
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
            }

    def test_01_concurrent_requests_within_limit(self):
        """测试1：并发请求数在限制范围内"""
        print("\n=== 测试1：并发请求数在限制范围内 ===")
        print(f"最大并发数限制: {self.max_concurrent_calls}")

        num_requests = self.max_concurrent_calls
        print(f"发送 {num_requests} 个并发请求")

        results = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_requests
        ) as executor:
            futures = [
                executor.submit(self._send_request, i) for i in range(num_requests)
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

        total_time = time.time() - start_time

        # 统计结果
        successful = sum(1 for r in results if r["success"])
        failed = num_requests - successful

        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 成功: {successful}/{num_requests}")
        print(f"✓ 失败: {failed}/{num_requests}")

        self.assertEqual(successful, num_requests, f"部分请求失败: {failed}")

    def test_02_concurrent_requests_exceed_limit(self):
        """测试2：并发请求数超过限制"""
        print("\n=== 测试2：并发请求数超过限制 ===")
        print(f"最大并发数限制: {self.max_concurrent_calls}")

        num_requests = self.max_concurrent_calls * 2
        print(f"发送 {num_requests} 个并发请求（超过限制）")

        results = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_requests
        ) as executor:
            futures = [
                executor.submit(self._send_request, i) for i in range(num_requests)
)
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

        total_time = time.time() - start_time

        # 统计结果
        successful = sum(1 for r in results if r["success"])
        failed = num_requests - successful

        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 成功: {successful}/{num_requests}")
        print(f"✓ 失败: {failed}/{num_requests}")

        # 验证所有请求最终都成功（只是排队处理）
        self.assertEqual(successful, num_requests, f"部分请求失败: {failed}")

    def test_03_sequential_vs_concurrent_comparison(self):
        """测试3：顺序请求 vs 并发请求性能对比"""
        print("\n=== 测试3：顺序请求 vs 并发请求性能对比 ===")

        num_requests = 4

        # 顺序请求
        print(f"顺序执行 {num_requests} 个请求...")
        sequential_results = []
        sequential_start = time.time()

        for i in range(num_requests):
            result = self._send_request(f"seq_{i}")
            sequential_results.append(result)

        sequential_time = time.time() - sequential_start

        # 并发请求
        print(f"并发执行 {num_requests} 个请求...")
        concurrent_results = []
        concurrent_start = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_requests
        ) as executor:
            futures = [
                executor.submit(self._send_request, f"conc_{i}")
                for i in range(num_requests)
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                concurrent_results.append(result)

        concurrent_time = time.time() - concurrent_start

        print(f"✓ 顺序执行耗时: {sequential_time:.2f}秒")
        print(f"✓ 并发执行耗时: {concurrent_time:.2f}秒")
        print(f"✓ 性能提升: {sequential_time / concurrent_time:.2f}x")

        # 并发应该比顺序快（即使有并发限制）
        self.assertLess(
            concurrent_time,
            sequential_time,
            "并发执行应该比顺序执行快",
        )

    def test_04_high_load_stress_test(self):
        """测试4：高负载压力测试"""
        print("\n=== 测试4：高负载压力测试 ===")

        num_requests = 20
        print(f"发送 {num_requests} 个请求进行压力测试")

        results = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_requests
        ) as executor:
            futures = [
                executor.submit(self._send_request, i) for i in range(num_requests)
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)

        total_time = time.time() - start_time

        # 统计结果
        successful = sum(1 for r in results if r["success"])
        failed = num_requests - successful
        avg_time = (
            sum(r["elapsed_time"] for r in results if r["success"]) / successful
            if successful > 0
            else 0
        )

        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 成功: {successful}/{num_requests}")
        print(f"✓ 失败: {failed}/{num_requests}")
        print(f"✓ 平均响应时间: {avg_time:.2f}秒")
        print(f"✓ 吞吐量: {num_requests / total_time:.2f} 请求/秒")

        # 验证成功率
        success_rate = successful / num_requests
        self.assertGreater(
            success_rate,
            0.9,
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
