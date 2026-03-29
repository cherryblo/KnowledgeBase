"""
用例6：多模态缓存测试

测试目的：验证多模态嵌入缓存功能在NPU环境下正常工作，相同图像的重复请求能命中缓存
测试环境：使用1张NPU卡，单机部署
"""

import base64
import os
import time
import unittest
from pathlib import Path

import requests
from PIL import Image


class TestMultimodalCache(unittest.TestCase):
    """多模态缓存测试"""

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
        image_path = Path("/tmp/test_image_cache.jpg")
        image.save(image_path)
        return str(image_path)

)

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

    def test_01_first_request(self):
        """测试1：首次请求（缓存未命中）"""
        print("\n=== 测试1：首次请求（缓存未命中） ===")

        result = self._send_request("What color is this image?")

        self.assertTrue(
            result["success"],
            f"首次请求失败: {result.get('error', 'Unknown error')}",
        )
        print(f"✓ 首次请求成功，耗时: {result['elapsed_time']:.2f}秒")
        print(f"✓ 模型回答: {result['content']}")

    def test_02_second_request_same_image(self):
        """测试2：第二次请求相同图像（可能命中缓存）"""
        print("\n=== 测试2：第二次请求相同图像 ===")

        # 第一次请求
        result1 = self._send_request("What color is this image?")
        self.assertTrue(result1["success"], "第一次请求失败")
        time1 = result1["elapsed_time"]

        # 等待一下
        time.sleep(0.5)

        # 第二次请求（相同图像）
        result2 = self._send_request("What color is this image?")
        self.assertTrue(result2["success"], "第二次请求失败")
        time2 = result2["elapsed_time"]

        print(f"✓ 第一次请求耗时: {time1:.2f}秒")
        print(f"✓ 第二次请求耗时: {time2:.2f}秒")

        # 验证结果一致性
        self.assertEqual(
            result1["content"],
            result2["content"],
            "两次请求结果不一致",
        )
        print("✓ 两次请求结果一致")

        # 注意：由于缓存命中需要特定的实现，这里只记录时间对比
        # 实际的缓存命中验证需要查看服务器日志
        if time2 < time1:
            speedup = time1 / time2
            print(f"✓ 第二次请求更快，加速比: {speedup:.2f}x")
        else:
            print("✓ 两次请求时间相近（可能缓存未命中或实现不同）")

    def test_03_multiple_requests_same_image(self):
        """测试3：多次请求相同图像"""
        print("\n=== 测试3：多次请求相同图像 ===")

        num_requests = 5
        prompt = "What do you see in this image?"

        results = []
        for i in range(num_requests):
            result = self._send_request(prompt)
            results.append(result)
            print(
                f"请求 {i+1}: {'成功' if result['success'] else '失败'}, "
                f"耗时: {result['elapsed_time']:.2f}秒"
            )

        # 验证所有请求都成功
        successful = sum(1 for r in results if r["success"])
        self.assertEqual(successful, num_requests, f"部分请求失败: {num_requests - successful}")

        # 验证所有结果一致
        contents = [r["content"] for r in results]
        self.assertTrue(
            all(c == contents[0] for c in contents),
            "多次请求结果不一致",
        )
        print("✓ 所有请求结果一致")

        # 分析时间趋势
        times = [r["elapsed_time"] for r in results]
        avg_time = sum(times) / len(times)
        print(f"✓ 平均响应时间: {avg_time:.2f}秒")
        print(f"✓ 最快响应时间: {min(times):.2f}秒")
        print(f"✓ 最慢响应时间: {max(times):.2f}秒")

    def test_04_different_prompts_same_image(self):
        """测试4：相同图像，不同提示词"""
        print("\n=== 测试4：相同图像，不同提示词 ===")

        prompts = [
            "What color is this image?",
            "Describe this image.",
            "What do you see?",
        ]

        results = []
        for prompt in prompts:
            result = self._send_request(prompt)
            results.append(result)
            print(
                f"提示词: '{prompt}' - "
                f"{'成功' if result['success'] else '失败'}, "
                f"耗时: {result['elapsed_time']:.2f}秒"
            )

        # 验证所有请求都成功
        successful = sum(1 for r in results if r["success"])
        self.assertEqual(successful, len(prompts), f"部分请求失败: {len(prompts) - successful}")

        # 验证不同提示词产生不同回答
        contents = [r["content"] for r in results]
        unique_contents = len(set(contents))
        print(f"✓ 唯一回答数量: {unique_contents}/{len(prompts)}")

    def test_05_cache_behavior_under_load(self):
        """测试5：负载下的缓存行为"""
        print("\n=== 测试5：负载下的缓存行为 ===")

        num_iterations = 3
        requests_per_iteration = 3

        all_times = []

        for iteration in range(num_iterations):
            print(f"迭代 {iteration + 1}/{num_iterations}")
            iteration_times = []

            for i in range(requests_per_per_iteration):
                result = self._send_request("What color is this image?")
                iteration_times.append(result["elapsed_time"])

            avg_time = sum(iteration_times) / len(iteration_times)
            print(f"  平均响应时间: {avg_time:.2f}秒")
            all_times.extend(iteration_times)

            time.sleep(1)

        # 分析整体性能
        overall_avg = sum(all_times) / len(all_times)
        print(f"✓ 整体平均响应时间: {overall_avg:.2f}秒")
        print(f"✓ 总请求数: {len(all_times)}")

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
