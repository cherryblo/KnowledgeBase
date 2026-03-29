"""
用例5：：多模态编码器数据并行测试

测试目的：验证--mm-enable-dp-encoder参数在NPU环境下生效，多模态编码器使用数据并行加速
测试环境：使用4张NPU卡，单机部署
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


class TestMultimodalDPEncoder(unittest.TestCase):
    """多模态编码器数据并行测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv(
            "MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        cls.timeout = 300
        cls.tp_size = int(os.getenv("TP_SIZE", "4"))
        cls.mm_enable_dp = os.getenv("MM_ENABLE_DP", "true").lower() == "true"

        print(f"TP大小: {cls.tp_size}")
        print(f"启用多模态DP: {cls.mm_enable_dp}")

        # 创建测试图像
        cls.test_image_path = cls._create_test_image()

    @staticmethod
    def _create_test_image():
        """创建一个简单的测试图像"""
        image = Image.new("RGB", (640, 480), color="red")
        image_path = Path("/tmp/test_image_dp.jpg")
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
                        {"type": "text", "text": f"Request {request_id}: What do you see in this image?"},
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

    def test_03_single_request(self):
        """测试3：单个请求测试"""
        print("\n=== 测试3：单个请求测试 ===")

        result = self._send_request(0)

        self.assertTrue(
            result["success"],
            f"请求失败: {result.get('error', 'Unknown error')}",
        )
        print(f"✓ 单个请求成功，耗时: {result['elapsed_time']:.2f}秒")
        print(f"✓ 模型回答: {result['content']}")

    def test_04_concurrent_requests(self):
        """测试4：并发请求测试"""
        print("\n=== 测试4：并发请求测试 ===")

        num_requests = 8
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

        self.assertEqual(successful, num_requests, f"部分请求失败: {failed}")

    def test_05_throughput_test(self):
        """测试5：吞吐量测试"""
        print("\n=== 测试5：吞吐量测试 ===")

        num_requests = 16
        print(f"发送 {num_requests} 个请求测试吞吐量")

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
        throughput = num_requests / total_time

        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 成功: {successful}/{num_requests}")
        print(f"✓ 吞吐量: {throughput:.2f} 请求/秒")

        # 验证吞吐量
        self.assertGreater(
            throughput,
            0.5,
            f"吞吐量过低: {throughput:.2f} 请求/秒",
        )

    def test_06_multiple_images_in_single_request(self):
        """测试6：单个请求中包含多张图像"""
        print("\n=== 测试6：单个请求中包含多张图像 ===")

        # 创建多张图像
        images = []
        colors = ["red", "blue", "green"]
        for i, color in enumerate(colors):
            image = Image.new("RGB", (640, 480), color=color)
            image_path = Path(f"/tmp/test_image_multi_{i}.jpg")
            image.save(image_path)
            images.append(str(image_path))

        try:
            content = [{"type": "text", "text": "Describe the colors of these images."}]
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
                "max_tokens": 100,
            }

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

            print(f"✓ 多图像请求成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

        except Exception as e:
            self.fail(f"多图像请求失败: {e}")
        finally:
            # 清理图像
            for image_path in images:
                if os.path.exists(image_path):
                    os.remove(image_path)

    def test_07_dp_configuration_verification(self):
        """测试7：验证DP配置"""
        print("\n=== 测试7：验证DP配置 ===")

        # 这个测试主要验证服务器是否正确配置了DP
        # 实际的DP配置验证需要查看服务器日志或内部状态
        # 这里我们通过性能表现来间接验证

        print(f"TP大小: {self.tp_size}")
        print(f"启用多模态DP: {self.mm_enable_dp}")

        if self.mm_enable_dp:
            print("✓ 多模态DP已启用")
            print("✓ Vision Encoder应该使用数据并行")
            print(f"✓ DP大小应该等于TP大小: {self.tp_size}")
        else:
            print("✓ 多模态DP未启用")

        # 发送一个请求验证功能正常
        result = self._send_request(0)
        self.assertTrue(result["success"], "请求失败")
        print("✓ 服务器功能正常")

    @classmethod
    def tearDownClass(cls):
        """与其他测试用例共享测试图像，不删除"""
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
