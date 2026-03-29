"""
用例10：并发加载测试

测试目的：验证在并发场景下，解密配置的正确加载
测试环境：使用1张NPU卡，单机部署
"""

import json
import os
import tempfile
import time
import unittest

import requests


class TestConcurrentLoading(unittest.TestCase):
    """并发加载测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        cls.timeout = 300

        # 创建解密配置文件
        cls.decryption_config_path = cls._create_decryption_config()

    @staticmethod
    def _create_decryption_config():
        """创建一个模拟的解密配置文件"""
        config = {
            "model_type": "qwen2",
            "quantization_config": {
                "quant_method": "w8a8",
                "group_size": 128,
            },
            "attention_config": {
                "use_flash_attention": True,
            },
        }

        config_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(config, config_path, indent=2)
        config_path.close()

        return config_path.name

    def test_01_server_health_check(self):
        """测试1：服务器健康检查"""
        print("\n=== 测试1：服务器健康检查 ===")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            self.assertEqual(response.status_code, 200)
            print("✓ 服务器健康检查通过")
        except Exception as e:
            self.fail(f"服务器健康检查失败: {e}")

    def test_02_concurrent_requests(self):
        """测试2：并发请求测试"""
        print("\n=== 测试2：并发请求测试 ===")

        import concurrent.futures

        def send_request(prompt):
        payload = {
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
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
                        "error": response.text,
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

        prompts = [
            "What is AI?",
            "Explain machine learning.",
            "What is deep learning?",
            "Define neural networks.",
            "What is natural language processing?",
            "Explain computer vision.",
            "What is reinforcement learning?",
            "What is a neural network?",
        ]

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(prompts)
        ) as executor:
            futures = [
                executor.submit(send_request, prompt) for prompt in prompts
            ]

            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # 统计结果
        successful = sum(1 for r in results if r["success"])
        avg_time = (
            sum(r["elapsed_time"] for r in results if r["success"]) / successful
            if successful > 0
            else 0
        )

        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 成功: {successful}/{len(prompts)}")
        print(f"✓ 平均响应时间: {avg_time:.2f}秒")
        print(f"✓ 吞吐量: {len(prompts) / total_time:.2f} 请求/秒")

        self.assertEqual(successful, len(prompts), f"部分请求失败: {len(prompts) - successful}")

    def test_03_concurrent_streaming_requests(self):
        """测试3：并发流式请求测试"""
        print("\n=== 测试3：并发流式请求测试 ===")

        import concurrent.futures

        def send_streaming_request(prompt):
            payload = {
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": 30,
                "stream": True,
            }

            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                )

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

                elapsed_time = time.time() - start_time

                return {
                    "success": True,
                    "elapsed_time": elapsed_time,
                    "content": full_content,
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

        prompts = [
            "Tell me a joke.",
            "What is the capital of France?",
            "Explain gravity.",
            "What is photosynthesis?",
        ]

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(prompts)
        ) as executor:
            futures = [
                executor.submit(send_streaming_request, prompt)
                for prompt in prompts
            ]

            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # 统计结果
        successful = sum(1 for r in results if r["success"])
        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 成功: {successful}/{len(prompts)}")

        self.assertEqual(successful, len(prompts), f"部分请求失败: {len(prompts) - successful}")

    def test_04_mixed_concurrent_requests(self):
        """测试4：混合并发请求测试（流式+非流式）"""
        print("\n=== 测试4：混合并发请求测试 ===")

        import concurrent.futures

        def send_request(prompt, stream=False):
            payload = {
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": 50,
                "stream": stream,
            }

            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )

                if stream:
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
                    content = full_content
                else:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]

                elapsed_time = time.time() - start_time

                return {
                    "success": True,
                    "elapsed_time": elapsed_time,
                    "content": content,
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

        requests_data = [
            ("What is AI?", False),
            ("Tell me a story.", True),
            ("What is deep learning?", False),
            ("Explain quantum computing.", True),
        ]

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(requests_data)
        ) as executor:
            futures = [
                executor.submit(send_request, prompt, stream)
                for prompt, stream in requests_data
            ]

            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # 统计结果
        successful = sum(1 for r in results if r["success"])
        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 成功: {successful}/{len(requests_data)}")

        self.assertEqual(successful, len(requests_data), f"部分请求失败: {len(requests_data) - successful}")

    def test_05_high_load_concurrent_test(self):
        """测试5：高负载并发测试"""
        print("\n=== 测试5：高负载并发测试 ===")

        import concurrent.futures

        def send_request(request_id):
            payload = {
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Request {request_id}: What is 2+2?",
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
                        "error": response.text,
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

        num_requests = 20
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=10
        ) as executor:
            futures = [
                executor.submit(send_request, i) for i in range(num_requests)
            ]

            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # 统计结果
        successful = sum(1 for r in results if r["success"])
        avg_time = (
            sum(r["elapsed_time"] for r in results if r["success"]) / successful
            if successful > 0
            else 0
        )

        print(f"✓ 请求数量: {num_requests}")
        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 成功: {successful}/{num_requests}")
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
            if os.path.exists(cls.decryption_config_path):
                os.remove(cls.decryption_config_path)
        except Exception as e:
            print(f"清理测试文件时出错: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
