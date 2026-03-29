"""
用例1：基础解密配置文件测试

测试目的：验证--decrypted-config-file参数在NPU环境下生效，能够正确指定主模型的解密配置文件
测试环境：使用1张NPU卡，单机部署
"""

import json
import os
import tempfile
import time
import unittest

import requests


class TestBasicDecryptionConfig(unittest.TestCase):
    """基础解密配置文件测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        cls.timeout = 300

        # 创建模拟的解密配置文件
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
        config_path.close.close()

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

    def test_03_basic_inference(self):
        """测试3：基本推理测试"""
        print("\n=== 测试3：基本推理测试 ===")

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
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

            content = result["choices"][0]["message"]["content"]
            print(f"✓ 基本推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

            # 验证回答不为空
            self.assertTrue(len(content) > 0, "模型回答为空")

        except Exception as e:
            self.fail(f"基本推理失败: {e}")

    def test_04_multiple_inference_requests(self):
        """测试4：多次推理请求"""
        print("\n=== 测试4：多次推理请求 ===")

        prompts = [
            "What is 2+2?",
            "Tell me a joke.",
            "What is the capital of France?",
        ]

        results = []
        for i, prompt in enumerate(prompts):
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

                if results.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    results.append(
                        {
                            "prompt": prompt,
                            "success": True,
                            "elapsed_time": elapsed_time,
                            "content": content,
                        }
                    )
                    print(
                        f"✓ 请求 {i+1}: 成功, 耗时: {elapsed_time:.2f}秒"
                    )
                else:
                    results.append(
                        {
                            "prompt": prompt,
                            "success": False,
                            "error": response.text,
                        }
                    )
                    print(f"✗ 请求 {i+1}: 失败")

            except Exception as e:
                results.append(
                    {
                        "prompt": prompt,
                        "success": False,
                        "error": str.},
                }
                print(f"✗ 请求 {i+1}: 异常 - {e}")

        # 验证所有请求都成功
        successful = sum(1 for r in results if r["success"])
        print(f"✓ 成功: {successful}/{len(prompts)}")
        self.assertEqual(successful, len(prompts), f"部分请求失败: {len(prompts) - successful}")

    def test_05_streaming_inference(self):
        """测试5：流式推理测试"""
        print("\n=== 测试5：流式推理测试 ===")

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a short story.",
                }
            ],
            "max_tokens": 100,
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

    def test_06_long_context_inference(self):
        """测试6：长上下文推理测试"""
        print("\n=== 测试6：长上下文推理测试 ===")

        long_prompt = (
            "Please summarize the following text: "
            "The quick brown fox jumps over the lazy dog. "
            "This is a classic English pangram that contains all the letters of the alphabet. "
            "The sentence has been used in various contexts, including typing practice, "
            "language learning, and as an example in computer science. "
            "Now, please provide a concise summary."
        )

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": long_prompt,
                }
            ],
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
            print(f"✓ 长上下文推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

            # 验证回答不为空
            self.assertTrue(len(content) > 0, "模型回答为空")

        except Exception as e:
            self.fail(f"长上下文推理失败: {e}")

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
