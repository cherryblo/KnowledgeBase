"""
用例3：仅主模型解密配置测试

测试目的：验证只设置主模型解密配置而不设置draft模型配置时的行为
测试环境：使用1张NPU卡，单机部署
"""

import json
import os
import tempfile
import time
import unittest

import requests


class TestMainOnlyDecryption(unittest.TestCase):
    """仅主模型解密配置测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        cls.timeout = 300

        # 仅创建主模型的解密配置文件
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

    def test_04_multiple_prompts(self):
        """测试4：多个提示词测试"""
        print("\n=== 测试4：多个提示词测试 ===")

        prompts = [
            "What is the capital of China?",
            "Who wrote Romeo and Juliet?",
            "What is the formula for water?",
            "Explain the theory of relativity.",
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

                if response.status_code == 200:
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
                        f"✓ 提示词 {i+1}: 成功, 耗时: {elapsed_time:.2f}秒"
                    )
                else:
                    results.append(
                        {
                            "prompt": prompt,
                            "success": False,
                            "error": response.text,
                        }
                    )
                    print(f"✗ 提示词 {i+1}: 失败")

            except Exception as e:
                results.append(
                    {
                        "prompt": prompt,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"✗ 提示词 {i+1}: 异常 - {e}")

        # 验证所有请求都成功
        successful = sum(1 for r in results if r["success"])
        print(f"✓ 成功: {successful}/{len(prompts)}")
        self.assertEqual(successful, len(prompts), f"部分请求失败: {len(prompts) - successful}")

    def test_05_conversation_history(self):
        """测试5：对话历史测试"""
        print("\n=== 测试5：对话历史测试 ===")

        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What is my name?"},
        ]

        payload = {
            "model": self.model_path,
            "messages": messages,
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
            print(f"✓ 对话历史推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

            # 验证回答中包含名字
            self.assertTrue(
                "Alice" in content.lower(),
                f"回答中未检测到名字: {content}",
            )

        except Exception as e:
            self.fail(f"对话历史推理失败: {e}")

    def test_06_system_prompt(self):
        """测试6：系统提示词测试"""
        print("\n=== 测试6：系统提示词测试 ===")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that always responds in Chinese.",
            },
            {"role": "user", "content": "Hello!"},
        ]

        payload = {
            "model": self.model_path,
            "messages": messages,
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
            print(f"✓ 系统提示词推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

            # 验证回答不为空
            self.assertTrue(len(content) > 0, "模型回答为空")

        except Exception as e:
            self.fail(f"系统提示词推理失败: {e}")

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
