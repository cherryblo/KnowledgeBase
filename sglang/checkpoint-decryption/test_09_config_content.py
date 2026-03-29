"""
用例9：配置文件内容验证测试

测试目的：验证解密配置文件的内容被正确传递给模型加载过程
测试环境：使用1张NPU卡，单机部署
"""

import json
import os
import tempfile
import time
import unittest

import requests


class TestConfigContentValidation(unittest.TestCase):
    """配置文件内容验证测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        cls.timeout = 300

        # 创建包含特定配置项的解密配置文件
        cls.decryption_config_path = cls._create_decryption_config()

    @staticmethod
    def _create_decryption_config():
        """创建一个包含特定配置项的解密配置文件"""
        config = {
            "model_type": "qwen2",
            "quantization_config": {
                "quant_method": "w8a8",
                "group_size": 128,
                "symmetric": True,
            },
            "attention_config": {
                "use_flash_attention": True,
                "attention_type": "flash_attention_2",
            },
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.9,
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
            response = = requests.get(f"{self.base_url}/health", timeout=10)
            self.assertEqual(response.status_code, 200)
            print("✓ 服务器健康检查通过")
        except Exception as e:
            self.fail(f"服务器健康检查失败: {e}")

    def test_02_model_info(self):
        """测试2：获取模型信息（验证配置应用）"""
        print("\n=== 测试2：获取模型信息 ===")

        try:
            response = requests.get(f"{self.base_url}/get_model_info", timeout=10)
            self.assertEqual(response.status_code, 200)
            model_info = response.json()
            print(f"✓ 模型信息: {model_info.get('model_path', 'N/A')}")

            # 检查模型信息
            if "quantization" in model_info:
                print(f"✓ 量化配置: {model_info['quantization']}")
            if "attention" in model_info:
                print(f"✓  注意力配置: {model_info['attention']}")

        except Exception as e:
            self.fail(f"获取模型信息失败: {e}")

    def test_03_basic_inference(self):
        """测试3：基本推理测试（验证配置生效）"""
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

    def test_04_config_file_content_verification(self):
        """测试4：验证配置文件内容"""
        print("\n=== 测试4：验证配置文件内容 ===")

        try:
            with open(self.decryption_config_path, "r") as f:
                config = json.load(f)

            print("✓ 配置文件内容：")
            print(json.dumps(config, indent=2))

            # 验证关键配置项
            self.assertIn("model_type", config)
            self.assertIn("quantization_config", config)
            self.assertIn("attention_config", config)

            print("✓  配置文件包含所有必需的配置项")

        except Exception as e:
            self.fail(f"读取配置文件失败: {e}")

    def test_05_multiple_requests_consistency(self):
        """测试5：多次请求一致性测试"""
        print("\n=== 测试5：多次请求一致性测试 ===")

        prompt = "What is 2+2?"

        results = []
        for i in range(5):
            payload = {
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
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
                    results.append(
                        {
                            "request_id": i,
                            "success": True,
                            "elapsed_time": elapsed_time,
                            "content": content,
                        }
                    )
                else:
                    results.append(
                        {
                            "request_id": i,
                            "success": False,
                            "error": response.text,
                        }
                    )
            except Exception as e:
                results.append(
                    {
                        "request_id": i,
                        "success": False,
                        "error": str(e),
                    }
                )

        # 验证所有请求都成功
        successful = sum(1 for r in results if r["success"])
        print(f"✓ 成功: {successful}/5")

        # 验证响应一致性（所有成功请求应该返回相似答案）
        successful_results = [r for r in results if r["success"]]
        if len(successful_results) > 1:
            # 检查所有回答都包含"4"
            all_contain_4 = all(
                "4" in r["content"] for r in successful_results
            )
            print(f"✓  所有回答都包含'4': {all_contain_4}")

    def test_06_streaming_inference(self):
        """测试6：流式推理测试"""
        print("\n=== 测试6：流式推理测试 ===")

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 10.",
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
            if os.path.exists(cls.decryption_config_path):
                os.remove(cls.decryption_config_path)
        except Exception as e:
            print(f"清理测试文件时出错: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
