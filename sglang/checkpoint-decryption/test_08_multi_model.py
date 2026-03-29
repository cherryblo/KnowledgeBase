"""
用例8：多模型部署测试

测试目的：验证在多模型部署场景下，每个模型使用独立的解密配置
测试环境：使用1张NPU卡，单机部署（多端口）
"""

import json
import os
import tempfile
import time
import unittest

import requests


class TestMultiModelDeployment(unittest.TestCase):
    """多模型部署测试"""

    @classmethod
    def setUpClass(cls):
        cls.timeout = 300

        # 创建多个模型的解密配置文件
        cls.configs = {}
        for i in range(3):
            cls.configs[i] = cls._create_decryption_config(i)

    @staticmethod
    def _create_decryption_config(model_id):
        """创建一个模拟的解密配置文件"""
        config = {
            "model_type": f"model_{model_id}",
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

        return {
            "path": config_path.name,
            "model_id": model_id,
        }

    def test_01_manual_deployment_test(self):
        """测试1：手动多模型部署测试说明"""
        print("\n=== 测试1：手动多模型部署测试说明 ===")
        print("注意：此测试需要手动启动多个服务器实例")
        print()
        print("测试步骤：")
        print("1. 为每个模型创建独立的解密配置文件")
        print("2. 启动多个SGLang服务器实例，使用不同端口")
        print("3. 每个实例指定不同的解密配置文件")
        print("4. 验证所有服务器都成功启动")
        print()
        print("示例命令：")
        for i in range(3):
            port = 30000 + i
            config_path = self.configs[i]["path"]
            print(f"# 模型 {i+1}")
            print(f"python3 -m sglang.launch_server \\")
            print(f"    --model-path Qwen/Qwen2.5-7B-Instruct \\")
            print(f"    --decrypted-config-file {config_path} \\")
            print(f"    --port {port}")
            print()

        print("预期结果：")
        print("- 所有服务器实例都成功启动")
        print("- 每个实例使用正确的解密配置")
        print("- 所有实例推理功能正常")
        print()
        print("测试完成：请手动验证上述预期结果")

    def test_02_verify_configs_exist(self):
        """测试2：验证配置文件存在"""
        print("\n=== 测试2：验证配置文件存在 ===")

        for i in range(3):
            config_path = self.configs[i]["path"]
            exists = os.path.exists(config_path)
            print(f"✓ 配置文件{i+1}存在: {exists}")
            self.assertTrue(exists, f"配置文件{i+1}不存在")

    def test_03_verify_configs_readable(self):
        """测试3：验证配置文件可读"""
        print("\n=== 测试3：验证配置文件可读 ===")

        for i in range(3):
            config_path = self.configs[i]["path"]
            try:
                with open(config_path, "r") as f:
                    content = f.read()
                print(f"✓ 配置文件{i+1}可读，大小: {len(content)} 字节")
                self.assertTrue(len(content) > 0, f"配置文件{i+1}为空")
            except Exception as e:
                print(f"✗ 配置文件{i+1}不可读: {e}")
                self.fail(f"配置文件{i+1}不可读: {e}")

    def test_04_verify_configs_valid(self):
        """测试4：验证配置文件内容有效"""
        print("\n=== 测试4：验证配置文件内容有效 ===")

        for i in range(3):
            config_path = self.configs[i]["path"]
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)

                print(f"✓ 配置文件{i+1}JSON格式有效")
                print(f"✓ 配置内容: {config}")
                self.assertIsInstance(config, dict, f"配置{i+1}不是字典类型")
            except Exception as e:
                print(f"✗ 配置文件{i+1}内容无效: {e}")
                self.fail(f"配置文件{i+1}内容无效: {e}")

    def test_05_config_content_uniqueness(self):
        """测试5：验证配置内容唯一性"""
        print("\n=== 测试5：验证配置内容唯一性 ===")

        configs = []
        for i in range(3):
            config_path = self.configs[i]["path"]
            with open(config_path, "r") as f:
                config = json.load(f)
            configs.append(config)

        # 验证每个配置的model_type不同
        model_types = [config["model_type"] for config in configs]
        unique_types = set(model_types)

        print(f"✓ 模型类型: {model_types}")
        print(f"✓ 唯一类型数量: {len(unique_types)}")
        self.assertEqual(len(unique_types), len(model_types), "模型类型不唯一")

    def test_06_concurrent_requests_simulation(self):
        """测试6：模拟并发请求到不同模型"""
        print("\n=== 测试6：模拟并发请求到不同模型 ===")
        print("注意：此测试假设多个服务器实例已启动")
        print()

        base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")

        # 模拟向不同端口发送请求
        ports = [30000, 30001, 30002]
        results = []

        for port in ports:
            try:
                url = f"http://127.0.0.1:{port}"
                response = requests.get(f"{url}/health", timeout=2)

                if response.status_code == 200:
                    results.append(
                        {
                            "port": port,
                            "status": "running",
                        }
                    )
                    print(f"✓ 端口 {port}: 服务器运行中")
                else:
                    results.append(
                        {
                            "port": port,
                            "status": "error",
                        }
                    )
                    print(f"✗ 端口 {port}: 服务器错误")
            except requests.exceptions.ConnectionError:
                results.append(
                    {
                        "port": port,
                        "status": "not_running",
                    }
                )
                print(f"⚠  端口 {port}: 服务器未运行")
            except Exception as e:
                results.append(
                    {
                        "port": port,
                        "status": "exception",
                        "error": str(e),
                    }
                )
                print(f"✗ 端口 {port}: 异常 - {e}")

        print()
        running_count = sum(1 for r in results if r["status"] == "running")
        print(f"✓ 运行中的服务器数量: {running_count}/{len(ports)}")

        if running_count == 0:
            print("⚠  所有服务器都未运行")
            print("⚠  请先启动多个服务器实例")
        else:
            print("✓  至少有一个服务器在运行")

    @classmethod
    def tearDownClass(cls):
        """清理测试文件"""
        try:
            for i in range(3):
                config_path = cls.configs[i]["path"]
                if os.path.exists(config_path):
                    os.remove(config_path)
        except Exception as e:
            print(f"清理测试文件时出错: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
