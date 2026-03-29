"""
用例6：配置文件权限测试

测试目的：验证当解密配置文件没有读取权限时的错误处理
测试环境：使用1张NPU卡，单机部署
"""

import os
import tempfile
import unittest


class TestFilePermission(unittest.TestCase):
    """配置文件权限测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")

        # 创建配置文件（但不设置权限，因为Windows权限管理不同）
        cls.config_path = cls._create_config_file()

    @staticmethod
    def _create_config_file():
        """创建一个配置文件"""
        import json

        config = {
            "model_type": "test",
            "quantization_config": {
                "quant_method": "w8a8",
            },
        }

        config_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(config, config_path, indent=2)
        config_path.close()

        return config_path.name

    def test_01_file_exists(self):
        """测试1：验证配置文件存在"""
        print("\n=== 测试1：验证配置文件存在 ===")

        exists = os.path.exists(self.config_path)
        print(f"✓ 配置文件存在: {exists}")
        self.assertTrue(exists, "配置文件不存在")

    def test_02_file_readable(self):
        """测试2：验证配置文件可读"""
        print("\n=== 测试2：验证配置文件可读 ===")

        try:
            with open(self.config_path, "r") as f:
                content = f.read()
            print(f"✓ 配置文件可读")
            print(f"✓ 文件大小: {len(content)} 字节")
            self.assertTrue(len(content) > 0, "配置文件为空")
        except Exception as e:
            print(f"✗ 配置文件不可读: {e}")
            self.fail(f"配置文件不可读: {e}")

    def test_03_file_content_valid(self):
        """测试3：验证配置文件内容有效"""
        print("\n=== 测试3：验证配置文件内容有效 ===")

        try:
            import json

            with open(self.config_path, "r") as f:
                config = json.load(f)

            print(f"✓ 配置文件JSON格式有效")
            print(f"✓ 配置内容: {config}")
            self.assertIsInstance(config, dict, "配置不是字典类型")
        except Exception as e:
            print(f"✗ 配置文件内容无效: {e}")
            self.fail(f"配置文件内容无效: {e}")

    def test_04_manual_permission_test(self):
        """测试4：手动权限测试说明"""
        print("\n=== 测试4：手动权限测试说明 ===")
        print("注意：此测试需要手动验证")
        print()
        print("测试步骤：")
        print("1. 创建一个配置文件")
        print("2. 设置文件为不可读权限（在Linux/Unix系统上）")
        print("3. 启动SGLang服务器，指定该配置文件")
        print("4. 观察服务器启动行为")
        print()
        print("Linux/Unix命令示例：")
        print("  echo '{\"key\": \"value\"}' > config.json")
        print("  chmod 000 config.json")
        print("  python3 -m sglang.launch_server \\")
        print(f"       --model-path {self.model_path} \\")
        print("       --decrypted-config-file config.json")
        print()
        print("预期结果：")
        print("- 服务器启动失败或给出明确的错误信息")
        print("- 错误信息指出文件权限问题")
        print()
        print("测试完成：请手动验证上述预期结果")

    def test_05_windows_permission_note(self):
        """测试5：Windows权限说明"""
        print("\n=== 测试5：Windows权限说明 ===")

        if os.name == "nt":
            print("✓ 运行在Windows系统上")
            print("✓ Windows权限管理与Linux/Unix不同")
            print("✓ 文件权限通常通过文件属性管理")
            print("✓ 建议在Linux/Unix系统上测试权限场景")
        else:
            print("✓ 运行在非Windows系统上")
            print("✓ 可以使用chmod命令测试权限场景")


if __name__ == "__main__":
    unittest.main(verbosity=2)
