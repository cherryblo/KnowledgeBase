"""
用例7：视频多模态推理测试

测试目的：验证Multi-Modal特性在NPU环境下能处理视频+文本输入
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


class TestVideoMultimodal(unittest.TestCase):
    """视频多模态推理测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv(
            "MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        cls.timeout = 300

        # 创建测试视频（使用图像序列模拟）
        cls.test_video_path = cls._create_test_video()

    @staticmethod
    def _create_test_video():
        """创建一个简单的测试视频（使用图像序列）"""
        # 创建一系列图像来模拟视频
        frames_dir = Path("/tmp/test_video_frames")
        frames_dir.mkdir(exist_ok=True)

        colors = ["red", "green", "blue", "yellow", "purple"]
        for i, color in enumerate(colors):
            image = Image.new("RGB", (320, 240), color=color)
            # 添加一些文字
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(image)
            draw.text((10, 10), f"Frame {i}", fill="white")
            image.save(frames_dir / f"frame_{i:03d}.jpg")

        return str(frames_dir)

    def _encode_image(self, image_path):
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

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

    def test_03_single_frame_inference(self):
        """测试3：单帧推理（作为视频帧）"""
        print("\n=== 测试3：单帧推理 ===")

        frames_dir = Path(self.test_video_path)
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        if not frame_files:
            self.skipTest("没有找到测试帧")

        # 使用第一帧
        frame_path = frame_files[0]
        image_base64 = self._encode_image(str(frame_path))

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "What do you see in this frame?"},
                    ],
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
            print(f"✓ 单帧推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

        except Exception as e:
            self.fail(f"单帧推理失败: {e}")

    def test_04_multiple_frames_inference(self):
        """测试4：多帧推理（模拟视频）"""
        print("\n=== 测试4：多帧推理 ===")

        frames_dir = Path(self.test_video_path)
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        if len(frame_files) < 2:
            self.skipTest("测试帧数量不足")

        # 使用前3帧
        selected_frames = frame_files[:3]

        content = [{"type": "text", "text": "Describe the sequence of these frames."}]
        for frame_path in selected_frames:
            image_base64 = self._encode_image(str(frame_path))
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base6464}"},
                }
            )

        payload = {
            "model": self.model_path,
            "messages": [{"role": "user", "content": content}],
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
            print(f"✓ 多帧推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

        except Exception as e:
            self.fail(f"多帧推理失败: {e}")

    def test_05_frame_sequence_analysis(self):
        """测试5：帧序列分析"""
        print("\n=== 测试5：帧序列分析 ===")

        frames_dir = Path(self.test_video_path)
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        if len(frame_files) < 3:
            self.skipTest("测试帧数量不足")

        # 使用所有帧
        content = [{"type": "text", "text": "Analyze the color changes in these frames."}]
        for frame_path in frame_files:
            image_base64 = self._encode_image(str(frame_path))
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
            print(f"✓ 帧序列分析成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

            # 验证回答中包含颜色相关信息
            self.assertTrue(
                any(
                    color in content.lower()
                    for color in ["red", "green", "blue", "color", "frame"]
                ),
                f"回答中未检测到颜色或帧信息: {content}",
            )

        except Exception as e:
            self.fail(f"帧序列分析失败: {e}")

    def test_06_streaming_video_inference(self):
        """测试6：流式视频推理"""
        print("\n=== 测试6：流式视频推理 ===")

        frames_dir = Path(self.test_video_path)
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        if not frame_files:
            self.skipTest("没有找到测试帧")

        # 使用第一帧
        frame_path = frame_files[0]
        image_base64 = self._encode_image(str(frame_path))

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "Describe this frame in detail."},
                    ],
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

    @classmethod
    def tearDownClass(cls):
        """清理测试文件"""
        try:
            frames_dir = Path(cls.test_video_path)
            if frames_dir.exists():
                import shutil

                shutil.rmtree(frames_dir)
        except Exception as e:
            print(f"清理测试文件时出错: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
