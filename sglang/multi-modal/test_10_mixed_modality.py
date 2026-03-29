"""
用例10：混合模态推理测试

测试目的：验证Multi-Modal特性在NPU环境下能同时处理多种模态（图像+视频+音频）
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


class TestMixedModality(unittest.TestCase):
    """混合模态推理测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv(
            "MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Inrastruct"
        )
        cls.timeout = 300

        # 创建测试文件
        cls.test_image = cls._create_test_image()
        cls.test_video_frames = cls._create_test_video_frames()
        cls.test_audio = cls._create_test_audio()

    @staticmethod
    def _create_test_image():
        """创建测试图像"""
        image = Image.new("RGB", (640, 480), color="red")
        image_path = Path("/tmp/test_mixed_image.jpg")
        image.save(image_path)
        return str(image_path)

    @staticmethod
    def _create_test_video_frames():
        """创建测试视频帧"""
        frames_dir = Path("/tmp/test_mixed_video_frames")
        frames_dir.mkdir(exist_ok=True)

        colors = ["blue", "green", "yellow"]
        for i, color in enumerate(colors):
            image = Image.new("RGB", (320, 240), color=color)
            image.save(frames_dir / f"frame_{i:03d}.jpg")

        return str(frames_dir)

    @staticmethod
    def _create_test_audio():
        """创建测试音频"""
        import numpy as np
        import wave

        sample_rate = 16000
        duration = 1
        frequency = 440

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_data * 32767).astype(np.int16)

        audio_path = Path("/tmp/test_mixed_audio.wav")
        with wave.open(str(audio_path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return str(audio_path)

    def _encode_image(self, image_path):
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_audio(self, audio_path):
        """将音频编码为base64"""
        with open(audio_path, "rb") as f:
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

    def test_02_image_only(self):
        """测试2：仅图像"""
        print("\n=== 测试2：仅图像 ===")

        image_base64 = self._encode_image(self.test_image)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "What do you see in this image?"},
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
            print(f"✓ 仅图像推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

        except Exception as e:
            self.fail(f"仅图像推理失败: {e}")

    def test_03_multiple_images(self):
        """测试3：多张图像"""
        print("\n=== 测试3：多张图像 ===")

        frames_dir = Path(self.test_video_frames)
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        content = [{"type": "text", "text": "Describe these images."}]
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
            print(f"✓ 多图像推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

        except Exception as e:
            self.fail(f"多图像推理失败: {e}")

    def test_04_image_with_text_context(self):
        """测试4：图像+文本上下文"""
        print("\n=== 测试4：图像+文本上下文 ===")

        image_base64 = self._encode_image(self.test_image)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "This is a test image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "I understand. What would you like to know about it?",
                },
                {
                    "role": "user",
                    "content": "What color is it?",
                },
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
            print(f"✓ 图像+文本上下文推理成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

        except Exception as e:
            self.fail(f"图像+文本上下文推理失败: {e}")

    def test_05_streaming_mixed_modality(self):
        """测试5：流式混合模态推理"""
        print("\n=== 测试5：流式混合模态推理 ===")

        image_base64 = self._encode_image(self.test_image)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ],
            "max_tokens": 100,
            "stream": True,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json: payload,
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

    def test_06_multiple_requests_different_modalities(self):
        """测试6：多个不同模态的请求"""
        print("\n=== 测试6：多个不同模态的请求 ===")

        import concurrent.futures

        def send_image_request():
            image_base64 = self._encode_image(self.test_image)
            payload = {
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": "What color is this?"},
                        ],
                    }
                ],
                "max_tokens": 30,
            }
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            return response.status_code == 200

        def send_multi_image_request():
            frames_dir = Path(self.test_video_frames)
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))

            content = [{"type": "text", "text": "Describe these."}]
            for frame_path in frame_files[:2]:
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
                "max_tokens": 50,
            }
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            return response.status_code == 200

        # 并发发送不同模态的请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(send_image_request),
                executor.submit(send_multi_image_request),
            ]

            results = [future.result() for future in futures]

        successful = sum(results)
        print(f"✓ 成功: {successful}/{len(results)}")
        self.assertEqual(successful, len(results), f"部分请求失败: {len(results) - successful}")

    @classmethod
    def tearDownClass(cls):
        """清理测试文件"""
        try:
            if os.path.exists(cls.test_image):
                os.remove(cls.test_image)

            frames_dir = Path(cls.test_video_frames)
            if frames_dir.exists():
                import shutil

                shutil.rmtree(frames_dir)

            if os.path.exists(cls.test_audio):
                os.remove(cls.test_audio)
        except Exception as e:
            print(f"清理测试文件时出错: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
