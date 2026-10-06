class Qwen2VLAdapter:
    adapter_name = "qwen2vl"

    def format_prompt(self, user_prompt: str, task: str = "describe") -> str:
        return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    def get_preferred_resolution(self) -> int:
        return 384
