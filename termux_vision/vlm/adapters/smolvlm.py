# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
class SmolVLMAdapter:
    adapter_name = "smolvlm"

    def format_prompt(self, user_prompt: str, task: str = "describe") -> str:
        return f"<|im_start|>user\n<image>\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    def get_preferred_resolution(self) -> int:
        return 384
