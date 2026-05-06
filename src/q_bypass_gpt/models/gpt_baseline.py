from __future__ import annotations

from q_bypass_gpt.models.qbypass_gpt import QBypassGPT


class BaselineGPT(QBypassGPT):
    # Same backbone, but config attention_type must be baseline.
    pass
