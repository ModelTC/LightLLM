# SPDX-License-Identifier: Apache-2.0

"""KDA over replay state: prefill keeps raw q/k with one SSM slot per request; decode appends history
that the replay kernel reads from each request's cursor."""

from lightllm.common.attention.backends.kda.triton import ReplayKDADecodeBackend, ReplayKDAPrefillBackend
from .kda import KDALinearAttBackend


class ReplayKDALinearAttBackend(KDALinearAttBackend):
    prefill_backend = ReplayKDAPrefillBackend
    decode_backend = ReplayKDADecodeBackend
    replay = True
