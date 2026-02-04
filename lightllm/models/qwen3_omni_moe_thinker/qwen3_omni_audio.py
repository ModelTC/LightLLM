import os
import json
import math
import torch
import librosa
import numpy as np
from io import BytesIO
from torch import Tensor, nn
from safetensors import safe_open
from torch.nn import functional as F
from typing import Callable, Optional, Union, List
from rpyc.utils.classic import obtain

from transformers.activations import ACT2FN

from lightllm.server.multimodal_params import AudioItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.models.vit.triton_kernel.flashattention_nopad import flash_attention_fwd
from lightllm.models.qwen3_omni_moe_thinker.audio_process import WhisperFeatureExtractor


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        encoder_attention_heads,
        attention_dropout,
        dropout,
        activation_function,
        activation_dropout,
        encoder_ffn_dim,
    ):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = Qwen3OmniMoeAudioAttention(d_model, encoder_attention_heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = dropout
        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


class Qwen3OmniMoeAudioAttention(nn.Module):
    def __init__(self, d_model, encoder_attention_heads, attention_dropout):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = encoder_attention_heads
        self.dropout = attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, _ = hidden_states.size()

        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        attn_output = g_cache_manager.alloc_tensor(q.shape, q.dtype, device=q.device)

        flash_attention_fwd(q, k, v, attn_output, cu_seqlens, max_seqlen)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.positional_embedding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen3OmniMoeAudioEncoder(nn.Module):
    def __init__(
        self,
        kvargs,
        dropout=0,
        d_model=1280,
        num_mel_bins=128,
        max_source_positions=1500,
        scale_embedding=False,
        n_window=50,
        encoder_layers=32,
        downsample_hidden_size=480,
        activation_function="gelu",
        output_dim=2048,
        n_window_infer=800,
        conv_chunksize=500,
        encoder_attention_heads=20,
        attention_dropout=0,
        activation_dropout=0,
        encoder_ffn_dim=5120,
    ):
        super().__init__()
        self.data_type = kvargs.get("data_type", "bfloat16")
        self.dropout = dropout
        self.embed_dim = d_model
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if scale_embedding else 1.0
        self.n_window = n_window
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, self.embed_dim)
        self.layers = nn.ModuleList(
            [
                Qwen3OmniMoeAudioEncoderLayer(
                    d_model,
                    encoder_attention_heads,
                    attention_dropout,
                    dropout,
                    activation_function,
                    activation_dropout,
                    encoder_ffn_dim,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.ln_post = nn.LayerNorm(d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(downsample_hidden_size, downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(downsample_hidden_size, downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            downsample_hidden_size * ((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(d_model, d_model)
        self.act = ACT2FN[activation_function]
        self.proj2 = nn.Linear(d_model, output_dim)
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self._init_datatype()

    def _init_datatype(self):
        if isinstance(self.data_type, torch.dtype):
            return
        if self.data_type in ["fp16", "float16"]:
            self.data_type = torch.float16
        elif self.data_type in ["bf16", "bfloat16"]:
            self.data_type = torch.bfloat16
        elif self.data_type in ["fp32", "float32"]:
            self.data_type = torch.float32
        else:
            raise ValueError(f"Unsupport datatype {self.data_type}!")
        return

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def load_model(self, weight_dir, config):
        processor_config_path = os.path.join(weight_dir, "preprocessor_config.json")
        with open(processor_config_path, "r") as f:
            processor_config_dict = json.load(f)
        self.processor = WhisperFeatureExtractor(**processor_config_dict)

        bin_weight_files = [file_ for file_ in os.listdir(weight_dir) if file_.endswith(".bin")]
        if bin_weight_files:
            weight_dict = {}
            for file_ in bin_weight_files:
                f = torch.load(os.path.join(weight_dir, file_), "cpu")
                for k, v in f.items():
                    if "thinker.audio_tower" in k:
                        weight_dict[k[len("thinker.audio_tower.") :]] = v
        else:
            hf_weight_files = [file_ for file_ in os.listdir(weight_dir) if file_.endswith(".safetensors")]
            weight_dict = {}
            for file_ in hf_weight_files:
                f = safe_open(os.path.join(weight_dir, file_), "pt", "cpu")
                for k in f.keys():
                    if "thinker.audio_tower" in k:
                        weight_dict[k[len("thinker.audio_tower.") :]] = f.get_tensor(k)

        self.load_state_dict(weight_dict)

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
            .to(padded_embed.device)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    def encode(self, audio_items: List[AudioItem], cpu_embed_cache_client: CpuEmbedCacheClient):
        # 每个元素是一个chunk
        batch_audios = []
        batch_audio_lens = []
        uuids = []
        items: List[AudioItem] = []
        # 记录每个chunk属于哪个audio_items下标
        chunk_owner_index = []
        for i, item in enumerate(audio_items):
            if isinstance(item, AudioItem):
                uuids.append(item.uuid)
                items.append(item)
                audio_data = read_shm(get_shm_name_data(item.uuid))
                audio = BytesIO(audio_data)
                audio, _ = librosa.load(audio, sr=16000)
            else:
                raise ValueError(f"cannot read audio which type is {type(item)}!")

            # padding to min audio len
            MIN_AUDIO_LEN = 480

            if audio.shape[0] < MIN_AUDIO_LEN:
                audio = np.pad(audio, (0, MIN_AUDIO_LEN - len(audio)), mode="constant", constant_values=0.0)

            if audio.shape[0] > self.max_length:
                start = 0
                while start < audio.shape[0]:
                    end = min(start + self.max_length, audio.shape[0])
                    chunk = audio[start:end]

                    if chunk.shape[0] < MIN_AUDIO_LEN:
                        chunk = np.pad(chunk, (0, MIN_AUDIO_LEN - chunk.shape[0]), mode="constant", constant_values=0.0)
                    batch_audios.append(chunk)
                    batch_audio_lens.append(min(chunk.shape[0], self.max_length))
                    chunk_owner_index.append(i)

                    start = end
            else:
                batch_audio_lens.append(min(audio.shape[0], self.max_length))
                batch_audios.append(audio)
                chunk_owner_index.append(i)

        batch_audio_lens = np.array(batch_audio_lens, dtype=np.int32)

        audios, audio_lens_after_cnn = self.processor._preprocess(
            batch_audios, sampling_rate=16000, return_tensors="pt"
        )
        audios = self.forward(audios, audio_lens_after_cnn)
        audio_lens_after_cnn = np.array(audio_lens_after_cnn, dtype=np.int32)
        audio_token_num = (audio_lens_after_cnn - 2) // 2 + 1

        num_audios = len(audio_items)
        per_audio_embeds = [[] for _ in range(num_audios)]

        for chunk_idx, owner in enumerate(chunk_owner_index):
            token_len = int(audio_token_num[chunk_idx])
            if token_len <= 0:
                continue
            per_audio_embeds[owner].append(audios[chunk_idx][:token_len])

        ready_audio = obtain(self.cache_client.root.get_items_embed(uuids))
        ids_to_set = []
        for i, ready in enumerate(ready_audio):
            if ready:
                continue

            uid = uuids[i]
            item = items[i]

            # 拼接该 audio 的所有 chunk embedding
            cur_embed = torch.cat(per_audio_embeds[i], dim=0)
            cpu_embed_cache_client.copy_to_cache(
                embed_tensor=cur_embed, start_index_in_cache=item.start_index_in_embed_cache
            )
            ids_to_set.append(uid)

        if ids_to_set:
            self.cache_client.root.set_items_embed(ids=ids_to_set)
            torch.cuda.current_stream().synchronize()
