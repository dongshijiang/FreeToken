"""qwen4_exp weight loading against synthetic checkpoints shaped like the released ones.

The tensors are tiny but the key names, dtypes and the fusion geometry that matters
(hc_lowrank=320 + hc_count=4 -> a 12-row zero pad; 128-row block scales) are the real ones.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.distributed.info import DistributedInfo
from freetoken.kernel.aot_models import SUPPORTED_MODELS, expert_bank_row_bytes
from freetoken.models.qwen4_exp.weight import (
    _ZERO_CENTERED_NORM_SUFFIXES,
    _DenseFuser,
    _TpSlicer,
    iter_weights,
    load_ple_table,
)
from freetoken.models.register import get_model_spec
from freetoken.moe.host_banks import HostBank, read_range_into

from .common import (
    LM,
    RADIXARK_NVFP4,
    hf_config,
    install_quant_config,
    meta_state_dict,
    mixed_precision_quant,
    parsed_config,
)

H = 128  # hidden_size; every block-fp8 projection needs in/out multiples of 128
HC = 4  # hc_count
LR = 320  # hc_lowrank; kept real so the merged HC pad is the real (-(320+4)) % 16 = 12
HCH = HC * H  # hyper-connection stream width
KH, VH, HD = 4, 4, 32  # GDN key / value heads, head dim: qkv rows 384, z rows 128; each q / v
# head group is 128 rows so every head-aligned block-fp8 segment stays a 128 multiple
QH, KVH, AHD = 4, 2, 64  # QSA q / kv heads, head dim: q rows 512, k / v rows 128
IHD = 64  # indexer head dim
BLOCK = 128
E, I = 3, 6  # routed experts, moe_intermediate_size
NGRAM_DIM, NGRAM_ROWS, NGRAM_SHARDS = 4, 7, 4


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _hc_weights(prefix: str, inject: bool) -> dict[str, torch.Tensor]:
    w = {
        f"{prefix}.hc_norm.weight": _bf16(HCH),
        f"{prefix}.input_mix_weight_down.weight": _bf16(LR, HCH),
        f"{prefix}.input_mix_weight_up.weight": _bf16(HCH, LR),
    }
    if inject:
        w[f"{prefix}.block_inject_weight.weight"] = _bf16(HC, HCH)
    return w


def _fp8_scale(weight: torch.Tensor) -> torch.Tensor:
    return torch.rand(weight.shape[0] // BLOCK, weight.shape[1] // BLOCK) + 0.5


def _raw_checkpoint(dense_fp8: bool = False) -> dict[str, torch.Tensor]:
    """Layer 0 = GDN + PLE, layer 1 = QSA; plus the mtp / visual / routed-expert noise.

    ``dense_fp8`` stores the attention and GDN qkv|z / out projections as 128x128 block-fp8 (e4m3 ``.weight`` + fp32 ``.weight_scale_inv``) like the community NVFP4-FP8 requants.
    """
    lm = "model.language_model"
    raw: dict[str, torch.Tensor] = {
        f"{lm}.embed_tokens.weight": _bf16(11, H),
        "lm_head.weight": _bf16(11, H),
    }
    raw.update(_hc_weights(f"{lm}.hyper_connection_mixer", inject=False))
    for layer in (0, 1):
        raw.update(_hc_weights(f"{lm}.layers.{layer}.attn_hyper_connection", inject=True))
        raw.update(_hc_weights(f"{lm}.layers.{layer}.mlp_hyper_connection", inject=True))
        raw.update({
            f"{lm}.layers.{layer}.mlp.gate.weight": _bf16(E, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.gate_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.up_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.down_proj.weight": _bf16(H, I),
            f"{lm}.layers.{layer}.mlp.shared_expert_gate.weight": _bf16(1, H),
        })
        for expert in range(E):
            base = f"{lm}.layers.{layer}.mlp.experts.{expert}"
            for proj, out, inn in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
                raw[f"{base}.{proj}.weight"] = torch.randint(
                    0, 256, (out, inn // 2), dtype=torch.uint8
                )
                raw[f"{base}.{proj}.weight_scale"] = torch.ones(
                    out, inn // 16 or 1, dtype=torch.float8_e4m3fn
                )
                raw[f"{base}.{proj}.weight_scale_2"] = torch.tensor(0.5)
                raw[f"{base}.{proj}.input_scale"] = torch.tensor(0.25)
    gdn = f"{lm}.layers.0.linear_attn"
    raw.update({
        f"{gdn}.in_proj_qkv.weight": _bf16(2 * KH * HD + VH * HD, H),
        f"{gdn}.in_proj_z.weight": _bf16(VH * HD, H),
        f"{gdn}.in_proj_b.weight": _bf16(VH, H),
        f"{gdn}.in_proj_a.weight": _bf16(VH, H),
        f"{gdn}.conv1d.weight": _bf16(2 * KH * HD + VH * HD, 1, 4),
        f"{gdn}.A_log": _bf16(VH),
        f"{gdn}.dt_bias": _bf16(VH),
        f"{gdn}.norm.weight": _bf16(HD),
        f"{gdn}.out_proj.weight": _bf16(H, VH * HD),
    })
    ple = f"{lm}.layers.0.ple"
    raw.update({
        f"{ple}.key_proj.weight": _bf16(HCH, H),
        f"{ple}.value_proj.weight": _bf16(H, H),
        f"{ple}.norm_key.weight": _bf16(HCH),
        f"{ple}.norm_query.weight": _bf16(HCH),
        f"{ple}.norm_conv.weight": _bf16(HCH),
        f"{ple}.conv1d.weight": _bf16(HCH, 1, 4),
        f"{ple}.ple_embedding.layer_multipliers": torch.randint(1, 1 << 40, (3,)),
        f"{ple}.ple_embedding.ngram_heads_offsets": torch.arange(4),
        f"{ple}.ple_embedding.ngram_heads_vocab_sizes": torch.full((4,), 5),
    })
    attn = f"{lm}.layers.1.self_attn"
    raw.update({
        f"{attn}.q_proj.weight": _bf16(2 * QH * AHD, H),
        f"{attn}.k_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.v_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.o_proj.weight": _bf16(H, QH * AHD),
        f"{attn}.q_norm.weight": _bf16(AHD),
        f"{attn}.k_norm.weight": _bf16(AHD),
        f"{attn}.indexer.index_qk_proj.weight": _bf16(5 * IHD, H),
        f"{attn}.indexer.q_layernorm.weight": _bf16(IHD),
        f"{attn}.indexer.k_layernorm.weight": _bf16(IHD),
    })
    raw.update({
        "mtp.hyper_connection_mixer.hc_norm.weight": _bf16(HCH),
        "mtp.layers.0.self_attn.q_proj.weight": _bf16(2 * QH * AHD, H),
        "mtp.layers.0.mlp.experts.gate_up_proj": _bf16(E, 2 * I, H),
        "mtp.layers.0.mlp.experts.down_proj": _bf16(E, H, I),
        "model.visual.blocks.0.attn.qkv.weight": _bf16(3 * H, H),
        "model.visual.merger.norm.weight": _bf16(H),
    })
    if dense_fp8:
        for module in (f"{gdn}.in_proj_qkv", f"{gdn}.in_proj_z", f"{gdn}.out_proj",
                       *(f"{attn}.{p}_proj" for p in "qkvo")):
            weight = raw[f"{module}.weight"]
            raw[f"{module}.weight"] = weight.to(torch.float8_e4m3fn)
            raw[f"{module}.weight_scale_inv"] = _fp8_scale(weight)
    return raw


FP8_DENSE_QUANT = mixed_precision_quant(gdn_layers=(0,), attn_layers=(1,), moe_layers=(0, 1))


def _config_json(quantization_config) -> dict:
    cfg = hf_config(
        num_layers=2, head_dim=AHD, num_q=QH, num_kv=KVH, index_head_dim=IHD, index_heads=2,
        budget=16, hidden=H, max_position=4096, rope_theta=10000.0,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=KH, linear_num_value_heads=VH,
        linear_key_head_dim=HD, linear_value_head_dim=HD,
        hc_lowrank=LR, ple_layer_ids=[1],
        num_experts=E, moe_intermediate_size=I, shared_expert_intermediate_size=I,
    )
    return {**vars(cfg), "text_config": vars(cfg.text_config), "quantization_config": quantization_config}


def _ngram_table() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shards = {
        f"{prefix}.shard_{i}.weight": (
            torch.arange(i * NGRAM_ROWS * NGRAM_DIM, (i + 1) * NGRAM_ROWS * NGRAM_DIM)
            .remainder(200).to(torch.uint8).view(NGRAM_ROWS, NGRAM_DIM).view(torch.float8_e4m3fn)
        )
        for i in range(NGRAM_SHARDS)
    }
    scale = torch.tensor([0.125], dtype=torch.bfloat16)
    shards[f"{prefix}.weight_scale"] = scale
    return shards, scale


def _write_checkpoint(folder, raw: dict[str, torch.Tensor], quantization_config) -> tuple[str, dict[str, torch.Tensor]]:
    table, _scale = _ngram_table()
    # Spread the dense tensors over two shards so the fusion buffer has to survive a file
    # boundary, and put the n-gram table in its own shards like the real checkpoint does.
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    shard_names = sorted(table)
    save_file({n: table[n] for n in shard_names[:2]}, str(folder / "model-plefp8-00000.safetensors"))
    save_file({n: table[n] for n in shard_names[2:]}, str(folder / "model-plefp8-00001.safetensors"))
    (folder / "config.json").write_text(json.dumps(_config_json(quantization_config)))
    return str(folder), {**raw, **table}


def _load(folder: str, *, vision: bool = True) -> dict[str, torch.Tensor]:
    install_quant_config(folder)
    return {
        name: tensor.clone()
        for name, tensor in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True, include_vision=vision
        )
    }


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    return _write_checkpoint(tmp_path_factory.mktemp("qwen4_exp_ckpt"), _raw_checkpoint(), None)


@pytest.fixture(scope="module")
def loaded(checkpoint) -> dict[str, torch.Tensor]:
    return _load(checkpoint[0])


@pytest.fixture(scope="module")
def checkpoint_fp8(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    torch.manual_seed(1)
    return _write_checkpoint(
        tmp_path_factory.mktemp("qwen4_exp_fp8_ckpt"), _raw_checkpoint(dense_fp8=True), FP8_DENSE_QUANT
    )


@pytest.fixture(scope="module")
def loaded_fp8(checkpoint_fp8) -> dict[str, torch.Tensor]:
    return _load(checkpoint_fp8[0])


def test_tower_keys_come_out_under_the_prefix_load_weight_filters(loaded):
    assert {n for n in loaded if "visual" in n} == {"visual.blocks.0.attn.qkv.weight", "visual.merger.norm.weight"}


def test_mtp_experts_and_table_never_loaded(loaded):
    for name in loaded:
        assert not name.startswith("mtp.")
        assert ".mlp.experts." not in name
        assert "ngram_embedding" not in name
        assert not name.endswith((".weight_scale", ".weight_scale_2", ".input_scale", ".weight_scale_inv"))


def test_hc_merge_is_down_then_inject_then_zero_pad(loaded, checkpoint):
    _folder, raw = checkpoint
    key = "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    merged = loaded[key]
    assert merged.shape == (LR + HC + 12, HCH)  # pad = (-(320 + 4)) % 16
    down = raw["model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"]
    inject = raw["model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight"]
    assert torch.equal(merged[:LR], down)
    assert torch.equal(merged[LR:LR + HC], inject)
    assert torch.equal(merged[LR + HC:], torch.zeros(12, HCH, dtype=merged.dtype))


def test_top_level_mixer_keeps_the_unmerged_down(loaded, checkpoint):
    _folder, raw = checkpoint
    got = loaded["model.hyper_connection_mixer.input_mix_weight_down.weight"]
    assert got.shape == (LR, HCH)
    assert torch.equal(
        got, raw["model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"]
    )
    assert torch.equal(
        loaded["model.hyper_connection_mixer.input_mix_weight_up.weight"],
        raw["model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"],
    )


def test_qkv_fusion_slices_back_to_q_k_v(loaded, checkpoint):
    _folder, raw = checkpoint
    attn = "model.language_model.layers.1.self_attn"
    parts = [raw[f"{attn}.{p}_proj.weight"] for p in ("q", "k", "v")]
    fused = loaded["model.layers.1.self_attn.qkv_proj.weight"]
    assert fused.shape == (2 * QH * AHD + 2 * KVH * AHD, H)  # q carries the output gate
    for part, back in zip(parts, torch.split(fused, [p.shape[0] for p in parts], dim=0)):
        assert torch.equal(part, back)


def test_gdn_in_proj_slices_round_trip(loaded, checkpoint):
    _folder, raw = checkpoint
    gdn = "model.language_model.layers.0.linear_attn"
    parts = [raw[f"{gdn}.in_proj_{p}.weight"] for p in ("qkv", "z", "b", "a")]
    fused = loaded["model.layers.0.linear_attn.in_proj.weight"]
    assert fused.shape == (sum(p.shape[0] for p in parts), H)
    splits = torch.split(fused, [p.shape[0] for p in parts], dim=0)
    for part, back in zip(parts, splits):
        assert torch.equal(part, back)


def test_gdn_scalars_upcast_to_the_fp32_params(loaded, checkpoint):
    """The checkpoint stores A_log/dt_bias in the model dtype; the layer keeps fp32 params."""
    _folder, raw = checkpoint
    gdn = f"{LM}.layers.0.linear_attn"
    for leaf in ("A_log", "dt_bias"):
        assert raw[f"{gdn}.{leaf}"].dtype == torch.bfloat16
        got = loaded[f"model.layers.0.linear_attn.{leaf}"]
        assert got.dtype == torch.float32
        assert torch.equal(got, raw[f"{gdn}.{leaf}"].float())


def test_shared_expert_gate_up_merge(loaded, checkpoint):
    _folder, raw = checkpoint
    base = "model.language_model.layers.1.mlp.shared_expert"
    merged = loaded["model.layers.1.mlp.shared_expert.gate_up_proj.weight"]
    assert torch.equal(merged[:I], raw[f"{base}.gate_proj.weight"])
    assert torch.equal(merged[I:], raw[f"{base}.up_proj.weight"])


ZERO_CENTERED = (
    "model.layers.0.attn_hyper_connection.hc_norm.weight",
    "model.layers.0.mlp_hyper_connection.hc_norm.weight",
    "model.hyper_connection_mixer.hc_norm.weight",
    "model.layers.0.ple.norm_key.weight",
    "model.layers.0.ple.norm_query.weight",
    "model.layers.0.ple.norm_conv.weight",
    "model.layers.1.self_attn.q_norm.weight",
    "model.layers.1.self_attn.k_norm.weight",
    "model.layers.1.self_attn.indexer.q_layernorm.weight",
    "model.layers.1.self_attn.indexer.k_layernorm.weight",
)


def test_zero_centered_norms_are_loaded_raw(loaded, checkpoint):
    """(1+w) is applied at runtime in fp32, so the loader must not fold it into the bf16 weight."""
    _folder, raw = checkpoint
    for name in ZERO_CENTERED:
        raw_name = name.replace("model.", "model.language_model.", 1)
        assert torch.equal(loaded[name], raw[raw_name]), name


def test_the_zero_centered_suffix_list_covers_every_such_norm():
    assert {n for n in ZERO_CENTERED if n.endswith(_ZERO_CENTERED_NORM_SUFFIXES)} == set(ZERO_CENTERED)
    assert not "model.layers.0.linear_attn.norm.weight".endswith(_ZERO_CENTERED_NORM_SUFFIXES)


def test_gdn_gated_norm_passes_through(loaded, checkpoint):
    _folder, raw = checkpoint
    assert torch.equal(
        loaded["model.layers.0.linear_attn.norm.weight"],
        raw["model.language_model.layers.0.linear_attn.norm.weight"],
    )


def test_hash_constants_stay_int64(loaded):
    for leaf in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
        assert loaded[f"model.layers.0.ple.ple_embedding.{leaf}"].dtype is torch.int64


def test_load_ple_table_concatenates_shards_in_index_order(checkpoint):
    folder, raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    table = load_ple_table(folder, args, pin=False)
    assert table.tensor.shape == (NGRAM_SHARDS * NGRAM_ROWS, NGRAM_DIM)
    assert table.tensor.dtype is torch.float8_e4m3fn
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    for shard in range(NGRAM_SHARDS):
        rows = table.tensor[shard * NGRAM_ROWS: (shard + 1) * NGRAM_ROWS]
        assert torch.equal(rows.view(torch.uint8),
                           raw[f"{prefix}.shard_{shard}.weight"].view(torch.uint8))
    assert table.weight_scale.dtype is torch.bfloat16
    assert float(table.weight_scale) == 0.125


def test_load_ple_table_rejects_a_shard_count_mismatch(checkpoint):
    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS + 1, ngram_head_dim=NGRAM_DIM)
    with pytest.raises(ValueError, match="shards 0"):
        load_ple_table(folder, args, pin=False)


# ======================================================================================
# read_range_into: the O_DIRECT byte-range read the PLE table load is built on
# ======================================================================================


@pytest.fixture(scope="module")
def blob(tmp_path_factory) -> tuple[str, bytes]:
    data = random.Random(7).randbytes(5_000_003)
    path = tmp_path_factory.mktemp("blob") / "data.bin"
    path.write_bytes(data)
    return str(path), data


@pytest.mark.parametrize("file_offset, nbytes, dest_offset", [
    (1, 4095, 0),                 # sub-block, unaligned source
    (2239, 1_000_000, 0),         # the real checkpoint's header-end phase
    (4095, 4097, 1),              # straddles two block boundaries
    (4_999_000, 1003, 123_456),   # runs to EOF
])
def test_read_range_into_matches_the_file(blob, file_offset, nbytes, dest_offset):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    got = read_range_into(view, path, file_offset=file_offset, nbytes=nbytes,
                          dest_offset=dest_offset, chunk=1 << 20)
    assert got == nbytes
    assert bytes(view[dest_offset:dest_offset + nbytes]) == data[file_offset:file_offset + nbytes]


def test_read_range_into_is_chunk_and_thread_safe(blob):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    read_range_into(view, path, file_offset=2239, nbytes=4_000_000, dest_offset=1024,
                    workers=8, chunk=64 << 10)
    assert bytes(view[1024:1024 + 4_000_000]) == data[2239:2239 + 4_000_000]


def test_read_range_into_rejects_a_short_destination(blob):
    path, _data = blob
    bank = HostBank((1024,), torch.uint8)
    with pytest.raises(ValueError, match="destination holds"):
        read_range_into(bank.memoryview(), path, file_offset=0, nbytes=1 << 20)


# ======================================================================================
# AOT shape table
# ======================================================================================


def test_aot_entry_carries_the_checkpoint_geometry():
    entry = next(m for m in SUPPORTED_MODELS
                 if m.architecture == "Qwen4ExpForConditionalGeneration")
    assert (entry.hidden_size, entry.moe_intermediate_size, entry.top_k) == (2560, 640, 10)
    assert entry.kv_groups == ((2, 256),)
    rows = expert_bank_row_bytes("nvfp4", entry.hidden_size, entry.moe_intermediate_size)
    assert set(rows) == {"gate_up_packed", "gate_up_scale", "gate_up_global",
                         "down_packed", "down_scale", "down_global"}
    for name, nbytes in rows.items():
        assert nbytes % 16 == 0, name  # fused multi-bank copy only engages on 16B multiples


def test_every_registry_architecture_is_claimed_by_an_aot_entry():
    from freetoken.models.register import _MODEL_REGISTRY

    claimed = {m.architecture for m in SUPPORTED_MODELS}
    claimed |= {a for m in SUPPORTED_MODELS for a in m.arch_aliases}
    assert "Qwen4ExpForConditionalGeneration" in claimed
    assert set(_MODEL_REGISTRY) - claimed == set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_fusion_pad_rides_the_tensor_device():
    """safetensors loads straight to cuda; a cpu-allocated pad row would break torch.cat."""
    fuser = _DenseFuser(None, get_model_spec("Qwen4ExpForConditionalGeneration").packed_modules_mapping)
    down = torch.randn(320, 64, device="cuda", dtype=torch.bfloat16)
    inject = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    assert fuser.fuse("model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", down) == []
    [(key, fused)] = fuser.fuse("model.layers.0.attn_hyper_connection.block_inject_weight.weight", inject)
    assert key == "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    assert fused.device.type == "cuda" and fused.shape[0] == 336
    assert torch.equal(fused[324:], torch.zeros(12, 64, device="cuda", dtype=torch.bfloat16))


# ======================================================================================
# the reader against the model the engine builds, for each released quant layout
# ======================================================================================


@pytest.fixture(scope="module")
def checkpoint_nvfp4(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    """bf16 dense tensors under a real ModelOptConfig whose ignore list covers them (RadixArk)."""
    torch.manual_seed(2)
    return _write_checkpoint(tmp_path_factory.mktemp("qwen4_exp_nvfp4_ckpt"), _raw_checkpoint(), RADIXARK_NVFP4)


FP8_MODULES = (
    "model.layers.1.self_attn.qkv_proj", "model.layers.1.self_attn.o_proj",
    "model.layers.0.linear_attn.in_proj_qkvz", "model.layers.0.linear_attn.out_proj",
)


@pytest.mark.parametrize("fixture", ["checkpoint", "checkpoint_nvfp4", "checkpoint_fp8"])
def test_emitted_keys_are_the_model_state_dict(fixture, request):
    """The reader fills exactly the buffers the engine builds from the same config, block-fp8 ones with the stored dtypes."""
    folder, _raw = request.getfixturevalue(fixture)
    loaded, state = _load(folder, vision=False), meta_state_dict(folder)
    assert set(loaded) == set(state)
    if fixture != "checkpoint_fp8":
        assert loaded["model.layers.0.linear_attn.in_proj.weight"].dtype is torch.bfloat16
        return
    for module in FP8_MODULES:
        for kind in (".weight", ".weight_scale_inv"):
            assert loaded[module + kind].shape == state[module + kind].shape, module + kind
        assert loaded[module + ".weight"].dtype is state[module + ".weight"].dtype is torch.float8_e4m3fn
        assert loaded[module + ".weight_scale_inv"].dtype is torch.float32  # the engine casts it to the bf16 buffer at load


def _assert_fused_per_kind(loaded, raw, fused: str, parts: list[str]) -> None:
    for kind in (".weight", ".weight_scale_inv"):
        sources = [raw[f"{p}{kind}"].view(torch.uint8) for p in parts]
        merged = loaded[fused + kind]
        assert merged.dtype is raw[f"{parts[0]}{kind}"].dtype
        for source, back in zip(sources, torch.split(merged.view(torch.uint8), [s.shape[0] for s in sources], dim=0)):
            assert torch.equal(source, back)


def test_fp8_projections_fuse_per_kind(loaded_fp8, checkpoint_fp8):
    _folder, raw = checkpoint_fp8
    attn, gdn = f"{LM}.layers.1.self_attn", f"{LM}.layers.0.linear_attn"
    _assert_fused_per_kind(loaded_fp8, raw, "model.layers.1.self_attn.qkv_proj", [f"{attn}.{p}_proj" for p in "qkv"])
    _assert_fused_per_kind(loaded_fp8, raw, "model.layers.0.linear_attn.in_proj_qkvz", [f"{gdn}.in_proj_qkv", f"{gdn}.in_proj_z"])
    assert torch.equal(loaded_fp8["model.layers.0.linear_attn.out_proj.weight_scale_inv"], raw[f"{gdn}.out_proj.weight_scale_inv"])
    assert torch.equal(
        loaded_fp8["model.layers.0.linear_attn.in_proj_ba.weight"],
        torch.cat([raw[f"{gdn}.in_proj_b.weight"], raw[f"{gdn}.in_proj_a.weight"]], dim=0),
    )
    for name in ("model.layers.0.linear_attn.in_proj_ba.weight", "model.layers.1.mlp.shared_expert.gate_up_proj.weight",
                 "model.layers.1.self_attn.indexer.index_qk_proj.weight", "lm_head.weight",
                 "model.hyper_connection_mixer.input_mix_weight_down.weight",
                 "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"):
        assert loaded_fp8[name].dtype is torch.bfloat16


ATTN = f"{LM}.layers.1.self_attn"
REJECTED = [
    pytest.param(None, lambda w: {f"{ATTN}.q_proj.weight": w, f"{ATTN}.q_proj.weight_scale_inv": _fp8_scale(w)},
                 r"q_proj\.weight_scale_inv", id="scale the config does not declare"),
    pytest.param(None, lambda w: {f"{ATTN}.o_proj.weight": w.to(torch.float8_e4m3fn)},
                 r"o_proj\.weight is torch\.float8", id="fp8 weight the config declares bf16"),
    pytest.param(FP8_DENSE_QUANT, lambda w: {f"{ATTN}.{p}_proj.weight": w.clone() for p in "qkv"},
                 r"[qkv]_proj\.weight is torch\.bfloat16", id="bf16 weight the config declares fp8"),
    pytest.param(FP8_DENSE_QUANT, lambda w: {f"{ATTN}.q_proj.weight": w[:-64].to(torch.float8_e4m3fn), f"{ATTN}.q_proj.weight_scale_inv": _fp8_scale(w)},
                 "128x128", id="part that is not a 128-row multiple"),
]


@pytest.mark.parametrize("quantization_config, tensors, match", REJECTED)
def test_checkpoint_disagreeing_with_its_quant_config_is_rejected(tmp_path, quantization_config, tensors, match):
    save_file(tensors(_bf16(2 * QH * AHD, H)), str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(_config_json(quantization_config)))
    with pytest.raises(ValueError, match=match):
        _load(str(tmp_path))


# ======================================================================================
# TP > 1: the slicer cuts, the stacked experts stay replicated
# ======================================================================================

GDN = "model.layers.0.linear_attn"
ATTN_LOCAL = "model.layers.1.self_attn"


def _tp_config():
    return parsed_config(
        num_layers=2, head_dim=AHD, num_q=QH, num_kv=KVH, index_head_dim=IHD, index_heads=2,
        budget=16, hidden=H, max_position=4096, rope_theta=10000.0,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=KH, linear_num_value_heads=VH,
        linear_key_head_dim=HD, linear_value_head_dim=HD,
        hc_lowrank=LR, ple_layer_ids=[1],
        num_experts=E, moe_intermediate_size=I, shared_expert_intermediate_size=I,
    )


def _slicer_at(monkeypatch, rank: int) -> _TpSlicer:
    from freetoken.distributed import info

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, 2))
    return _TpSlicer(_tp_config())


def test_tp2_gdn_qkv_and_conv_cuts_are_per_segment(monkeypatch):
    s0, s1 = _slicer_at(monkeypatch, 0), _slicer_at(monkeypatch, 1)
    qkv = torch.arange(3 * KH * HD * H).view(3 * KH * HD, H).to(torch.bfloat16)
    conv = torch.arange(3 * KH * HD * 4).view(3 * KH * HD, 1, 4).to(torch.bfloat16)
    for name, tensor in ((f"{GDN}.in_proj_qkv.weight", qkv), (f"{GDN}.conv1d.weight", conv)):
        got0, got1 = s0.shard(name, tensor), s1.shard(name, tensor)
        assert got0.shape[0] == sum(rows // 2 for rows in s0.qkv_rows), name
        # each rank holds its half of every segment, in q|k|v order, not a flat dim-0 half
        start = pos = 0
        for rows in s0.qkv_rows:
            seg = tensor[start : start + rows]
            assert torch.equal(got0[pos : pos + rows // 2], seg[: rows // 2]), name
            assert torch.equal(got1[pos : pos + rows // 2], seg[rows // 2 :]), name
            pos += rows // 2
            start += rows
    with pytest.raises(ValueError, match=r"rows is not the q\|k\|v layout"):
        s0.shard(f"{GDN}.in_proj_qkv.weight", qkv[:-4])


def test_tp2_gdn_row_and_column_cuts(monkeypatch):
    s0 = _slicer_at(monkeypatch, 0)
    z = torch.arange(VH * HD * H).view(VH * HD, H).to(torch.bfloat16)
    ba = torch.arange(VH * H).view(VH, H).to(torch.bfloat16)
    vec = torch.arange(VH).to(torch.float32)
    out = torch.arange(H * VH * HD).view(H, VH * HD).to(torch.bfloat16)
    norm = torch.arange(HD).to(torch.bfloat16)
    for name, tensor in ((f"{GDN}.in_proj_z.weight", z), (f"{GDN}.in_proj_b.weight", ba),
                         (f"{GDN}.in_proj_a.weight", ba), (f"{GDN}.dt_bias", vec),
                         (f"{GDN}.A_log", vec)):
        assert torch.equal(s0.shard(name, tensor), tensor.chunk(2, dim=0)[0]), name
    assert torch.equal(s0.shard(f"{GDN}.out_proj.weight", out), out.chunk(2, dim=1)[0])
    assert torch.equal(s0.shard(f"{GDN}.norm.weight", norm), norm)


def test_tp2_attention_cut_is_whole_heads_and_replicates_the_indexer(monkeypatch):
    s0 = _slicer_at(monkeypatch, 0)
    # q rows are per-head [q|gate] interleaved: the dim-0 half cut is whole heads
    q = torch.arange(2 * QH * AHD * H).view(2 * QH * AHD, H).to(torch.bfloat16)
    assert torch.equal(s0.shard(f"{ATTN_LOCAL}.q_proj.weight", q), q[: QH * AHD])
    k = torch.arange(KVH * AHD * H).view(KVH * AHD, H).to(torch.bfloat16)
    assert torch.equal(s0.shard(f"{ATTN_LOCAL}.k_proj.weight", k), k[: KVH * AHD // 2])
    o = torch.arange(H * QH * AHD).view(H, QH * AHD).to(torch.bfloat16)
    assert torch.equal(s0.shard(f"{ATTN_LOCAL}.o_proj.weight", o), o[:, : QH * AHD // 2])
    idx = torch.arange(5 * IHD * H).view(5 * IHD, H).to(torch.bfloat16)
    assert torch.equal(s0.shard(f"{ATTN_LOCAL}.indexer.index_qk_proj.weight", idx), idx)


def _reassemble(rank0: torch.Tensor, rank1: torch.Tensor, parts: list[int]) -> torch.Tensor:
    """Rebuild a merged buffer whose every part is a dim-0 TP half: each rank stores
    part0|part1|... halves, so a part comes back from concatenating the two ranks' slots."""
    halves = [p // 2 for p in parts]
    slots0 = rank0.split(halves, dim=0)
    slots1 = rank1.split(halves, dim=0)
    return torch.cat([torch.cat(pair, dim=0) for pair in zip(slots0, slots1)], dim=0)


def test_tp2_visual_shards_follow_the_vision_parallel_classes():
    from freetoken.models.qwen4_exp.weight import _shard_visual

    # attn.qkv: three equal q|k|v segments, a whole-head cut inside each
    seg = 4 * 8
    qkv = torch.arange(3 * seg * 3).view(3 * seg, 3).float()
    r0 = _shard_visual("visual.blocks.0.attn.qkv.weight", qkv, 0, 2)
    r1 = _shard_visual("visual.blocks.0.attn.qkv.weight", qkv, 1, 2)
    assert r0.shape == (3 * seg // 2, 3)
    half = seg // 2
    for s in range(3):
        rank0, rank1 = r0[s * half : (s + 1) * half], r1[s * half : (s + 1) * half]
        assert torch.equal(torch.cat([rank0, rank1]), qkv[s * seg : (s + 1) * seg])
    bias = torch.arange(3 * seg).float()
    b0 = _shard_visual("visual.blocks.0.attn.qkv.bias", bias, 0, 2)
    want_bias = torch.cat([bias[s * seg : (s + 1) * seg].chunk(2, 0)[0] for s in range(3)])
    assert torch.equal(b0, want_bias)
    # attn.proj / linear_fc2: row parallels, bias replicated
    proj = torch.arange(8 * 6).view(8, 6).float()
    assert torch.equal(_shard_visual("visual.blocks.0.attn.proj.weight", proj, 0, 2), proj.chunk(2, 1)[0])
    assert torch.equal(_shard_visual("visual.blocks.0.mlp.linear_fc2.weight", proj, 1, 2), proj.chunk(2, 1)[1])
    vec = torch.arange(8).float()
    assert torch.equal(_shard_visual("visual.blocks.0.attn.proj.bias", vec, 1, 2), vec)
    assert torch.equal(_shard_visual("visual.blocks.0.mlp.linear_fc2.bias", vec, 0, 2), vec)
    # linear_fc1: a single-segment column cut, bias cut along dim 0 too
    fc1 = torch.arange(10 * 3).view(10, 3).float()
    assert torch.equal(_shard_visual("visual.blocks.0.mlp.linear_fc1.weight", fc1, 0, 2), fc1.chunk(2, 0)[0])
    fb = torch.arange(10).float()
    assert torch.equal(_shard_visual("visual.blocks.0.mlp.linear_fc1.bias", fb, 1, 2), fb.chunk(2, 0)[1])
    # conv / pos embed / norms replicate
    norm = torch.arange(3).float()
    assert torch.equal(_shard_visual("visual.blocks.0.norm1.weight", norm, 1, 2), norm)


def test_tp2_rank_slices_concatenate_back_to_the_tp1_load(monkeypatch, checkpoint):
    from freetoken.distributed import info

    folder, _raw = checkpoint
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    full = _load(folder)
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 2))
    rank0 = _load(folder)
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(1, 2))
    rank1 = _load(folder)
    assert set(rank0) == set(rank1) == set(full)
    slicer = _TpSlicer(_tp_config())
    in_proj_rows = sum(rows // 2 for rows in slicer.qkv_rows) + VH * HD // 2 + VH  # qkv, z, b, a
    assert rank0["model.layers.0.linear_attn.in_proj.weight"].shape == (in_proj_rows, H)
    assert rank0["model.layers.0.linear_attn.conv1d.weight"].shape == (sum(r // 2 for r in slicer.qkv_rows), 1, 4)
    assert rank0["model.layers.1.self_attn.qkv_proj.weight"].shape == (QH * AHD + KVH * AHD, H)
    assert rank0["model.layers.0.linear_attn.out_proj.weight"].shape == (H, VH * HD // 2)
    cut1 = {f"model.layers.{l}.mlp.shared_expert.down_proj.weight" for l in (0, 1)}
    cut1 |= {"model.layers.0.linear_attn.out_proj.weight", "model.layers.1.self_attn.o_proj.weight"}
    cut1 |= {n for n in full if n.startswith("visual.") and n.endswith(("attn.proj.weight", "linear_fc2.weight"))}
    cut0 = {
        "model.embed_tokens.weight", "lm_head.weight",
        "model.layers.0.linear_attn.dt_bias", "model.layers.0.linear_attn.A_log",
    }
    cut0 |= {n for n in full if n.startswith("visual.") and n.endswith(("linear_fc1.weight", "linear_fc1.bias"))}
    # merged buffers: every part is cut per rank, so a rank holds part-halves in merge order
    segmented = {
        "model.layers.0.linear_attn.in_proj.weight": [*slicer.qkv_rows, VH * HD, VH, VH],
        "model.layers.0.linear_attn.conv1d.weight": list(slicer.qkv_rows),
        "model.layers.1.self_attn.qkv_proj.weight": [2 * QH * AHD, KVH * AHD, KVH * AHD],
        **{f"model.layers.{l}.mlp.shared_expert.gate_up_proj.weight": [I, I] for l in (0, 1)},
        **{n: [whole.shape[0] // 3] * 3 for n, whole in full.items()
           if n.startswith("visual.") and n.endswith(("attn.qkv.weight", "attn.qkv.bias"))},
    }
    for name, whole in full.items():
        if name in segmented:
            assert torch.equal(_reassemble(rank0[name], rank1[name], segmented[name]), whole), name
        elif name in cut0:
            assert torch.equal(torch.cat([rank0[name], rank1[name]], dim=0), whole), name
        elif name in cut1:
            assert torch.equal(torch.cat([rank0[name], rank1[name]], dim=1), whole), name
        else:
            assert torch.equal(rank0[name], whole) and torch.equal(rank1[name], whole), name


@pytest.fixture(scope="module")
def stacked_checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    """The bf16 release layout: pre-stacked routed experts next to the usual dense tensors.

    Seeded like ``checkpoint`` so every dense tensor matches it and the TP1 no-regression
    assertion below can compare the two loads key by key.
    """
    torch.manual_seed(0)
    raw = _raw_checkpoint()
    for layer in (0, 1):
        raw[f"{LM}.layers.{layer}.mlp.experts.gate_up_proj"] = _bf16(E, 2 * I, H)
        raw[f"{LM}.layers.{layer}.mlp.experts.down_proj"] = _bf16(E, H, I)
    folder, stored = _write_checkpoint(tmp_path_factory.mktemp("qwen4_exp_stacked_ckpt"), raw, None)
    for key in STACKED_KEYS:  # expose the stacked tensors under the emitted names too
        stored[key] = stored[f"{LM}.{key.split('.', 1)[1]}"]
    return folder, stored


STACKED_KEYS = {
    f"model.layers.{layer}.mlp.experts.{leaf}"
    for layer in (0, 1) for leaf in ("gate_up_proj", "down_proj")
}


def test_tp1_stacked_experts_load_alongside_the_dense_fusions(stacked_checkpoint, loaded, checkpoint):
    folder, raw = stacked_checkpoint
    full = _load(folder)
    assert set(full) - set(loaded) == STACKED_KEYS
    for name, tensor in full.items():
        if name in STACKED_KEYS:
            assert torch.equal(tensor, raw[name])
        else:
            assert torch.equal(tensor, loaded[name]), name


def _want_stacked_shard(name: str, whole: torch.Tensor, rank: int) -> torch.Tensor:
    """The offload banks shard experts on the intermediate axis (MoEConfig.local_intermediate):
    each rank keeps its gate|up rows together and the matching down columns."""
    i0 = I // 2
    if name.endswith("gate_up_proj"):
        return torch.cat(
            [whole[:, rank * i0:(rank + 1) * i0], whole[:, I + rank * i0:I + (rank + 1) * i0]], dim=1
        )
    return whole[..., rank * i0:(rank + 1) * i0]


def test_tp2_stacked_experts_follow_the_bank_intermediate_shard(monkeypatch, stacked_checkpoint):
    from freetoken.distributed import info

    folder, raw = stacked_checkpoint
    for rank in (0, 1):
        monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank, 2))
        got = _load(folder)
        for key in STACKED_KEYS:
            want = _want_stacked_shard(key, raw[key], rank)
            assert got[key].shape == want.shape, key  # a replicated or dim-0 down cut would show here
            assert torch.equal(got[key], want), key


def test_tp2_experts_only_load_yields_just_the_stacked_experts(monkeypatch, stacked_checkpoint):
    from freetoken.distributed import info

    folder, raw = stacked_checkpoint
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 2))
    got = {
        name: tensor.clone()
        for name, tensor in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=False
        )
    }
    assert set(got) == STACKED_KEYS
    for name, tensor in got.items():
        assert torch.equal(tensor, _want_stacked_shard(name, raw[name], 0)), name


def test_tp2_quantized_checkpoint_stays_tp1_only(monkeypatch, checkpoint_fp8):
    from freetoken.distributed import info

    folder, _raw = checkpoint_fp8
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 2))
    with pytest.raises(NotImplementedError, match="TP=1"):
        list(iter_weights(folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
