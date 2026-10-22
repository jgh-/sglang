# SPDX-License-Identifier: Apache-2.0
"""Frames reach a caller-supplied callback while the decode is still running."""

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    encode_while_decoding as ewd,
)


class _FakeEncoder:
    def __init__(self):
        self.written = []
        self.closed = False
        self.aborted = False

    def write(self, frames):
        self.written.append(frames)

    def close(self):
        self.closed = True

    def abort(self):
        self.aborted = True


def _batch(**overrides):
    values = {
        "extra": {},
        "fps": 24,
        "num_frames": 4,
        "output_compression": None,
        "x264_preset": None,
        "save_output": True,
        "return_file_paths_only": True,
        "return_raw_frames": False,
        "return_trajectory_decoded": False,
        "data_type": None,
        "enable_frame_interpolation": False,
        "enable_upscaling": False,
    }
    values.update(overrides)
    return type("B", (), values)()


def test_callback_is_delivered_without_a_file():
    """A caller can ask for frames without asking for an MP4."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        encode_while_decoding as ewd,
    )

    seen = []
    stage = object.__new__(ewd.EncodeWhileDecoding)
    stage._batch = type(
        "B",
        (),
        {
            "extra": {"frame_callback": seen.append},
            "fps": 24,
            "num_frames": 4,
            "output_compression": None,
            "x264_preset": None,
        },
    )()
    stage._finish_frames = None
    stage._audio = None
    stage._audio_sample_rate = None
    stage._encoder = None
    stage._failed = False
    stage._writes_file = False
    stage.save_file_path = None
    stage._encoder_name = None

    import torch

    frames = torch.zeros(1, 3, 4, 8, 8)
    stage(frames)
    assert len(seen) == 1
    assert seen[0].shape == frames.shape
    assert stage.finish() is None


def test_callback_alone_is_enough_to_be_eligible():
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        encode_while_decoding as ewd,
    )

    batch = type("B", (), {"extra": {"frame_callback": lambda f: None}})()
    assert ewd.wants_frame_callback(batch)
    plain = type("B", (), {"extra": {}})()
    assert not ewd.wants_frame_callback(plain)


def _stage(monkeypatch, save_file_path="/tmp/x.mp4", batch=None):
    encoder = _FakeEncoder()
    monkeypatch.setattr(ewd.EncodeWhileDecoding, "_open", lambda self, frames: encoder)
    return ewd.EncodeWhileDecoding(save_file_path, batch or _batch()), encoder


def test_new_behaviour_callback_and_file_both_happen(monkeypatch):
    """With a callback registered the file is still written, as before."""
    received = []
    stage, encoder = _stage(
        monkeypatch, batch=_batch(extra={"frame_callback": received.append})
    )
    frames = torch.zeros(1, 3, 2, 8, 8)
    stage(frames)

    assert len(received) == 1 and received[0] is frames
    assert len(encoder.written) == 1
    assert stage.finish() == ["/tmp/x.mp4"]
    assert encoder.closed


def test_existing_behaviour_without_a_callback(monkeypatch):
    """No callback: identical to before - frames go to the encoder only."""
    stage, encoder = _stage(monkeypatch)
    frames = torch.zeros(1, 3, 2, 8, 8)
    stage(frames)

    assert len(encoder.written) == 1
    assert stage.finish() == ["/tmp/x.mp4"]


def test_callback_only_never_opens_a_file(monkeypatch):
    received = []
    stage, encoder = _stage(
        monkeypatch,
        save_file_path=None,
        batch=_batch(extra={"frame_callback": received.append}),
    )
    stage(torch.zeros(1, 3, 2, 8, 8))

    assert len(received) == 1
    assert encoder.written == []
    assert stage.finish() is None


def _original_eligibility(batch, server_args, batch_size):
    """Upstream's condition, copied verbatim, as the behaviour to preserve."""
    return (
        batch.save_output
        and batch.return_file_paths_only
        and not batch.return_raw_frames
        and not (batch.extra or {}).get("retain_decoded_frames", False)
        and batch.data_type == ewd.DataType.VIDEO
        and not batch.enable_frame_interpolation
        and not batch.enable_upscaling
        and batch_size == 1
        and not (batch.extra or {}).get("dynamic_batch_output_paths")
        and not server_args.enable_torch_compile
        and type(server_args.pipeline_config).post_decoding
        is PipelineConfig.post_decoding
        and ewd.current_platform.is_cuda()
    )


def test_eligibility_is_unchanged():
    """Characterisation: the gate must answer exactly as upstream does."""
    server_args = type(
        "S", (), {"enable_torch_compile": False, "pipeline_config": PipelineConfig()}
    )()
    cases = [
        _batch(data_type=ewd.DataType.VIDEO),
        _batch(data_type=ewd.DataType.IMAGE),
        _batch(save_output=False, data_type=ewd.DataType.VIDEO),
        _batch(return_raw_frames=True, data_type=ewd.DataType.VIDEO),
        _batch(return_file_paths_only=False, data_type=ewd.DataType.VIDEO),
        _batch(enable_upscaling=True, data_type=ewd.DataType.VIDEO),
        _batch(extra={"retain_decoded_frames": True}, data_type=ewd.DataType.VIDEO),
        _batch(extra={"frame_callback": lambda f: None}, data_type=ewd.DataType.VIDEO),
    ]
    for batch in cases:
        for batch_size in (1, 2):
            assert ewd.can_encode_while_decoding(batch, server_args, batch_size) == (
                _original_eligibility(batch, server_args, batch_size)
            )
    compiled = type(
        "S", (), {"enable_torch_compile": True, "pipeline_config": PipelineConfig()}
    )()
    batch = _batch(data_type=ewd.DataType.VIDEO)
    assert ewd.can_encode_while_decoding(batch, compiled, 1) == _original_eligibility(
        batch, compiled, 1
    )


def test_stage_is_created_for_a_callback_without_a_file(monkeypatch):
    """The callback must not require a file, which is the point of the hook."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages import decoding as dstage

    monkeypatch.setattr(dstage, "streamed_video_output_path", lambda *a, **k: None)
    monkeypatch.setattr(
        dstage.DecodingStage, "__init__", lambda self, *a, **k: None, raising=True
    )
    stage = object.__new__(dstage.DecodingStage)
    stage.vae = type("V", (), {"supports_decode_on_frames": True})()
    batch = type(
        "B",
        (),
        {
            "extra": {"frame_callback": lambda f: None},
            "return_trajectory_decoded": False,
            "latents": torch.zeros(1),
        },
    )()

    made = {}

    class _Fake:
        def __init__(self, path, batch, **kw):
            made["path"] = path
            made["kw"] = kw

    monkeypatch.setattr(dstage, "EncodeWhileDecoding", _Fake)
    monkeypatch.setattr(dstage, "isinstance", lambda obj, cls: True, raising=False)
    result = dstage.DecodingStage._encode_while_decoding(
        stage, batch, type("S", (), {})()
    )
    assert result is not None
    assert made["path"] is None


def test_stage_is_not_created_without_a_file_or_a_callback(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages import decoding as dstage

    monkeypatch.setattr(dstage, "streamed_video_output_path", lambda *a, **k: None)
    monkeypatch.setattr(
        dstage.DecodingStage, "__init__", lambda self, *a, **k: None, raising=True
    )
    stage = object.__new__(dstage.DecodingStage)
    stage.vae = type("V", (), {"supports_decode_on_frames": True})()
    batch = type(
        "B",
        (),
        {
            "extra": {},
            "return_trajectory_decoded": False,
            "latents": torch.zeros(1),
        },
    )()
    monkeypatch.setattr(dstage, "isinstance", lambda obj, cls: True, raising=False)
    assert (
        dstage.DecodingStage._encode_while_decoding(stage, batch, type("S", (), {})())
        is None
    )


def test_request_without_extra_is_tolerated():
    """Upstream fixtures use SimpleNamespace requests with no `extra` field."""
    from types import SimpleNamespace

    request = SimpleNamespace(fps=13, num_frames=13)
    assert ewd.wants_frame_callback(request) is False


def test_stream_with_no_path_and_a_callback(monkeypatch):
    """The shape H3 and LTX now build: no output path, callback only."""
    from types import SimpleNamespace

    received = []
    request = SimpleNamespace(
        fps=13,
        num_frames=4,
        output_compression=None,
        x264_preset=None,
        extra={"frame_callback": received.append},
    )
    stage = ewd.EncodeWhileDecoding(None, request)
    opened = []
    monkeypatch.setattr(stage, "_open", lambda frames: opened.append(frames) or None)

    stage(torch.zeros(1, 3, 4, 8, 8))

    assert len(received) == 1
    assert opened == []  # no encoder for a callback-only request
    assert stage.finish() is None


def test_callback_request_on_a_custom_stage_warns_instead_of_no_op(monkeypatch, caplog):
    """18 model stages decode themselves; a callback there must not vanish silently."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages import decoding as dstage

    class _Custom(dstage.DecodingStage):
        def forward(self, batch, server_args):  # overrides -> no streaming
            return None

    stage = object.__new__(_Custom)
    stage.vae = type("V", (), {"supports_decode_on_frames": True})()
    batch = type(
        "B",
        (),
        {
            "extra": {"frame_callback": lambda f: None},
            "return_trajectory_decoded": False,
            "latents": torch.zeros(1),
        },
    )()

    import logging

    with caplog.at_level(logging.WARNING):
        assert _Custom._encode_while_decoding(stage, batch, type("S", (), {})()) is None
    assert any("does not stream decoded frames" in r.message for r in caplog.records)


def test_no_warning_when_no_callback_was_asked_for(monkeypatch, caplog):
    from sglang.multimodal_gen.runtime.pipelines_core.stages import decoding as dstage

    class _Custom(dstage.DecodingStage):
        def forward(self, batch, server_args):
            return None

    stage = object.__new__(_Custom)
    stage.vae = type("V", (), {"supports_decode_on_frames": True})()
    batch = type(
        "B",
        (),
        {"extra": {}, "return_trajectory_decoded": False, "latents": torch.zeros(1)},
    )()

    import logging

    with caplog.at_level(logging.WARNING):
        assert _Custom._encode_while_decoding(stage, batch, type("S", (), {})()) is None
    assert not any(
        "does not stream decoded frames" in r.message for r in caplog.records
    )
