import torch

import numpy as np
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

import logging

logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)


MODEL_NAME = "stt_en_fastconformer_hybrid_large_streaming_multi"
DEFAULT_LOOKAHEAD_SIZE = 80


class StreamingTranscription:
    def __init__(
        self,
        model_name=MODEL_NAME,
        lookahead_size=DEFAULT_LOOKAHEAD_SIZE,
        decoder_type="rnnt",
    ):
        self.asr_model = self._load_model(model_name, lookahead_size, decoder_type)
        self.preprocessor = self._init_preprocessor()
        self._init_streaming_params()

    def _load_model(self, model_name, lookahead_size, decoder_type):
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

        ENCODER_STEP_LENGTH = 80  # ms
        if model_name == "stt_en_fastconformer_hybrid_large_streaming_multi":
            if lookahead_size not in [0, 80, 480, 1040]:
                raise ValueError(
                    f"Invalid lookahead_size {lookahead_size}. Allowed values: 0, 80, 480, 1040 ms"
                )
            left_context_size = asr_model.encoder.att_context_size[0]
            asr_model.encoder.set_default_att_context_size(
                [left_context_size, int(lookahead_size / ENCODER_STEP_LENGTH)]
            )

        asr_model.encoder.setup_streaming_params()
        asr_model.change_decoding_strategy(decoder_type=decoder_type)

        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            decoding_cfg.compute_timestamps = False
            if hasattr(asr_model, "joint"):
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1

        asr_model.change_decoding_strategy(decoding_cfg)
        asr_model.eval()
        return asr_model

    def _init_preprocessor(self):
        cfg = OmegaConf.create(self.asr_model._cfg.preprocessor)
        OmegaConf.set_struct(cfg, False)
        cfg.dither = 0.0
        cfg.pad_to = 0
        cfg.normalize = "None"
        preprocessor = EncDecCTCModelBPE.from_config_dict(cfg)
        preprocessor.to(self.asr_model.device)
        return preprocessor

    def _init_streaming_params(self):
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = (
            self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        )
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        pre_encode_cache_size = (
            self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        )
        num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros(
            (1, num_channels, pre_encode_cache_size), device=self.asr_model.device
        )

    def _extract_transcriptions(self, hyps):
        if isinstance(hyps[0], Hypothesis):
            return [hyp.text for hyp in hyps]
        return hyps

    def _preprocess_audio(self, audio):
        device = self.asr_model.device
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length

    def transcribe_chunk(self, new_chunk):
        audio_data = new_chunk.astype(np.float32) / 32768.0
        processed_signal, processed_signal_length = self._preprocess_audio(audio_data)

        processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
        processed_signal_length += self.cache_pre_encode.shape[1]

        self.cache_pre_encode = processed_signal[
            :, :, -self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1] :
        ]

        with torch.no_grad():
            (
                self.pred_out_stream,
                transcribed_texts,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = self.asr_model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )

        final_streaming_tran = self._extract_transcriptions(transcribed_texts)
        self.step_num += 1

        return final_streaming_tran[0]

    def reset_transcription_cache(self):
        self._init_streaming_params()
        self.step_num = 0
