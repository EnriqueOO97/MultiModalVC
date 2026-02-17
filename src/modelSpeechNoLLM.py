
import sys
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq.models import register_model
from .model import MMS_LLaMA, MMS_LLaMA_Config

# Local import from vendored module
from .divise_conformer.encoder import ConformerEncoder

logger = logging.getLogger(__name__)

@dataclass
class MMS_Speech_NoLLM_Config(MMS_LLaMA_Config):
    speech_upsample_factor: int = field(
        default=4, metadata={"help": "Upsample factor from video frames to audio frames (e.g. 4 for 25Hz->100Hz, 2 for 50Hz)"}
    )
    # Modality Dropout Probabilities
    p_modality_av: float = field(
        default=1.0, metadata={"help": "Probability of standard AV behavior (no dropout)"}
    )
    p_modality_video_only: float = field(
        default=0.0, metadata={"help": "Probability of zeroing audio encoder output"}
    )
    p_modality_audio_only: float = field(
        default=0.0, metadata={"help": "Probability of zeroing video encoder output"}
    )

@register_model("MMS_Speech_NoLLM", dataclass=MMS_Speech_NoLLM_Config)
class MMS_Speech_NoLLM(MMS_LLaMA):
    """
    No-LLM ablation model.
    Identical to MMS_LLaMA_Speech EXCEPT:
    - Does not load/use the LLM
    - proj1 input is Q-Former output (1024) instead of LLM output (3072)
    - Skips all instruction embedding and LLM forward pass
    """
    
    def __init__(self, avhubert, whisper, cfg):
        """
        Initialize WITHOUT LLM/tokenizer.
        We skip the parent's super().__init__ and manually set up.
        """
        # Skip MMS_LLaMA.__init__ which requires llm/tokenizer
        # Call grandparent (BaseFairseqModel) directly
        from fairseq.models import BaseFairseqModel
        BaseFairseqModel.__init__(self)
        
        self.cfg = cfg
        self.avhubert = avhubert
        self.whisper = whisper
        self.llama = None  # No LLM
        self.tokenizer = None  # No tokenizer
        
        # ===== COPIED EXACTLY FROM MMS_LLaMA.__init__ (lines 96-160) =====
        
        for param in self.avhubert.parameters():
            param.requires_grad = False
            
        for param in self.whisper.parameters():
            param.requires_grad = False
    
        self.modality_fuse = cfg.modality_fuse
        if self.modality_fuse == 'concat':
            self.embed = cfg.whisper_embed_dim + cfg.avhubert_embed_dim
        elif self.modality_fuse == 'add':
            self.embed = cfg.whisper_embed_dim
        elif self.modality_fuse == 'cross-att':
            from .sub_model.modules import Multimodal_Attention
            self.multimodal_attention_layer = Multimodal_Attention(embed_dim=cfg.whisper_embed_dim, num_heads=8)
            self.embed = cfg.whisper_embed_dim
        
        #### Qformer ####
        if cfg.use_qformer:
            import math
            import os
            from .sub_model.Qformer import BertConfig, BertLMHeadModel
            from .sub_model.modules import Projector, Speech_Rate_Predictor
            
            if cfg.window_level:
                cfg.max_queries = 1
            self.afeat_1d_conv = nn.Conv1d(in_channels=cfg.whisper_embed_dim, out_channels=cfg.whisper_embed_dim, kernel_size=2, stride=2, padding=0) # 50Hz -> 25Hz
            if cfg.use_sr_predictor:
                max_queries = int(cfg.queries_per_sec * 20 * 2)
            else:
                max_queries = int(cfg.queries_per_sec * 20)
                
            qformer_config = BertConfig.from_pretrained("bert-large-uncased")
            qformer_config.num_hidden_layers = cfg.qformer_layers
            qformer_config.encoder_width = self.embed
            qformer_config.hidden_size = cfg.qformer_dim 
            qformer_config.add_cross_attention = True
            qformer_config.cross_attention_freq = 1
            qformer_config.query_length = max_queries
            self.Qformer = BertLMHeadModel(config=qformer_config)
            self.query_tokens = nn.Parameter(
                torch.zeros(1, max_queries, qformer_config.hidden_size)
            )
            self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
            

            if cfg.use_sr_predictor:
                max_queries = int(cfg.queries_per_sec * 20 * 2)
                self.sr_predictor = Speech_Rate_Predictor(num_layers=cfg.sr_predictor_layers)
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sr_ckpt_path = f'{root_dir}/pretrained_models/sr_predictor/checkpoint.pt'
                sr_state = torch.load(sr_ckpt_path)['model']
                sr_state_ = {}
                for k, v in sr_state.items():
                    sr_state_[k[13:]] = v
                self.sr_predictor.load_state_dict(sr_state_)
                for param in self.sr_predictor.parameters():
                    param.requires_grad = False

            # DIFFERENCE: avfeat_to_llm projects to qformer_dim (1024), not llama_embed_dim (3072)
            self.avfeat_to_llm = Projector(input_dim=qformer_config.hidden_size,
                                        hidden_dim=math.floor((qformer_config.hidden_size + cfg.qformer_dim)/2),
                                        output_dim=cfg.qformer_dim) 
        else:
            self.afeat_1d_conv = nn.Conv1d(in_channels=cfg.whisper_embed_dim, out_channels=cfg.whisper_embed_dim, kernel_size=4, stride=4, padding=0) # 50Hz -> 12.5Hz
            self.vfeat_1d_conv = nn.Conv1d(in_channels=cfg.whisper_embed_dim, out_channels=cfg.whisper_embed_dim, kernel_size=2, stride=2, padding=0) # 25Hz -> 12.5Hz
            self.avfeat_to_llm = Projector(input_dim=self.embed,
                                        hidden_dim=math.floor((self.embed + cfg.qformer_dim)/2),
                                        output_dim=cfg.qformer_dim) 
            
            
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_params = [n for n,p in self.named_parameters() if p.requires_grad == False]
        
        # ===== END COPIED FROM MMS_LLaMA.__init__ =====
        
        # ===== COPIED EXACTLY FROM MMS_LLaMA_Speech.__init__ (lines 27-51) =====
        # DIFFERENCE: proj1 input is 1024 (qformer_dim) instead of 3072 (llama_embed_dim)
        
        # New Speech Generation Components
        # 1. Projection 1024 -> 768 (CHANGED from 3072 -> 768)
        self.proj1 = nn.Linear(cfg.qformer_dim, 768)
        
        # 2. Projection 768 -> 512
        self.proj2 = nn.Linear(768, 512)
        
        # 3. Conformer
        # We use size="L" as requested (attention_dim=512)
        # Note: ConformerEncoder expects 'size' argument.
        self.conformer = ConformerEncoder(size="L")
        
        # 4. Mel Head 512 -> 128
        self.mel_head = nn.Linear(512, 128)
        
        # LayerNorm for stabilizing activations between stages
        self.ln1 = nn.LayerNorm(768)   # after proj1, before interpolation
        self.ln2 = nn.LayerNorm(512)   # after proj2, before conformer
        self.ln3 = nn.LayerNorm(512)   # after conformer, before mel_head
        
        # Initialize weights for linear layers
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.constant_(self.proj1.bias, 0.0)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.constant_(self.proj2.bias, 0.0)
        nn.init.xavier_uniform_(self.mel_head.weight)
        nn.init.constant_(self.mel_head.bias, 0.0)
        
        # ===== END COPIED FROM MMS_LLaMA_Speech.__init__ =====

        # Learned mask embeddings for modality dropout
        self.audio_mask_emb = nn.Parameter(torch.FloatTensor(cfg.whisper_embed_dim).uniform_())   # (1024,)
        self.video_mask_emb = nn.Parameter(torch.FloatTensor(cfg.avhubert_embed_dim).uniform_())  # (1024,)

    @classmethod
    def build_model(cls, cfg, task):
        """
        Build model WITHOUT loading LLM.
        COPIED from MMS_LLaMA.build_model (lines 163-256) with LLM loading removed.
        """
        import os
        from fairseq import checkpoint_utils, tasks
        from fairseq.dataclass.utils import convert_namespace_to_omegaconf
        from argparse import Namespace
        from avhubert.hubert_asr import HubertEncoderWrapper
        from transformers import WhisperForConditionalGeneration
        from .sub_model.modules import WhisperEncoderWrapper
        
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if hasattr(cfg, 'w2v_path') and cfg.w2v_path is not None and cfg.w2v_path != '???':
            w2v_path = cfg.w2v_path
        else:
            w2v_path = f'{root_dir}/pretrained_models/avhubert/large_vox_iter5.pt'

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        avhubert = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            avhubert.w2v_model.load_state_dict(state["model"], strict=False)

        avhubert.w2v_model.remove_pretraining_modules()

        whisper_ = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").model.encoder
        whisper = WhisperEncoderWrapper(whisper_)

        # NO LLM LOADING - This is the key difference!
        
        return cls(avhubert, whisper, cfg)

    def forward(self, **kwargs):
        """
        Forward method. Defaults to forward_speech for this model.
        COPIED EXACTLY from MMS_LLaMA_Speech.forward (lines 53-57)
        """
        return self.forward_speech(**kwargs)

    def forward_speech(self, **kwargs):
        """
        Forward method for No-LLM Speech Synthesis.
        COPIED from MMS_LLaMA_Speech.forward_speech (lines 59-333)
        with LLM-specific code REMOVED.
        """
        # =========================================================================
        # PART A: PREPARE INPUTS (Identical to original forward steps 1-4)
        # COPIED EXACTLY from modelSpeech.py lines 64-130
        # =========================================================================
        
        # 1. Whisper & AVHubert feature extraction (no grad)
        with torch.no_grad():
            # Whisper encoder: B x T x D
            whisper_enc_out = self.whisper(kwargs['source'])

            # Prepare input for AVHubert (audio is None)
            avhubert_source = {'audio': None, 'video': kwargs['source']['video']}
            avhubert_output = self.avhubert(source=avhubert_source, padding_mask=kwargs['padding_mask'])
            # Transpose from (T x B x D) to (B x T x D)
            avhubert_output['encoder_out'] = avhubert_output['encoder_out'].transpose(0, 1)

        video_lengths = torch.sum(~avhubert_output['padding_mask'], dim=1).tolist()
        max_vid_len = max(video_lengths)
        
        # 2. Speech rate predictor and query length calculation
        if self.cfg.use_sr_predictor:
            len_queries, resized_len_list = self.query_length_calculation(whisper_enc_out, video_lengths, max_vid_len)
        else:
            len_queries = [max(int(vid_len / 25 * self.cfg.queries_per_sec), self.cfg.queries_per_sec) for vid_len in video_lengths]

        # 3. Feature processing and modality fusion
        whisper_enc_out = self.afeat_1d_conv(whisper_enc_out.transpose(1, 2)).transpose(1, 2) 

        if self.cfg.use_qformer:
            padding_mask = (~avhubert_output['padding_mask']).long()
            len_feat = video_lengths 
        else:
            padding_mask = avhubert_output['padding_mask'][:, 1::2]
            padding_mask = (~padding_mask).long()
            len_feat = torch.sum(padding_mask, dim=1).tolist()
            avhubert_output['encoder_out'] = self.vfeat_1d_conv(
                avhubert_output['encoder_out'].transpose(1, 2)
            ).transpose(1, 2)

        B_dim, T_v, _ = avhubert_output['encoder_out'].size()
        whisper_enc_out = whisper_enc_out[:, :T_v, :]

        # === MODALITY DROPOUT (training only) ===
        mode = 'av'  # default
        if self.training:
            mode = random.choices(
                ['av', 'video_only', 'audio_only'],
                weights=[self.cfg.p_modality_av, self.cfg.p_modality_video_only, self.cfg.p_modality_audio_only]
            )[0]
            if mode == 'video_only':
                whisper_enc_out = self.audio_mask_emb.unsqueeze(0).unsqueeze(0).expand_as(whisper_enc_out)
            elif mode == 'audio_only':
                avhubert_output['encoder_out'] = self.video_mask_emb.unsqueeze(0).unsqueeze(0).expand_as(avhubert_output['encoder_out'])

        # === NaN DEBUG LOGGING ===
        #print(f"[NaN-DEBUG] modality_mode={mode}")
        #print(f"[NaN-DEBUG] audio_mask_emb: min={self.audio_mask_emb.min().item():.4f}, max={self.audio_mask_emb.max().item():.4f}, has_nan={torch.isnan(self.audio_mask_emb).any().item()}")
        #print(f"[NaN-DEBUG] video_mask_emb: min={self.video_mask_emb.min().item():.4f}, max={self.video_mask_emb.max().item():.4f}, has_nan={torch.isnan(self.video_mask_emb).any().item()}")
        #print(f"[NaN-DEBUG] whisper_enc_out: min={whisper_enc_out.min().item():.4f}, max={whisper_enc_out.max().item():.4f}, has_nan={torch.isnan(whisper_enc_out).any().item()}")
        #print(f"[NaN-DEBUG] avhubert_out: min={avhubert_output['encoder_out'].min().item():.4f}, max={avhubert_output['encoder_out'].max().item():.4f}, has_nan={torch.isnan(avhubert_output['encoder_out']).any().item()}")

        # Fuse modalities
        if self.modality_fuse == 'concat':
            av_feat = torch.cat([whisper_enc_out, avhubert_output['encoder_out']], dim=2)
        elif self.modality_fuse == 'add':
            av_feat = whisper_enc_out + avhubert_output['encoder_out']
        elif self.modality_fuse == 'cross-att':
            av_feat = self.multimodal_attention_layer(
                audio_feature=whisper_enc_out,
                visual_feature=avhubert_output['encoder_out']
            )
        else:
            raise ValueError(f"Unknown modality fusion type: {self.modality_fuse}")

        #print(f"[NaN-DEBUG] av_feat after fusion: min={av_feat.min().item():.4f}, max={av_feat.max().item():.4f}, has_nan={torch.isnan(av_feat).any().item()}")

        # 4. Q-Former compression (COPIED from modelSpeech.py lines 121-129)
        if self.cfg.use_qformer:
            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            #print(f"[NaN-DEBUG] Q-Former output: min={query_output.min().item():.4f}, max={query_output.max().item():.4f}, has_nan={torch.isnan(query_output).any().item()}")
            query_output = self.avfeat_to_llm(query_output)  # (B, N_queries, 1024)
            #print(f"[NaN-DEBUG] after avfeat_to_llm: min={query_output.min().item():.4f}, max={query_output.max().item():.4f}, has_nan={torch.isnan(query_output).any().item()}")
            queries = query_output
            query_lengths = len_queries
        else:
            queries = self.avfeat_to_llm(av_feat)  # (B, N_feat, 1024)
            query_lengths = len_feat

        # =========================================================================
        # PART B: NO-LLM SPEECH GENERATION FLOW
        # Instead of LLM forward, we go directly to speech head.
        # The input to proj1 is Q-Former output (1024 dim) instead of LLM output (3072 dim).
        # =========================================================================
        
        # Extract valid query lengths for each batch item
        B = queries.size(0)
        av_lengths = query_lengths  # These are the valid lengths
        
        # Stack with padding (same logic as modelSpeech.py lines 233-238)
        max_av_len = max(av_lengths)
        av_hidden_padded = queries.new_zeros((B, max_av_len, queries.size(2)))
        
        for i in range(B):
            av_hidden_padded[i, :av_lengths[i], :] = queries[i, :av_lengths[i], :]
        
        #print(f"\n=== FORWARD PASS SHAPES ===")
        #print(f"[1] Q-Former output (no LLM): {av_hidden_padded.shape}")
        
        # Apply projection ONLY to AV tokens (COPIED from modelSpeech.py line 243)
        x = self.proj1(av_hidden_padded.to(self.proj1.weight.dtype))
        x = self.ln1(x)
        #print(f"[2] After proj1 + ln1 (1024->768): {x.shape}")
        
        # =========================================================================
        # INTERPOLATION - COPIED EXACTLY from modelSpeech.py lines 246-314
        # =========================================================================
        
        # Try to get raw audio lengths (in samples) from inputs.
        audio_lengths = None
        if isinstance(kwargs.get('source'), dict):
            audio_lengths = kwargs['source'].get('audio_lengths', None)
        if audio_lengths is None:
            audio_lengths = kwargs.get('audio_lengths', None)

        if audio_lengths is None:
            audio = kwargs['source'].get('audio', None) if isinstance(kwargs.get('source'), dict) else None
            if audio is not None:
                # If raw waveform is provided, infer lengths from tensor shape.
                if audio.dim() == 2:
                    audio_lengths = torch.full(
                        (audio.size(0),), audio.size(1), device=audio.device, dtype=torch.long
                    )
                elif audio.dim() == 3 and audio.size(1) == 1:
                    audio_lengths = torch.full(
                        (audio.size(0),), audio.size(2), device=audio.device, dtype=torch.long
                    )

        if audio_lengths is None:
            raise ValueError(
                "Audio lengths (in samples) are required for speech interpolation. "
                "Provide source['audio_lengths'] or audio_lengths in net_input."
            )

        # Compute target mel lengths per sample
        n_fft = 1024
        hop_length = 160
        pad = (n_fft - hop_length) // 2
        audio_lengths = audio_lengths.to(dtype=torch.long)
        target_lengths = torch.div(
            audio_lengths + 2 * pad - n_fft, hop_length, rounding_mode='floor'
        ) + 1
        target_lengths = torch.clamp(target_lengths, min=1)

        #DEBUG: Show the predicted target lengths for comparison with actual mel file lengths
        #print(f"[MODEL DEBUG] audio_lengths: {audio_lengths[:5].tolist()}... min={audio_lengths.min().item()}, max={audio_lengths.max().item()}")
        #print(f"[MODEL DEBUG] target_lengths (for interpolation): {target_lengths[:5].tolist()}... min={target_lengths.min().item()}, max={target_lengths.max().item()}, unique={len(set(target_lengths.tolist()))}")

        max_target_len = int(target_lengths.max().item())

        # Permute for interpolate: (B, C, T)
        x = x.transpose(1, 2)
        #print(f"[3] Transposed for interpolation: {x.shape}")

        # Per-sample interpolation to target audio-derived length, then pad to max
        B, C, T_av = x.size()
        x_up = x.new_zeros((B, C, max_target_len))
        
        for i in range(B):
            actual_av_len = av_lengths[i]
            x_slice = x[i:i+1, :, :actual_av_len]  # (1, C, actual_len)
            
            tgt_len = int(target_lengths[i].item())
            x_i = F.interpolate(x_slice, size=tgt_len, mode='linear', align_corners=False)
            x_up[i, :, :tgt_len] = x_i[0]

        # Permute back: (B, T, C)
        x = x_up.transpose(1, 2)
        #print(f"[4] After interpolation: {x.shape}")
        
        # =========================================================================
        # PROJECTION + CONFORMER + MEL HEAD
        # COPIED EXACTLY from modelSpeech.py lines 317-328
        # =========================================================================
        
        # 8. Projection 2: 768 -> 512
        x = self.proj2(x)
        x = self.ln2(x)
        #print(f"[5] After proj2 + ln2 (768->512): {x.shape}")
        
        # 9. Conformer
        x = self.conformer(x)
        x = self.ln3(x)
        #print(f"[6] After conformer + ln3: {x.shape}")
        
        # 10. Mel Head: 512 -> 128
        melspec = self.mel_head(x)
        #print(f"[7] Final melspec: {melspec.shape}")
        #print(f"=== END FORWARD PASS ===\n")
        
        # COPIED EXACTLY from modelSpeech.py lines 330-333
        return {
            "melspec": melspec,     # (B, T_audio, 128)
            "hidden_states": None   # No LLM hidden states
        }

    # =========================================================================
    # INHERITED METHODS from MMS_LLaMA
    # query_length_calculation and compression_using_qformer are inherited
    # =========================================================================

    def upgrade_state_dict_named(self, state_dict, name):
        from fairseq.models import BaseFairseqModel
        BaseFairseqModel.upgrade_state_dict_named(self, state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        from fairseq.models import BaseFairseqModel
        BaseFairseqModel.set_num_updates(self, num_updates)
        self.num_updates = num_updates
        

