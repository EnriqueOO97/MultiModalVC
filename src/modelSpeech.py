
import sys
import logging
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
class MMS_LLaMA_Speech_Config(MMS_LLaMA_Config):
    speech_upsample_factor: int = field(
        default=4, metadata={"help": "Upsample factor from video frames to audio frames (e.g. 4 for 25Hz->100Hz, 2 for 50Hz)"}
    )

@register_model("MMS_LLaMA_Speech", dataclass=MMS_LLaMA_Speech_Config)
class MMS_LLaMA_Speech(MMS_LLaMA):
    def __init__(self, avhubert, whisper, llm, tokenizer, cfg):
        super().__init__(avhubert, whisper, llm, tokenizer, cfg)
        
        # New Speech Generation Components
        # 1. Projection 3072 -> 768
        self.proj1 = nn.Linear(cfg.llama_embed_dim, 768)
        
        # 2. Projection 768 -> 512
        self.proj2 = nn.Linear(768, 512)
        
        # 3. Conformer
        # We use size="L" as requested (attention_dim=512)
        # Note: ConformerEncoder expects 'size' argument.
        self.conformer = ConformerEncoder(size="L")
        
        # 4. Mel Head 512 -> 128
        self.mel_head = nn.Linear(512, 128)
        
        # Initialize weights for linear layers
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.constant_(self.proj1.bias, 0.0)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.constant_(self.proj2.bias, 0.0)
        nn.init.xavier_uniform_(self.mel_head.weight)
        nn.init.constant_(self.mel_head.bias, 0.0)
        
        # Cache for instruction token length (computed once on first forward pass)
        self.cached_instruction_len = None

    def forward(self, **kwargs):
        """
        Forward method. Defaults to forward_speech for this model.
        """
        return self.forward_speech(**kwargs)

    def forward_speech(self, **kwargs):
        """
        New forward method for Single-Pass Spectrogram Prediction.
        Bypasses autoregressive generation and LM head.
        """
        # =========================================================================
        # PART A: PREPARE INPUTS (Identical to original forward steps 1-4)
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

        # 4. Prepare inputs for LLM (SIMPLIFIED FOR SPEECH - no labels needed)
        instructions = kwargs['source']['instruction']  # List[torch.Tensor]
        
        if self.cfg.use_qformer:
            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            #print(f"[DEBUG] Q-Former Output Shape (before LLM proj): {query_output.shape}")
            query_output = self.avfeat_to_llm(query_output)  # (B, N_queries, 3072)
            queries = query_output
            query_lengths = len_queries
        else:
            queries = self.avfeat_to_llm(av_feat)  # (B, N_feat, 3072)
            query_lengths = len_feat
        
        # Build LLM inputs: [instruction_embeddings, av_query_embeddings]
        # No labels needed for speech synthesis!
        llm_input_list = []
        lengths = []
        
        for i in range(len(instructions)):
            instruction = instructions[i]
            len_query = query_lengths[i]
            query = queries[i][:len_query, :]
            
            # Embed instruction tokens
            inst_emb = self.llama.model.embed_tokens(instruction.unsqueeze(0)).squeeze(0)
            
            # Concatenate: [instruction, av_queries] - NO labels!
            combined = torch.cat([inst_emb, query], dim=0)
            llm_input_list.append(combined)
            lengths.append(combined.size(0))
        
        # Pad to max sequence length (left-padding like in original)
        max_seq_len = max(lengths)
        batch_size = len(llm_input_list)
        embedding_dim = llm_input_list[0].size(1)
        
        # Get pad embedding
        pad_token_id = self.tokenizer("<|finetune_right_pad_id|>").input_ids[1]
        pad_token_tensor = torch.tensor([pad_token_id], device=instructions[0].device)
        pad_embedding = self.llama.model.embed_tokens(pad_token_tensor).squeeze(0)
        
        # Initialize padded inputs
        llm_inputs = pad_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len, embedding_dim).clone()
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=instructions[0].device)
        
        for i, seq in enumerate(llm_input_list):
            seq_len = seq.size(0)
            # Left-pad: place sequence at right end
            llm_inputs[i, max_seq_len - seq_len:] = seq
            attention_mask[i, max_seq_len - seq_len:] = 1

        # =========================================================================
        # PART B: NEW SPEECH GENERATION FLOW
        # =========================================================================

        # 5. Run LLaMA Backbone (Backbone ONLY, no LM Head)
        # We use self.llama.model, which is the LlamaModel (transformers implementation)
        # It returns BaseModelOutputWithPast
        outputs = self.llama.model(
            inputs_embeds=llm_inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last hidden state: (Batch, Seq_Len_Total, 3072)
        # Note: Seq_Len_Total includes instruction tokens + query tokens + (optionally label/pad tokens)
        hidden_states = outputs.last_hidden_state

        # CRITICAL: We need to extract ONLY the tokens corresponding to the generated speech content.
        # In this training/forward setup, the 'visual/audio queries' were inserted into the input.
        # If we are effectively "transducing" these query tokens into spectrograms, we should focus on them.
        # However, the user said "lets say the output tokens in the last hidden state are 12 tokens... I want to linear projection... then interpolate".
        # This implies we take the whole sequence or the relevant part.
        # Given the instruction "continue from the last hidden state", we process the *entire* sequence output by the transformer, 
        # OR we might need to mask out instructions.
        # For now, I will process the whole hidden_state as is common in end-to-end models, or rely on valid lengths.
        # But wait, logic suggests we only want the "content" part. 
        # The user's prompt implies: "output tokens before they are converted into text tokens".
        # Since this is a forward pass replacing the text decoding, we are operating on the Latent representation of the input.
        # Let's proceed with projecting the full sequence `x`.
        
        # 6. Slice hidden_states to extract ONLY AV tokens (before projection)
        # Optimization: Cache instruction length on first forward pass
        B = hidden_states.size(0)
        max_seq_len = hidden_states.size(1)
        
        #print(f"\n=== FORWARD PASS SHAPES ===")
        #print(f"[1] LLaMA hidden_states: {hidden_states.shape}")
        
        # Compute or retrieve cached instruction length
        if self.cached_instruction_len is None:
            # First forward pass: compute and cache
            self.cached_instruction_len = instructions[0].size(0)
            # print(f"[CACHE] Computed instruction_len = {self.cached_instruction_len} (cached for future passes)")
        
        inst_len = self.cached_instruction_len
        
        # Simple slicing: Skip [padding + instruction], take rest (AV tokens)
        av_hidden_list = []
        av_lengths = []
        
        for i in range(B):
            q_len = len_queries[i]
            seq_len = inst_len + q_len
            padding_len = max_seq_len - seq_len
            av_start_idx = padding_len + inst_len
            
            # Extract AV tokens: (T_av, 3072)
            av_slice = hidden_states[i, av_start_idx:, :]  # Take from av_start to end
            av_slice = av_slice[:q_len, :]  # Trim to exact AV length
            
            av_hidden_list.append(av_slice)
            av_lengths.append(q_len)
        
        # Stack with padding
        max_av_len = max(av_lengths)
        av_hidden_padded = hidden_states.new_zeros((B, max_av_len, hidden_states.size(2)))
        
        for i, av_slice in enumerate(av_hidden_list):
            av_hidden_padded[i, :av_lengths[i], :] = av_slice
        
        #print(f"[2] Sliced AV tokens: {av_hidden_padded.shape}")
        
        # Apply projection ONLY to AV tokens
        x = self.proj1(av_hidden_padded.to(self.proj1.weight.dtype))
        #print(f"[3] After proj1 (3072->768): {x.shape}")
        
        # 7. Interpolation (Upsampling) tied to AUDIO length
        # Target length is computed from input audio sample length so it matches mel-spectrogram supervision.
        # Mel extraction params (see generateLogMel-Spectrograms.py):
        #   SR=16000, N_FFT=1024, HOP_LENGTH=160, pad=(N_FFT-HOP_LENGTH)/2
        # With center=False and reflective padding, mel length is:
        #   floor((L + 2*pad - N_FFT)/HOP_LENGTH) + 1
        # For these values this simplifies to floor(L / HOP_LENGTH).

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
        #print(f"[4] Transposed for interpolation: {x.shape}")

        # Per-sample interpolation to target audio-derived length, then pad to max
        # NOTE: x now contains ONLY AV tokens (already sliced above), so no need to slice again
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
        #print(f"[5] After interpolation: {x.shape}")
        
        # 8. Projection 2: 768 -> 512
        x = self.proj2(x)
        #print(f"[6] After proj2 (768->512): {x.shape}")
        
        # 9. Conformer
        x = self.conformer(x)
        #print(f"[7] After conformer: {x.shape}")
        
        # 10. Mel Head: 512 -> 128
        melspec = self.mel_head(x)
        #print(f"[8] Final melspec: {melspec.shape}")
        #print(f"=== END FORWARD PASS ===\n")
        
        return {
            "melspec": melspec,     # (B, T_audio, 80)
            "hidden_states": hidden_states # For debug/reference
        }
