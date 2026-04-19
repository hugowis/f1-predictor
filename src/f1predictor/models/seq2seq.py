"""
Sequence-to-sequence model with GRU for F1 lap time prediction.

Architecture:
- Encoder: GRU that processes input sequence of past laps
- Decoder: GRU that generates predictions for future laps
- Embeddings: learned representations for categorical features
- Teacher forcing: during training, feed ground-truth outputs back to decoder

Supports:
- Variable input/output sequence lengths
- Multi-year training with categorical embeddings
- Flexible feature handling (numeric + categorical)
"""

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import logging

from .base import BaseModel
from dataloaders.utils import get_numeric_columns, get_categorical_columns, get_boolean_columns, get_compound_columns

logger = logging.getLogger(__name__)


class Seq2Seq(BaseModel):
    """
    Sequence-to-sequence model with GRU encoder-decoder architecture.
    
    Used for stint-based sequences: takes 1-5 laps as input, predicts 1-5 laps as output.
    
    Parameters
    ----------
    input_size : int
        Number of input features (numeric + embedded categorical)
    output_size : int
        Number of output features (usually 1 for lap time)
    hidden_size : int, optional
        GRU hidden dimension. Default is 128.
    num_layers : int, optional
        Number of stacked GRU layers. Default is 2.
    dropout : float, optional
        Dropout rate. Default is 0.2.
    encoder_input_size : int, optional
        Encoder input size (may differ from output_size). Default = input_size.
    decoder_output_size : int, optional
        Decoder output features. Default = output_size.
    embedding_dims : dict, optional
        Embedding dimensions for categorical features.
        E.g., {'Driver': 32, 'Team': 16, 'Circuit': 16, 'Year': 8}
    vocab_sizes : dict, optional
        Vocabulary sizes for categorical features.
        E.g., {'Driver': 76, 'Team': 18, 'Circuit': 35, 'Year': 8}
    device : str, optional
        Device to place model on. Default is 'cpu'.
    
    Attributes
    ----------
    encoder : nn.GRU
        GRU encoder
    decoder : nn.GRU
        GRU decoder with embedding/normalization layer
    embeddings : nn.ModuleDict
        Categorical embeddings
    fc_encoder_output : nn.Linear
        Linear layer for encoder output
    fc_decoder_input : nn.Linear
        Linear layer to prepare decoder input
    
    Examples
    --------
    >>> config = {
    ...     'input_size': 33,
    ...     'output_size': 1,
    ...     'hidden_size': 128,
    ...     'num_layers': 2,
    ...     'dropout': 0.2,
    ...     'embedding_dims': {'Driver': 32, 'Team': 16},
    ...     'vocab_sizes': {'Driver': 76, 'Team': 18},
    ... }
    >>> model = Seq2Seq(**config)
    >>> 
    >>> # Encoder input: (batch, seq_len, input_size)
    >>> encoder_input = torch.randn(32, 5, config['input_size'])
    >>> 
    >>> # Decoder input: (batch, output_seq_len, output_size)
    >>> decoder_input = torch.randn(32, 5, config['output_size'])
    >>> 
    >>> # Forward with teacher forcing
    >>> output = model(encoder_input, decoder_input, teacher_forcing=True)
    >>> print(output.shape)  # (32, 5, 1)
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        decoder_type: str = 'gru',
        rnn_type: str = None,
        encoder_input_size: Optional[int] = None,
        decoder_output_size: Optional[int] = None,
        embedding_dims: Optional[Dict[str, int]] = None,
        vocab_sizes: Optional[Dict[str, int]] = None,
        device: str = 'cpu',
        decoder_extra_features_size: int = 7,
        use_mlp_decoder_head: bool = True,
        decoder_head_hidden: int = 64,
        decoder_head_layers: int = 2,
        decoder_head_dropout: float = 0.0,
        decoder_skip_type: str = 'none',
        encoder_pool: bool = False,
        encoder_pool_k: int = 5,
        **kwargs
    ):
        config = {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'decoder_type': decoder_type,
            'encoder_input_size': encoder_input_size or input_size,
            'decoder_output_size': decoder_output_size or output_size,
            'embedding_dims': embedding_dims or {},
            'vocab_sizes': vocab_sizes or {},
            'device': device,
            'decoder_extra_features_size': decoder_extra_features_size,
        }
        super().__init__(config)
        
        self.input_size = input_size
        self.output_size = output_size
        # Number of compound classes (one-hot compound columns)
        # Default to number of compound columns from dataloader utils
        try:
            self.compound_classes = kwargs.get('compound_classes', len(get_compound_columns()))
        except Exception:
            self.compound_classes = kwargs.get('compound_classes', 4)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        # Determine unified RNN type (GRU or LSTM).
        # Priority: explicit rnn_type param > decoder_type param > kwargs encoder/decoder > default 'gru'
        chosen = (rnn_type or decoder_type or kwargs.get('encoder_type') or kwargs.get('decoder_type') or 'gru')
        self.rnn_type = chosen.lower()
        if self.rnn_type not in ('gru', 'lstm'):
            raise ValueError("rnn_type must be either 'gru' or 'lstm'")
        # Keep legacy attributes for compatibility
        self.decoder_type = self.rnn_type
        self.encoder_type = self.rnn_type
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        self.total_embedding_size = 0

        if embedding_dims and vocab_sizes:
            for feature_name, emb_dim in embedding_dims.items():
                vocab_size = vocab_sizes.get(feature_name, 1)
                self.embeddings[feature_name] = nn.Embedding(vocab_size, emb_dim)
                self.total_embedding_size += emb_dim

        # Calculate actual input size for encoder.
        # If embeddings are present, assume `input_size` includes the categorical
        # placeholders; compute numeric base by subtracting number of categorical
        # features and add learned embedding size.
        num_categorical = len(self.embeddings) if len(self.embeddings) > 0 else 0
        if encoder_input_size is not None:
            self.encoder_input_size = encoder_input_size
        else:
            base_numeric = max(0, input_size - num_categorical)
            self.encoder_input_size = base_numeric + self.total_embedding_size
        
        # ENCODER: GRU or LSTM that processes past laps
        if self.rnn_type == 'gru':
            self.encoder = nn.GRU(
                input_size=self.encoder_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=False,
            )
        else:
            self.encoder = nn.LSTM(
                input_size=self.encoder_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=False,
            )
        
        # Encoder output projection for decoder hidden-state init (B3).
        # encoder_pool=False: project last encoder output (legacy).
        # encoder_pool=True:  project concat(last_output, mean_pool_last_k).
        if encoder_pool_k < 1:
            raise ValueError(f"encoder_pool_k must be >= 1, got {encoder_pool_k}")
        self.encoder_pool = bool(encoder_pool)
        self.encoder_pool_k = int(encoder_pool_k)
        proj_in = 2 * hidden_size if self.encoder_pool else hidden_size
        self.fc_encoder_to_hidden = nn.Linear(proj_in, hidden_size)
        
        self.decoder_extra_features_size = decoder_extra_features_size

        # DECODER: GRU that generates future laps.
        # Input = previous lap time + strategy-known extra features (tyre life,
        # fuel proxy, stint lap, compound one-hot) — all known at inference time.
        decoder_input_size = output_size + decoder_extra_features_size
        # Decoder can be GRU or LSTM based on `decoder_type`.
        if self.rnn_type == 'gru':
            self.decoder = nn.GRU(
                input_size=decoder_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=False,
            )
        else:
            # LSTM decoder
            self.decoder = nn.LSTM(
                input_size=decoder_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=False,
            )
        
        # Decoder output projection to lap time prediction.
        # Configurable MLP: [Linear -> GELU -> Dropout] x (layers-1) -> Linear.
        # layers=1 collapses to a single Linear (same as use_mlp_decoder_head=False).
        self.decoder_head_hidden = decoder_head_hidden
        self.decoder_head_layers = decoder_head_layers
        self.decoder_head_dropout = decoder_head_dropout
        if not use_mlp_decoder_head or decoder_head_layers <= 1:
            self.fc_decoder_output = nn.Linear(hidden_size, output_size)
        else:
            layers: list = []
            in_dim = hidden_size
            for _ in range(decoder_head_layers - 1):
                layers.append(nn.Linear(in_dim, decoder_head_hidden))
                layers.append(nn.GELU())
                if decoder_head_dropout > 0.0:
                    layers.append(nn.Dropout(decoder_head_dropout))
                in_dim = decoder_head_hidden
            layers.append(nn.Linear(in_dim, output_size))
            self.fc_decoder_output = nn.Sequential(*layers)

        # Skip connection over decoder extras (B2).
        # - 'additive': small MLP residual added to decoder output.
        # - 'film': per-step gamma/beta modulating decoder output.
        # Both variants are zero-initialised in _init_weights so an untrained
        # model starts as the identity ('none') transform.
        if decoder_skip_type not in ('none', 'additive', 'film'):
            raise ValueError(
                f"decoder_skip_type must be 'none', 'additive', or 'film'; got {decoder_skip_type!r}"
            )
        self.decoder_skip_type = decoder_skip_type
        if decoder_skip_type == 'additive':
            self.decoder_skip = nn.Sequential(
                nn.Linear(decoder_input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        elif decoder_skip_type == 'film':
            # Output = [gamma_residual, beta]; gamma = 1 + gamma_residual, so
            # zero-init => gamma=1, beta=0 (identity).
            self.decoder_film = nn.Linear(decoder_input_size, 2 * hidden_size)
        # Pit stop head (binary logit)
        self.fc_pit = nn.Linear(hidden_size, 1)
        # Compound prediction head (logits over compound classes)
        self.fc_compound = nn.Linear(hidden_size, self.compound_classes)
        
        self.to(device)
        self._init_weights()
    
    def _init_weights(self):
        """Initialise weights for stable training across random seeds.

        - Recurrent weights (weight_hh): orthogonal init — standard best
          practice for GRU/LSTM; prevents vanishing/exploding gradients and
          dramatically reduces seed sensitivity.
        - Input weights (weight_ih): Xavier uniform.
        - Biases: zeros (LSTM forget-gate bias set to 1 for better gradient
          flow through time).
        - Linear projection layers: Xavier uniform, bias = zeros.
        - Embeddings: normal(0, 0.1) for compact, stable initial vectors.
        """
        for rnn in [self.encoder, self.decoder]:
            for name, param in rnn.named_parameters():
                if 'weight_hh' in name:
                    # Orthogonal init for hidden-to-hidden (recurrent) weights
                    # Each GRU gate weight is a [hidden, hidden] block stacked;
                    # init each block independently.
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # For LSTM: set forget gate bias to 1
                    if self.rnn_type == 'lstm':
                        n = param.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)

        for fc in [
            self.fc_encoder_to_hidden,
            self.fc_pit,
            self.fc_compound,
        ]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

        if isinstance(self.fc_decoder_output, nn.Sequential):
            for layer in self.fc_decoder_output:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        else:
            nn.init.xavier_uniform_(self.fc_decoder_output.weight)
            nn.init.zeros_(self.fc_decoder_output.bias)

        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.1)

        # Skip connection identity init (B2): additive residual final layer is
        # zero -> skip returns 0; FiLM weights/bias zero -> gamma=1, beta=0.
        if self.decoder_skip_type == 'additive':
            first, last = self.decoder_skip[0], self.decoder_skip[-1]
            nn.init.xavier_uniform_(first.weight)
            nn.init.zeros_(first.bias)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif self.decoder_skip_type == 'film':
            nn.init.zeros_(self.decoder_film.weight)
            nn.init.zeros_(self.decoder_film.bias)

    def _apply_decoder_skip(self, decoder_output, decoder_input):
        """Combine decoder RNN output with a skip from the decoder input (B2)."""
        if self.decoder_skip_type == 'none':
            return decoder_output
        if self.decoder_skip_type == 'additive':
            return decoder_output + self.decoder_skip(decoder_input)
        # film
        film = self.decoder_film(decoder_input)
        gamma_res, beta = film.chunk(2, dim=-1)
        return (1.0 + gamma_res) * decoder_output + beta

    def _decode_step(self, current_input, decoder_hidden):
        """Run a single decoder step and project to output heads.

        Returns (step_output, step_pit_logit, step_comp_logits, decoder_hidden).
        """
        decoder_output, decoder_hidden = self.decoder(current_input, decoder_hidden)
        head_input = self._apply_decoder_skip(decoder_output, current_input)
        step_output = self.fc_decoder_output(head_input)
        step_pit_logit = self.fc_pit(decoder_output)
        step_comp_logits = self.fc_compound(decoder_output)
        return step_output, step_pit_logit, step_comp_logits, decoder_hidden

    def _apply_embeddings(self, encoder_input):
        """Apply categorical embeddings and return a single tensor.

        Handles three input formats:
        1. Dict with 'numeric' and 'categorical' keys (AutoregressiveLapDataloader)
        2. Raw tensor with total_dim == expected column count (old format)
        3. Plain tensor (already processed) — returned as-is

        Returns
        -------
        torch.Tensor
            Shape (batch, seq_len, processed_features)
        """
        target_device = self.device

        # Format 1: structured dict from dataloader
        if isinstance(encoder_input, dict) and 'numeric' in encoder_input and 'categorical' in encoder_input:
            numeric = encoder_input['numeric']
            categorical = encoder_input['categorical']

            numeric_tensor = numeric if isinstance(numeric, torch.Tensor) else torch.from_numpy(numeric)
            numeric_tensor = numeric_tensor.to(target_device)

            cat_tensor = categorical if isinstance(categorical, torch.Tensor) else torch.from_numpy(categorical)
            cat_tensor = cat_tensor.to(target_device)

            cat_names = encoder_input.get('cat_names', [])
            emb_list = []
            if cat_tensor.numel() > 0:
                for i in range(cat_tensor.shape[-1]):
                    feat_name = cat_names[i] if i < len(cat_names) else None
                    if feat_name and feat_name in self.embeddings:
                        idxs = cat_tensor[:, :, i].long()
                        emb_list.append(self.embeddings[feat_name](idxs))

            if emb_list:
                cat_emb = torch.cat(emb_list, dim=-1)
            else:
                cat_emb = torch.zeros(numeric_tensor.shape[0], numeric_tensor.shape[1], 0, device=target_device)

            return torch.cat([numeric_tensor, cat_emb], dim=-1)

        # Format 2: raw tensor with categorical indices at known positions
        if isinstance(encoder_input, torch.Tensor) and self.embeddings and encoder_input.dim() == 3:
            encoder_input = encoder_input.to(target_device)
            total_dim = encoder_input.size(-1)
            n_num = len(get_numeric_columns())
            n_cat = len(get_categorical_columns())
            n_bool = len(get_boolean_columns())
            n_comp = len(get_compound_columns())
            expected = n_num + n_cat + n_bool + n_comp
            if total_dim == expected:
                num_part = encoder_input[:, :, :n_num]
                cat_part = encoder_input[:, :, n_num:n_num + n_cat].long()
                bool_part = encoder_input[:, :, n_num + n_cat:n_num + n_cat + n_bool]
                comp_part = encoder_input[:, :, n_num + n_cat + n_bool:]

                emb_list = []
                for i, feat_name in enumerate(get_categorical_columns()):
                    if feat_name in self.embeddings:
                        emb_list.append(self.embeddings[feat_name](cat_part[:, :, i]))

                if emb_list:
                    cat_emb = torch.cat(emb_list, dim=-1)
                else:
                    cat_emb = torch.zeros(num_part.shape[0], num_part.shape[1], 0, device=target_device)

                return torch.cat([num_part, bool_part, comp_part, cat_emb], dim=-1)

        # Format 3: already processed tensor
        if isinstance(encoder_input, torch.Tensor):
            return encoder_input.to(target_device)

        return encoder_input

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        teacher_forcing: bool = True,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with encoder-decoder architecture.
        
        Parameters
        ----------
        encoder_input : torch.Tensor
            Shape (batch_size, encoder_seq_len, input_size)
            Sequence of past laps to encode
        decoder_input : torch.Tensor
            Shape (batch_size, decoder_seq_len, output_size)
            Ground truth outputs for decoder (used in teacher forcing)
        teacher_forcing : bool, optional
            Whether to use teacher forcing. Default is True.
        teacher_forcing_ratio : float, optional
            Probability of using teacher forcing (0.0 to 1.0). Default is 1.0 (always).
        
        Returns
        -------
        torch.Tensor
            Shape (batch_size, decoder_seq_len, output_size)
            Predicted lap times
        """
        # Apply categorical embeddings and build final encoder tensor
        encoder_input = self._apply_embeddings(encoder_input)

        batch_size = encoder_input.size(0)
        decoder_seq_len = decoder_input.size(1)
        
        # ENCODER: process input sequence
        enc_result = self.encoder(encoder_input)
        if self.encoder_type == 'lstm':
            encoder_output, (enc_h, enc_c) = enc_result
            encoder_hidden = enc_h
        else:
            encoder_output, encoder_hidden = enc_result
        # encoder_output: (batch, encoder_seq_len, hidden_size)
        # encoder_hidden: (num_layers, batch, hidden_size)
        
        # Use last encoder hidden state to initialize decoder (B3 optional pool).
        last_enc = encoder_output[:, -1, :]
        if self.encoder_pool:
            k = min(self.encoder_pool_k, encoder_output.size(1))
            pooled = encoder_output[:, -k:, :].mean(dim=1)
            enc_feat = torch.cat([last_enc, pooled], dim=-1)
        else:
            enc_feat = last_enc
        base_hidden = self.fc_encoder_to_hidden(enc_feat).unsqueeze(0)
        # expand to (num_layers, batch, hidden_size)
        expanded = base_hidden.expand(self.num_layers, batch_size, self.hidden_size).contiguous()
        if self.decoder_type == 'gru':
            decoder_hidden = expanded
        else:
            # LSTM: initialize (h0, c0). Use zeros for cell state.
            c0 = torch.zeros_like(expanded)
            decoder_hidden = (expanded, c0)
        
        # DECODER: generate output sequence
        # Fast path: when using full teacher forcing (deterministic), run the
        # decoder in a single vectorized call to avoid Python-level timestep
        # loop and many small kernel launches which drastically reduce GPU
        # utilization.
        if teacher_forcing and teacher_forcing_ratio >= 0.999:
            # decoder_input shape: (batch, seq_len, output_size)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # decoder_output: (batch, seq_len, hidden_size)
            head_input = self._apply_decoder_skip(decoder_output, decoder_input)
            lap_out = self.fc_decoder_output(head_input)  # (batch, seq_len, output_size)
            pit_out = self.fc_pit(decoder_output).squeeze(-1)  # (batch, seq_len)
            comp_out = self.fc_compound(decoder_output)  # (batch, seq_len, C)

            return {
                'lap': lap_out,
                'pit_logits': pit_out,
                'compound_logits': comp_out,
            }

        outputs_lap = []
        outputs_pit = []
        outputs_compound = []
        current_input = decoder_input[:, 0:1, :]  # Start with first ground truth (lap time)

        for t in range(decoder_seq_len):
            # Decoder step: single time step
            step_output, step_pit_logit, step_comp_logits, decoder_hidden = self._decode_step(current_input, decoder_hidden)

            outputs_lap.append(step_output)
            outputs_pit.append(step_pit_logit.squeeze(-1))
            outputs_compound.append(step_comp_logits)

            # Decide whether to use teacher forcing or own prediction
            use_teacher = teacher_forcing and (torch.rand(1).item() < teacher_forcing_ratio)

            if use_teacher and t < decoder_seq_len - 1:
                current_input = decoder_input[:, t+1:t+2, :]
            else:
                if self.decoder_extra_features_size > 0 and t < decoder_seq_len - 1:
                    current_input = torch.cat(
                        [step_output, decoder_input[:, t+1:t+2, self.output_size:]], dim=-1
                    )
                else:
                    current_input = step_output
        
        # Concatenate all outputs
        lap_out = torch.cat(outputs_lap, dim=1)  # (batch, decoder_seq_len, output_size)
        pit_out = torch.cat(outputs_pit, dim=1)  # (batch, decoder_seq_len)
        comp_out = torch.cat(outputs_compound, dim=1)  # (batch, decoder_seq_len, C)

        return {
            'lap': lap_out,
            'pit_logits': pit_out,
            'compound_logits': comp_out,
        }

    def encode(self, encoder_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence to latent representation.
        
        Parameters
        ----------
        encoder_input : torch.Tensor
            Shape (batch_size, seq_len, input_size)
        
        Returns
        -------
        tuple of torch.Tensor
            (encoder_output, encoder_hidden)
            - encoder_output: (batch_size, seq_len, hidden_size)
            - encoder_hidden: (num_layers, batch_size, hidden_size)
        """
        encoder_output, encoder_hidden = self.encoder(encoder_input)
        return encoder_output, encoder_hidden
    
    def decode(
        self,
        decoder_input: torch.Tensor,
        decoder_hidden: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode from hidden state to output sequence (autoregressive).
        
        Parameters
        ----------
        decoder_input : torch.Tensor
            Initial decoder input, shape (batch_size, 1, output_size)
        decoder_hidden : torch.Tensor
            Initial hidden state, shape (num_layers, batch_size, hidden_size)
        max_length : int, optional
            Maximum number of steps to decode. If None, uses decoder_input length.
        
        Returns
        -------
        torch.Tensor
            Shape (batch_size, max_length, output_size)
            Decoded sequence
        """
        batch_size = decoder_input.size(0)
        max_length = max_length or decoder_input.size(1)
        
        outputs_lap = []
        outputs_pit = []
        outputs_compound = []
        current_input = decoder_input[:, 0:1, :]

        for i in range(max_length):
            step_output, step_pit_logit, step_comp_logits, decoder_hidden = self._decode_step(current_input, decoder_hidden)

            outputs_lap.append(step_output)
            outputs_pit.append(step_pit_logit.squeeze(-1))
            outputs_compound.append(step_comp_logits)
            if self.decoder_extra_features_size > 0 and i < max_length - 1:
                current_input = torch.cat(
                    [step_output, decoder_input[:, i+1:i+2, self.output_size:]], dim=-1
                )
            else:
                current_input = step_output

        lap_out = torch.cat(outputs_lap, dim=1)
        pit_out = torch.cat(outputs_pit, dim=1)
        comp_out = torch.cat(outputs_compound, dim=1)

        return {
            'lap': lap_out,
            'pit_logits': pit_out,
            'compound_logits': comp_out,
        }
