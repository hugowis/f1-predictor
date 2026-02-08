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


class Seq2SeqGRU(BaseModel):
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
    >>> model = Seq2SeqGRU(**config)
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
        encoder_input_size: Optional[int] = None,
        decoder_output_size: Optional[int] = None,
        embedding_dims: Optional[Dict[str, int]] = None,
        vocab_sizes: Optional[Dict[str, int]] = None,
        device: str = 'cpu',
        **kwargs
    ):
        config = {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'encoder_input_size': encoder_input_size or input_size,
            'decoder_output_size': decoder_output_size or output_size,
            'embedding_dims': embedding_dims or {},
            'vocab_sizes': vocab_sizes or {},
            'device': device,
        }
        super().__init__(config)
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
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
        
        # ENCODER: GRU that processes past laps
        self.encoder = nn.GRU(
            input_size=self.encoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        
        # Encoder output projection (optional, for decoder initialization)
        self.fc_encoder_to_hidden = nn.Linear(hidden_size, hidden_size)
        
        # DECODER: GRU that generates future laps
        # Decoder input will be the previous lap time (autoregressive scalar).
        # If you later want to include embeddings or extra context per step,
        # adjust `decoder_input_size` and ensure `current_input` includes them.
        decoder_input_size = output_size
        
        self.decoder = nn.GRU(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        
        # Decoder output projection to lap time prediction
        self.fc_decoder_output = nn.Linear(hidden_size, output_size)
        
        # Optional: additional layer for better expressiveness
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)  # Concatenates encoder+decoder
        
        self.to(device)
    
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
        # If encoder_input is a structured dict from the dataloader, build the
        # final tensor here by applying categorical embeddings and concatenating
        # with numeric features. This centralizes embedding behavior in the model.
        if isinstance(encoder_input, dict) and 'numeric' in encoder_input and 'categorical' in encoder_input:
            numeric = encoder_input['numeric']
            categorical = encoder_input['categorical']

            if isinstance(numeric, torch.Tensor):
                numeric_tensor = numeric
            else:
                numeric_tensor = torch.from_numpy(numeric)

            # Ensure numeric on model device
            try:
                numeric_tensor = numeric_tensor.to(self.device)
            except Exception:
                pass

            emb_list = []
            if isinstance(categorical, torch.Tensor):
                cat_tensor = categorical
            else:
                cat_tensor = torch.from_numpy(categorical)

            # For each categorical feature, run through embedding if available
            if cat_tensor.numel() > 0:
                num_cat = cat_tensor.shape[-1]
                for i in range(num_cat):
                    feat_name = encoder_input.get('cat_names', [])[i] if i < len(encoder_input.get('cat_names', [])) else None
                    idxs = cat_tensor[:, :, i].long()
                    if feat_name and feat_name in self.embeddings:
                        emb_layer = self.embeddings[feat_name]
                        try:
                            emb_device = next(emb_layer.parameters()).device
                            idxs = idxs.to(emb_device)
                        except Exception:
                            emb_device = None
                        emb_out = emb_layer(idxs)
                        try:
                            emb_out = emb_out.to(numeric_tensor.device)
                        except Exception:
                            pass
                        emb_list.append(emb_out)

            if emb_list:
                cat_emb = torch.cat(emb_list, dim=-1)
            else:
                cat_emb = torch.zeros(numeric_tensor.shape[0], numeric_tensor.shape[1], 0, device=numeric_tensor.device)

            # Final encoder input: numeric features + embeddings
            encoder_input = torch.cat([numeric_tensor, cat_emb], dim=-1)

        # If encoder_input is a raw tensor (old dataloader format) and includes
        # categorical indices in fixed positions, extract them and apply
        # embeddings here as well.
        if isinstance(encoder_input, torch.Tensor) and self.embeddings and encoder_input.dim() == 3:
            total_dim = encoder_input.size(-1)
            # Assume dataloader order: numeric, categorical, boolean, compound
            n_num = len(get_numeric_columns())
            n_cat = len(get_categorical_columns())
            n_bool = len(get_boolean_columns())
            n_comp = len(get_compound_columns())
            expected = n_num + n_cat + n_bool + n_comp
            if total_dim == expected:
                # Slice parts
                num_part = encoder_input[:, :, 0:n_num]
                cat_part = encoder_input[:, :, n_num:n_num + n_cat].long()
                bool_part = encoder_input[:, :, n_num + n_cat:n_num + n_cat + n_bool]
                comp_part = encoder_input[:, :, n_num + n_cat + n_bool:n_num + n_cat + n_bool + n_comp]

                # Build embeddings
                emb_list = []
                for i, feat_name in enumerate(get_categorical_columns()):
                    if feat_name in self.embeddings:
                        emb_layer = self.embeddings[feat_name]
                        try:
                            idxs = cat_part[:, :, i].long().to(next(emb_layer.parameters()).device)
                        except Exception:
                            idxs = cat_part[:, :, i].long()
                        emb_out = emb_layer(idxs)
                        try:
                            emb_out = emb_out.to(num_part.device)
                        except Exception:
                            pass
                        emb_list.append(emb_out)

                if emb_list:
                    cat_emb = torch.cat(emb_list, dim=-1)
                else:
                    cat_emb = torch.zeros(num_part.shape[0], num_part.shape[1], 0, device=num_part.device)

                # Final numeric input: numeric + boolean + compound + embeddings
                encoder_input = torch.cat([num_part, bool_part, comp_part, cat_emb], dim=-1)

        batch_size = encoder_input.size(0)
        decoder_seq_len = decoder_input.size(1)
        
        # ENCODER: process input sequence
        encoder_output, encoder_hidden = self.encoder(encoder_input)
        # encoder_output: (batch, encoder_seq_len, hidden_size)
        # encoder_hidden: (num_layers, batch, hidden_size)
        
        # Use last encoder hidden state to initialize decoder
        decoder_hidden = self.fc_encoder_to_hidden(encoder_output[:, -1, :]).unsqueeze(0)
        # decoder_hidden: (1, batch, hidden_size) -> expand to (num_layers, batch, hidden_size)
        decoder_hidden = decoder_hidden.expand(self.num_layers, batch_size, self.hidden_size).contiguous()
        
        # DECODER: generate output sequence
        outputs = []
        current_input = decoder_input[:, 0:1, :]  # Start with first ground truth
        
        for t in range(decoder_seq_len):
            # Decoder step: single time step
            decoder_output, decoder_hidden = self.decoder(current_input, decoder_hidden)
            # Ensure decoder_hidden is contiguous for next iteration (GPU requirement)
            decoder_hidden = decoder_hidden.contiguous()
            # decoder_output: (batch, 1, hidden_size)
            
            # Project hidden state to output
            step_output = self.fc_decoder_output(decoder_output)  # (batch, 1, output_size)
            outputs.append(step_output)
            
            # Decide whether to use teacher forcing or own prediction
            use_teacher = teacher_forcing and (torch.rand(1).item() < teacher_forcing_ratio)
            
            if use_teacher and t < decoder_seq_len - 1:
                # Use ground truth from decoder_input
                current_input = decoder_input[:, t+1:t+2, :]
            else:
                # Use model's prediction (autoregressive)
                current_input = step_output
        
        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)  # (batch, decoder_seq_len, output_size)
        return outputs

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
        
        outputs = []
        current_input = decoder_input[:, 0:1, :]
        
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(current_input, decoder_hidden)
            step_output = self.fc_decoder_output(decoder_output)
            outputs.append(step_output)
            current_input = step_output
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
