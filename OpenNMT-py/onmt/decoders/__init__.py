"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.transformer_dpp_prevl import TransformerDecoderDPPPrevl
from onmt.decoders.transformer_dpp_prevst import TransformerDecoderDPPPrevst
from onmt.decoders.transformer_dpp_prevl_p import TransformerDecoderDPPPrevlP
from onmt.decoders.transformer_dpp_prevl_EQK import TransformerDecoderDPPPrevlEQK

from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder,
           "transformerDPPPrevl":TransformerDecoderDPPPrevl,
           "transformerDPPPrevlP":TransformerDecoderDPPPrevlP,
           "transformerDPPPrevst":TransformerDecoderDPPPrevst,
           "transformerDPPPrevlEQK": TransformerDecoderDPPPrevlEQK}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec", "TransformerDecoderDPPPrevl",
           "TransformerDecoderDPPPrevst", "TransformerDecoderDPPPrevlP",
           "TransformerDecoderDPPPrevlEQK" ]
