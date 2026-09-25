"""Real-time text + vision (optionally + audio) emotion state and grounded replies for a character robot, trained on MELD."""

import warnings

import huggingface_hub
import transformers

# Keep command output readable: the load reports only say that base models drop their LM heads (expected),
# and the mask warning comes from inside WavLM's attention.
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
huggingface_hub.logging.set_verbosity_error()
huggingface_hub.utils.disable_progress_bars()
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")
