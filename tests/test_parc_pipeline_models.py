"""Optional real CPU backend checks; no downloads and no benchmark evidence."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/parc_pipeline"))
from parc_generation import (
    PromptMessage,
    PromptSegment,
    generate_one,
    load_generator,
    pack_prompt,
)
from parc_retrieval import SentenceTransformerEmbedder, TransformersReranker

HAS_MODELS = all(
    importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers")
)


@unittest.skipUnless(
    HAS_MODELS, "install requirements-full.txt for real CPU model smoke checks"
)
class RealBackendSmokeTest(unittest.TestCase):
    def test_real_sentence_transformer_normalized_embeddings(self):
        import numpy as np
        from sentence_transformers import SentenceTransformer, models
        from transformers import BertConfig, BertModel

        with tempfile.TemporaryDirectory() as folder:
            transformer_path = Path(folder) / "transformer"
            tokenizer = self.tokenizer()
            tokenizer.save_pretrained(transformer_path)
            BertModel(
                BertConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=1,
                    intermediate_size=32,
                )
            ).save_pretrained(transformer_path)
            model = SentenceTransformer(
                modules=[models.Transformer(str(transformer_path)), models.Pooling(16)]
            )
            model_path = Path(folder) / "embedder"
            model.save(str(model_path))
            values = SentenceTransformerEmbedder(str(model_path), "cpu", 2).encode(
                ["question", "evidence question"]
            )
            self.assertEqual(values.shape, (2, 16))
            np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-6)

    def tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "question": 3, "evidence": 4},
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
        )
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        return tokenizer

    def test_real_causal_model_load_pack_and_greedy_generation(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        with tempfile.TemporaryDirectory() as folder:
            tokenizer = self.tokenizer()
            tokenizer.save_pretrained(folder)
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=len(tokenizer),
                    n_layer=1,
                    n_head=1,
                    n_embd=16,
                    n_positions=128,
                    bos_token_id=2,
                    eos_token_id=2,
                    pad_token_id=1,
                )
            )
            model.save_pretrained(folder)
            loaded, tokenizer = load_generator(folder, "cpu", "float32", 42)
            config = {
                "decoding": {"max_input_tokens": 100, "target_input_tokens": 90},
                "packing": {
                    "priority": ["history", "context", "instruction"],
                    "omission_marker": "<CUT>",
                },
            }
            packed = pack_prompt(
                tokenizer,
                [
                    PromptMessage(
                        "user",
                        (PromptSegment("protected", "question evidence", "question"),),
                    )
                ],
                config,
            )
            output = generate_one(loaded, tokenizer, packed, "cpu", 3, True)
            self.assertIsInstance(output["prediction"], str)
            self.assertGreater(output["input_tokens"], 0)
            self.assertLessEqual(output["output_tokens"], 3)

    def test_real_reranker_load_and_batch_scoring(self):
        import numpy as np
        from transformers import BertConfig, BertForSequenceClassification

        with tempfile.TemporaryDirectory() as folder:
            tokenizer = self.tokenizer()
            tokenizer.save_pretrained(folder)
            BertForSequenceClassification(
                BertConfig(
                    vocab_size=len(tokenizer),
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=1,
                    intermediate_size=32,
                    num_labels=1,
                )
            ).save_pretrained(folder)
            reranker = TransformersReranker(folder, "cpu", 2, 32)
            scores = reranker.score(
                "question", ["evidence", "question evidence", "evidence evidence"]
            )
            self.assertEqual(scores.shape, (3,))
            self.assertTrue(np.isfinite(scores).all())
            np.testing.assert_array_equal(
                scores, scores.astype(np.float16).astype(np.float32)
            )
