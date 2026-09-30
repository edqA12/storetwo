# Local retrieval model

Upstream: BAAI/bge-small-zh-v1.5 (MIT).
ONNX conversion: Xenova/bge-small-zh-v1.5, model_quantized.onnx.
Downloaded from the public Hugging Face model repository on 2026-09-10.
SHA-256 values for the exact model/tokenizer files are recorded in manifest.json.

The application uses CLS pooling and L2 normalization. Query-only prefix:
为这个句子生成表示以用于检索相关文章：

The runtime uses CPUExecutionProvider with two intra-op threads. It does not
download files or send documents/questions to Hugging Face at runtime.
This model is general Chinese text retrieval, not a medical diagnostic model.

https://huggingface.co/BAAI/bge-small-zh-v1.5
https://huggingface.co/Xenova/bge-small-zh-v1.5
