# llmcalculator
(creative name, I know)

## About

It is a common problem when using LLMs to try and estimate how much RAM is required to load a model to VRAM, or if you can 
load it at all. There are some simple tricks, like a model's load requirement being (params/2e9) for q4, but more precision 
is often required for benchmarking and optimization purposes. Furthermore, you don't want to download tens of gigabytes of 
sharded models, only to not be able to run it at all! 

llmcalculator is a **cli tool** for Windows that allows you to calculate exactly how much RAM loading a model takes, as well as
how much of that is its context vs the model weights itself.

## Usage

There are two modes provided in the all-in-one file: an **interactive** mode and the **cli**. The former is made for **occasional use**
where you would like to test one or two models. It provides a nice formatted output to 3 digits of precision, as well as an interactive
and clear input scheme to make it easy. Additionally, it features safeguards, default fields with common values to make it easier for you,
and postprocessing to accept a wide variety of input schemes (for example, it will sanitize paths for you- no need to change every "\" in 
your Windows path to "/"!)

### CLI

The **cli** is designed for workflows and integrated systems where the executable is being supplied and its output read by 
other programs. The syntax is as follows:

```
llmcalculator.exe <path_to_config.json> <parameters (float, billions)> <quant_format (str, gguf OR exl2)> <ctx_size (int)> [<kv_cache_bit_size (IF exl2, 16/8/4)> <batch_size (IF gguf, int)>] [<bpw (IF exl2, float)> <quant_size (IF gguf, string)>]
```

where groups in **curly braces, []**, are exclusive: you include one from each group based on your desired quantization format.

- `path_to_config.json`
  - Path on your local machine. Supports relative and absolute paths.
  - Including quotes/any style of "/" is permitted.
  - A `config.json` can be found as output from most llm tools, eg. `llama.cpp`, `exllamav2`.
    - [Here is an example.](https://huggingface.co/microsoft/phi-4/blob/main/config.json)
- `parameters`
  - Model size in billions. Float.
  - Self-explanatory.
- `quant_format`
  - Either `gguf` or `exl2`.
  - Case insensitive.
- `ctx_size`
  - Context size. Int.
- `kv_cache_bit_size` (Conditional: `exl2` only)
  - Bit size of the KV cache used. Integer.
  - Values supported: `16`, `8`, `4` (note this parameter is technically bits/fp)
  - Use 16 if you're not sure what to use.
- `batch_size` (Conditional: `gguf` only)
  - The batch size used. Integer.
  - Use 512 if you aren't sure what to use.
- `bpw` (Conditional: `exl2` only)
  - Bits per weight. Float.
  - Example: For 2.5bpw, enter 2.5
- `quant_size` (Conditional: `gguf` only)
  - Type of quant used. String.
  - Options : `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ3_XXS`, `IQ3_XS`, `Q2_K`, `Q3_K_S`, `IQ3_S`, `IQ3_M`, `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `IQ4_NL`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`
  - Case insensitive.

The cli will output json-formatted data in format
```json
{
  "model_size": <modelsize>
  "context_size": <contextsize>
  "total_size": <totalsize>
}
```

## Roadmap

This project is **complete**. Guaranteed updates will only focus on bugs/speed improvements, but some other changes may be made.

- [ ]  C++ module integration- support `#include`ing the file in cpp workflows.
- [ ]  Native Linux/MacOS support
- [ ]  Support for estimating throughput/latencies
