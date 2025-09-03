# Sampling

This document describes the sampling API in llama.cpp.

## Overview

llama.cpp provides a modular sampling system that allows chaining multiple sampling strategies together.

## Available Samplers

### Greedy

Selects the token with the highest probability.

```cpp
llama_sampler * sampler = llama_sampler_init_greedy();
```

### Dist

Samples from the probability distribution.

```cpp
llama_sampler * sampler = llama_sampler_init_dist(seed);
```

### Temperature

Applies temperature scaling to the logits.

```cpp
llama_sampler * sampler = llama_sampler_init_temp(temperature);
```

### Top-K

Keeps only the top K tokens with highest probability.

```cpp
llama_sampler * sampler = llama_sampler_init_top_k(k);
```

### Top-P (Nucleus)

Keeps tokens until their cumulative probability reaches P.

```cpp
llama_sampler * sampler = llama_sampler_init_top_p(p, min_keep);
```

### Typical

Samples from tokens that are typical for the distribution.

```cpp
llama_sampler * sampler = llama_sampler_init_typical(p, min_keep);
```

### Min-P

Keeps tokens with probability >= P * max_prob.

```cpp
llama_sampler * sampler = llama_sampler_init_min_p(p, min_keep);
```

### Mirostat

Applies Mirostat algorithm for controlling perplexity.

```cpp
llama_sampler * sampler = llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m);
```

### Mirostat v2

Applies Mirostat v2 algorithm.

```cpp
llama_sampler * sampler = llama_sampler_init_mirostat_v2(seed, tau, eta);
```

### Grammar

Applies GBNF grammar constraints.

```cpp
llama_sampler * sampler = llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
```

### Penalties

Applies repetition and frequency penalties.

```cpp
llama_sampler * sampler = llama_sampler_init_penalties(last_n, repeat, freq, present);
```

### DRY

Applies DRY sampling to prevent repetition.

```cpp
llama_sampler * sampler = llama_sampler_init_dry(vocab, n_ctx_train, multiplier, base, allowed_length, penalty_last_n, seq_breakers, num_breakers);
```

### DeepConf

Applies confidence-based filtering with early termination.

```cpp
llama_sampler * sampler = llama_sampler_init_deepconf(confidence_threshold, top_k, enable_early_termination, entropy_weight, top_k_weight);
```

## Chaining Samplers

Samplers can be chained together using `llama_sampler_chain_init`.

```cpp
llama_sampler * chain = llama_sampler_chain_init(params);
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));
llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
```

## DeepConf Sampler

The DeepConf sampler implements confidence-based token filtering based on entropy and top-k log probabilities.

### Parameters

- `confidence_threshold`: Minimum confidence score required (0.0 to 1.0)
- `top_k`: Number of top tokens to consider for confidence calculation
- `enable_early_termination`: Whether to limit candidates when confidence is low
- `entropy_weight`: Weight for entropy in confidence score (0.0 to 1.0)
- `top_k_weight`: Weight for top-k confidence in confidence score (0.0 to 1.0)

### Usage Example

```cpp
// Create DeepConf sampler
llama_sampler * deepconf = llama_sampler_init_deepconf(0.8f, 10, true, 0.6f, 0.4f);

// Chain with other samplers
llama_sampler * chain = llama_sampler_chain_init(params);
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));
llama_sampler_chain_add(chain, deepconf);
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

// Use in generation
llama_token token = llama_sampler_sample(chain, ctx, -1);
llama_sampler_accept(chain, token);
```

### Confidence Calculation

The confidence score is calculated as:

```
confidence = entropy_weight * (1.0 - normalized_entropy) + top_k_weight * average_top_k_logits
```

Where:

- `normalized_entropy` is the entropy divided by maximum possible entropy
- `average_top_k_logits` is the average of the top K logits

When confidence falls below the threshold and early termination is enabled, the sampler limits the candidate tokens to the top K.
