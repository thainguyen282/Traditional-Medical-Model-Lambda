# General Vietnamese and medical benchmark comparison

Results are drawn from saved completed summaries in this output directory. CPT-SFT means the final `exp0015` adapter; separate CPT-only and SFT-only scores are unavailable.

| Benchmark | Qwen3-4B-Thinking-2507 | CPT-SFT exp0015 | 
| --- | --- | --- | 
| INCLUDE (550 questions) | 60.18% (331/550) | 10.91% (60/550) | 
| mLogiQA (80 questions) | 68.3% (55/80) | 26.25% (21/80) | 
| MT-AIME24 (30 questions) | 76.67% (23/30) | 0.00% (0/30) | 
| PolyMath (500 questions) | 59.80% (299/500) | 12% (60/500) | 
| Medical VM14K (564 questions) | 40.81 (230/564) | 62.77% (354/564) | 

The available medical comparison uses **Qwen3-4B-Instruct-2507**, which scores **46.81% (264/564)**. CPT-SFT exceeds that Instruct model by **15.96 percentage points**, or 90 questions. This is not a medical comparison against the unadapted Thinking base.

The general benchmarks use sampled thinking-enabled BF16 inference; the medical benchmark uses deterministic, 4-bit, constrained answer-only inference with thinking disabled. CPT-SFT has many invalid or truncated general-benchmark answers, so the observed general-score declines need to be interpreted alongside those diagnostics. Missing results prevent a complete aggregate or a controlled claim about the effect of adaptation across both domains.

- [General Vietnamese results, baseline comparisons, diagnostics, and source summaries](general_vi_baselines/README.md)
- [Medical results, evaluation settings, and source summaries](medical_baselines/README.md)
