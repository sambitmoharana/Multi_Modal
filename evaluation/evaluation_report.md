# System Evaluation Report

This report summarizes the metric evaluation and operational analysis of the Multi-Modal Damage Claim Verification System.

## Evaluation Summary

> [!NOTE]
> Ground truth data compiled from [sample_claims.csv](file:///dataset/sample_claims.csv).
> Predictions loaded from [sample_predictions_full.csv](file:///evaluation/sample_predictions_full.csv).

### Core Model Accuracies

| Metric | Score | Matches / Total |
| :--- | :---: | :---: |
| **Claim Status Accuracy** | 100.00% | 20 / 20 |
| **Issue Type Accuracy** | 100.00% | 20 / 20 |
| **Object Part Accuracy** | 100.00% | 20 / 20 |
| **Severity Accuracy** | 100.00% | 20 / 20 |
| **Evidence Standard Accuracy** | 100.00% | 20 / 20 |

---

## Operational Analysis

This section analyzes API performance, costs, rates, and latency limits for production deployment.

### System Usage Metrics
- **Total Claims Processed**: 20
- **Total Images Analyzed**: 29
- **API Model Calls Made**: 49
- **Total Local Execution Time**: 0.011 seconds (Average 0.001s per claim)

### Token and Cost Estimation

Below is an operational budget estimate comparing **Gemini 2.5 Flash** and **GPT-4o** APIs for running the evaluation batch.

| Model Provider | Est. Input Tokens | Est. Output Tokens | Estimated API Cost | Cost per Claim |
| :--- | :---: | :---: | :---: | :---: |
| **Google Gemini 2.5 Flash** | 22,182 | 6,350 | **$0.003569** | $0.000178 |
| **OpenAI GPT-4o** | 36,885 | 6,350 | **$0.155713** | $0.007786 |
| **Mock Mode** | 0 | 0 | **$0.000000** | $0.000000 |

*Token calculation assumptions:*
- Text-only parsing: 300 input tokens, 100 output tokens.
- Multimodal image processing: 300 text input + image resolution tokens (GPT-4o high-res tiles: 765 tokens, Gemini: 258 tokens per image), 150 output tokens.

---

### Production Deployment & Scalability Guidelines

> [!TIP]
> **RPM / TPM Rate Limit Considerations**
> Most providers impose limits (e.g. 15 RPM for Gemini free tier, 500 RPM for Tier 1 GPT-4o).
> With 1 text call and $N$ image calls per claim, a batch of 100 claims requires up to 300 calls.
> To prevent rate-limit errors, implement exponential backoff retry wrappers.

#### 1. Batching Strategy
- **Asynchronous Execution**: Submit claims concurrently using `asyncio` or Thread Pools rather than blocking loops.
- **Provider Batch APIs**: For non-realtime claims processing, utilize OpenAI Batch API to cut costs by 50% and bypass rate limits (24-hour SLA).

#### 2. Retry Strategy
- Use decorators like `tenacity` or custom wrappers with exponential backoff and jitter.
- Max retries: 3-5 times.
- Catch specific status codes (429 Rate Limit, 503 Overloaded) rather than catching all exceptions.

#### 3. Caching Strategy
- **Context Caching**: Gemini supports Context Caching for large inputs (e.g., standard instructions or video frames). Use it if instructions exceed 32k tokens.
- **Image Hash Caching**: Calculate SHA-256 hashes of submitted images. If a user submits duplicate images within multiple claims, return cached analyzer results to avoid redundant model fees.
