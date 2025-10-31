# A Single Sample ASR STT Evaluation

I was tired of guessing which Whisper model size to use for speech-to-text, so I ran a quick evaluation on my own setup to figure it out.  

 ## Limitations 

 - This was a "back of the envelope" style experiment and not intended as a definitive evaluation of local ASR inference. 
 - My hardware (AMD, ROCM) isn't ideal for STT 
 - STT accuracy and ASR 
 - There are many variables when it comes to ASR accuracy ranging from microphone to  background noise conditions through to how you speak. I've seen measureable differences in results based upon the mic I use (etc)
 -  Inference: CPU (!)


## For Next Time!

I've attempted something like this (informally!) on Android with Futo but can never remember the results ...

## A Thought

This seemed like a big project but I thought it was worth it. If you're spending hours per day doing transcription (I am) then this is probably very worth your time. You can almost certainly set up a more robust evaluation than I can.

---

# My Question Set 

### Q1: How much does model size actually matter for accuracy?

And more specifically: **on my hardware** where does diminishing returns set in?

Diminishing returns in the case of STT means something like (to me): not worth STT becoming annoyingly laggy for minor gains in accuracy / decreases in WER. Colloquially, we'lll call this the "sweet spot."

Answer: on my HW, it's at Whisper Medium (approximately and accpetable latency / the "sweet spot" depends heavily upon what STT workload you're doing; for live transcription even small lags are obviously significantly more problematic).

![WER by Model Size](results/wer_by_size.png)

On my test sample:
- **tiny**: 15.05% WER - Fast but rough
- **base**: 9.95% WER - Significant improvement
- **small**: 11.17% WER - Slight regression (interesting!)
- **medium**: 6.07% WER - Best accuracy in my test
- **large-v3-turbo**: 7.04% WER - Good balance

**My takeaway**: The biggest accuracy jump was from tiny → base. After that, diminishing returns for the speed cost.

---

### Q2: Is faster-whisper really as good as OpenAI Whisper?'

This question can be extended to the other Whisper variants. These are wonderful, but I've always been curious about what accuracy was like.

**What I found:** On my test, yes - same accuracy, slightly faster.

![Engine Comparison](results/engine_comparison.png)

Testing the same base model on my hardware:
- **faster-whisper**: 9.95% WER in 5.01s
- **openai-whisper**: 9.95% WER in 6.17s

**My takeaway**: Identical accuracy on this sample, faster-whisper was ~1.2x faster. Good enough reason for me to use it.

---

### Q3: What's the speed vs. accuracy tradeoff?

**What I found:** For my use case, base or small seems like the sweet spot.

![Accuracy vs Speed Tradeoff](results/accuracy_speed_tradeoff.png)

![Speed by Model Size](results/speed_by_size.png)

On my hardware:
- **tiny**: Super fast (2.73s) but 15% WER is rough for my needs
- **base**: Good balance - 10% WER in 5s
- **small**: Similar to base, slightly slower
- **medium**: Best accuracy (6% WER) but 7x slower than tiny
- **large-v3-turbo**: 33s for 7% WER - more than I need

**My takeaway**: For daily transcription of my own voice, base or small hits the sweet spot for me.


---

### Q4: Which model should I use for my daily STT workflow?

There are many ASR models I would love to try out but on AMD it's just too much work to attempt to resolve dependencies. But if you're on NVIDIA and have CUDA ... there are many more questions worth exploring. On my hardware, and for round two, I would love to look at how much ASR outperforms "legacy" STT.

**My personal answer:** base model with faster-whisper.

**Why it works for me:**
- ~10% WER is acceptable for my daily use (I dictate a lot)
- 5 seconds per clip is fast enough
- 140MB model size is manageable
- Good balance for my workflow

**When I'd use something else:**
- **tiny**: Quick tests or long recordings where speed matters more
- **medium/large**: Publishing/professional work where I need better accuracy

---

## All Variants Tested

I also threw distil-whisper into the mix to see if it lived up to the speed claims.

![Variants Comparison](results/variants_comparison.png)

On my test:
- **faster-whisper**: 9.95% WER, 4.87s ✓
- **openai-whisper**: 9.95% WER, 6.51s
- **distil-whisper**: 21.6% WER, 38.49s ✗

**My takeaway**: distil-whisper was both slower AND less accurate on my sample. Unexpected, but that's what I got.

---

## Summary of My Results

- **Best accuracy** (on this sample): medium (6.07% WER)
- **Fastest**: tiny (2.73s)
- **My choice for daily use**: base (9.95% WER, 5s)
- **Recommended engine**: faster-whisper

---

## What's in This Repo

**Scripts** (in `scripts/`):
- `benchmark_stt.py` - Core benchmarking script
- `test_all_sizes.py` - Tests all Whisper sizes (tiny → large-v3)
- `compare_engines.py` - Compares faster-whisper vs openai-whisper
- `compare_fast_variants.py` - Compares different faster-whisper implementations
- `visualize_results.py` - Generates charts from benchmark results

**Results** (in `results/`):
- CSV files with WER, CER, and timing data from my tests
- Visualizations (PNG charts)
- Summary statistics

**Sample Data**:
- `audio/` - My test audio file
- `text/` - Reference transcription

**Documentation**:
- `answers.md` - Full analysis and methodology
- `setup.md` - How I set up my environment
- `models.md` - Info about the models tested
- `examples.md` - Usage examples if you want to run your own tests

---

## Running Your Own Tests

Want to benchmark on your own voice and hardware? Here's how:

1. Set up the conda environment (see `setup.md`)
2. Record your own audio samples and create reference transcriptions
3. Put audio in `audio/`, reference text in `text/`
4. Run the scripts:

```bash
# Test all model sizes
python scripts/test_all_sizes.py --audio audio/your_test.wav --reference text/your_test.txt

# Compare engines
python scripts/compare_engines.py --audio audio/your_test.wav --reference text/your_test.txt

# Generate visualizations from your results
python scripts/visualize_results.py
```

---

 ## Hardware Context

My test environment:
- **GPU**: AMD Radeon RX 7700 XT (but using CPU inference)
- **CPU**: Intel Core i7-12700F
- **RAM**: 64 GB
- **OS**: Ubuntu 25.04

Your performance will differ based on your setup.

---

## Models Storage

Models get downloaded to `~/models/stt/` with subdirectories for different engines (openai-whisper, faster-whisper, etc).

## License

MIT 
