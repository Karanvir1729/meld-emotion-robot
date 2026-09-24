# MELD emotion robot

A small, local, real-time prototype for an emotion-aware character robot. One spoken utterance goes in
(its **audio** and its **transcript**); two things come out, in order:

1. a **structured state** with a MELD emotion category, calibrated confidence, and what the words and the
   voice each suggested, and
2. a **short in-character reply** ("Pip"), streamed token by token and grounded only in that state.

Track: **text + audio**. Dataset: [MELD](https://affective-meld.github.io/). Everything runs on a laptop
with **0.67 B parameters** in total (cap: 6 B). No remote calls.

```
[3] Rachel: I said I was fine.
    state: anger (0.71, high)  words=neutral  voice=anger  agree=False  MELD label: anger  [91 ms]
    Pip: Hey, you sound more fed up than fine. Want to tell me what happened?
    [first token 402 ms, done 1310 ms]
```
*(illustrative; measured examples are in the sections below)*

## Quick start

Requirements: macOS or Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg (only for converting your
own recordings), ~6 GB of disk, ~5 GB of RAM.

```bash
make setup      # environment (torch, transformers, ...)
make data       # MELD CSVs + 16 kHz audio, 1.5 GB download
make features   # cache frozen WavLM features, ~25 min on an M4
make train      # deployed model + ablations, ~45 min on an M4
make eval       # test-set tables -> runs/report.md
make demo       # replay a MELD dialogue through state + reply
make test       # unit tests (no data or models needed)
```

Then talk to it yourself:

```bash
uv run python -m emo.demo --text "I said I was fine." --audio clip.wav   # 16 kHz mono; ffmpeg -i in.m4a -ac 1 -ar 16000 clip.wav
uv run python -m emo.serve                                               # JSON Lines on stdin/stdout, for a robot controller
```

## How it works

```
 audio (16 kHz, <= 10 s) ─► WavLM-base-plus, frozen ─► mean of each of 13 hidden layers ─► learned layer mix ─► proj ─┐
                                                                                                  audio head (view) ┘   │
 text + previous turn ─────► DistilRoBERTa, fine-tuned ─► <s> vector ──────────────────────────────────────────────────┼─► fused head ─► emotion, probs
                                                                                                   text head (view) ┘   │
                                                                                                                        ▼
 state {emotion, confidence, certainty, views, context} ─► one bracketed "stage note" ─► Qwen2.5-0.5B-Instruct ─► reply
```

| Component | Model | Parameters | Role |
|---|---|---|---|
| Audio encoder | `microsoft/wavlm-base-plus` | 94.4 M | frozen; 13 time-averaged hidden states per clip |
| Text encoder | `distilbert/distilroberta-base` | 82.1 M | fine-tuned on MELD, sees the utterance and the previous turn |
| Heads | own code ([emo/model.py](emo/model.py)) | 0.5 M | layer mix + projection, text head, audio head, fused MLP, temperature |
| Reply generator | `Qwen/Qwen2.5-0.5B-Instruct` | 494.0 M | frozen; prompted only from the state |
| **Total on the inference path** | | **670.9 M** | cap is 6 000 M |

**Training** ([emo/train.py](emo/train.py)): the text encoder, the audio projection and all heads train
jointly on MELD train; WavLM stays frozen (its features are cached once). Loss = CE(fused) + 0.3·CE(text head)
+ 0.3·CE(audio head). For 15 % of training samples the audio is hidden, so the fused head also learns to work
from text alone (that is also what happens live when a clip is missing or shorter than 0.3 s). The best epoch is
picked on dev weighted-F1, then one temperature is fitted on dev so that `confidence` means something.

**State** ([emo/session.py](emo/session.py)): the fused head gives `emotion` and calibrated `probs`; the text
and audio heads give `views` ("the words alone read X, the voice sounds Y"); `certainty` buckets the calibrated
confidence (high ≥ 0.70, medium ≥ 0.45, else low). A `deque` of the last 4 turns is the dialogue memory: its last
entry is the classifier's context, all of it goes to the reply generator, and it is echoed in the state.

**Reply** ([emo/responder.py](emo/responder.py)): the state is rendered into one bracketed stage note in front
of the person's words, e.g. `[sounds angry; the words alone read calm but the voice sounds angry, you may gently
notice the mismatch; stay calm and steady, acknowledge the frustration, no jokes, no arguing] I said I was fine.`
The seven `STYLE` lines are the whole character policy. Replies are validated (1–30 words, one line, no
meta-language); otherwise a hand-written line for that emotion is used and the event says `"source": "fallback"`.

## Real-time: definition and measurements

Real-time here means **turn-level responsiveness for one person talking to the robot**. The caller
(push-to-talk or a voice-activity detector, out of scope) hands over one finished utterance with its transcript;
the robot's face should react within the human turn gap and speech should start well under a second later.
Targets, as p95 measured from the moment the utterance is handed over, models warm, one session:

| Event | Target p95 | Why |
|---|---|---|
| `state` | 300 ms | face / posture reacts inside the ~200–300 ms turn gap |
| first `token` | 800 ms | text-to-speech can start on the first words |
| `done` | 2 500 ms | the whole reply, ≤ 30 words |

The state is always emitted before the first token. "Input over time" is handled at three levels: a stream of
utterances per session (each answered before the next), the 4-turn memory across them, and streaming inside a
turn. Not real-time here: frame-level affect while the person is still speaking, and barge-in.

Measured numbers (`make bench`, 200 test utterances after 10 warm-up turns) are in the Results section.

## Interface: JSON Lines

`uv run python -m emo.serve` reads one request per line and streams the turn's events back, one per line.
A robot controller runs it as a subprocess. `python -m emo.demo` prints the same events for humans.

Request: `{"session_id": "s1", "text": "I said I was fine.", "audio_path": "clip.flac"}` (`audio_path`
optional; 16 kHz mono WAV/FLAC) or `{"session_id": "s1", "reset": true}`.

```json
{"event": "ready", "device": "mps", "parameters_millions": {"classifier": 82.59, "responder": 494.03, "audio_encoder": 94.38, "total": 671.0}}
{"event": "state", "session_id": "s1", "turn": 1, "text": "I said I was fine.", "audio_seconds": 1.9,
 "emotion": "anger", "confidence": 0.71, "certainty": "high",
 "probs": {"neutral": 0.12, "joy": 0.01, "sadness": 0.05, "anger": 0.71, "fear": 0.02, "disgust": 0.06, "surprise": 0.03},
 "views": {"text": "neutral", "audio": "anger", "agree": false},
 "context": [], "latency_ms": {"state": 91}}
{"event": "token", "session_id": "s1", "turn": 1, "text": " Hey"}
{"event": "done", "session_id": "s1", "turn": 1, "response": "Hey, you sound more fed up than fine. Want to tell me what happened?",
 "source": "llm", "prompt": "[sounds angry; ...] I said I was fine.", "latency_ms": {"state": 91, "first_token": 402, "response": 1310}}
```

Field rules: `emotion` and `views.text` are one of the seven MELD labels; `views.audio` is `null` without
usable audio; `agree` is `null` when either view is missing; `probs` has exactly seven keys and sums to 1;
`confidence = max(probs)`; `source` is `llm` or `fallback`; `context` holds at most 4 earlier turns; `error`
events carry a `message` and a `hint` and never stop the server.

## Results

RESULTS_PLACEHOLDER

## Decisions and trade-offs

| Decision | Choice | Why | Not taken |
|---|---|---|---|
| Track | text + audio | tone is what the brief is about, and the raw MELD video (10.9 GB) would not fit next to the models on this laptop | text + vision |
| Data | official MELD CSVs + a 1.5 GB HuggingFace repack of the clips as 16 kHz FLAC | the repack's own CSVs had stripped apostrophes; the official ones are clean UTF-8 | the 10.9 GB raw release |
| Audio encoder | WavLM-base-plus, frozen, 13 time-averaged layers, learned layer mix | the standard SUPERB recipe for emotion; caching the features once makes every audio experiment cheap, and the live path calls the same function | fine-tuning WavLM (hours per run, no cache) |
| Text encoder | DistilRoBERTa, fully fine-tuned | 6 layers train in ~8 min per epoch here and land in the published text-only range | a public "emotion" DistilRoBERTa (it was trained on MELD, so the test set would leak) |
| Dialogue context | the previous utterance as a second text segment | it is exactly the memory the robot keeps; ablated by the `both-nocontext` row | a dialogue-level model |
| Fusion | concat -> one hidden layer -> 7 classes, plus a text head and an audio head, 15 % modality dropout | one hidden layer can model "words say X but voice says Y"; the extra heads give the `views` for free; dropout keeps the fused head usable without audio | a single linear layer (purely additive), cross-attention |
| Confidence | one temperature fitted on dev; high / medium / low buckets | the reply prompt and a robot both gate on it, so it has to mean something; bucket accuracy is reported | raw softmax |
| Reply generator | Qwen2.5-0.5B-Instruct, frozen, bf16, prompted only from the state | smallest instruct model that holds a persona; 1 GB on disk | Qwen2.5-1.5B (3 GB would not fit), fine-tuning, few-shot prompts |
| ASR | none | the brief gives the transcript; whisper-tiny (38 M) fits the budget and plugs into `Session.step` in one line, but its word errors would need their own evaluation | whisper-tiny |
| Interface | in-process `Session` + JSON Lines on stdin/stdout | zero web dependencies, testable with a shell pipe, spawned by a robot controller | HTTP / websocket (a ~30-line wrapper) |
| Evidence | single seed, paired bootstrap interval on the audio gain, ablations, disagreement analysis | a small audio gain is inside seed noise; the interval says whether it is real | multi-seed runs (time) |

## Hardware and observed resources

HARDWARE_PLACEHOLDER

## Limitations and what was left out

**Intentionally left out** (each with its natural plug-in point): the vision track and the three-modality
extension (disk); reinforcement learning; ASR (`Session.step`); microphone capture, voice-activity detection and
barge-in (the caller's job); an HTTP or websocket server (wrap `Session`); speaker identity; fine-tuning WavLM;
class-weighted or focal losses; hyper-parameter search; multiple seeds; a larger or fine-tuned reply model;
text-to-speech and face animation; quantization (unnecessary at 0.67 B).

**Known limitations**
- MELD is sitcom audio: laugh track, music, overlapping speakers, and a studio mix; a robot's microphone will sound different.
- At deployment the previous turn is usually the robot's own reply, which MELD never contains; the `both-nocontext` row bounds how much the model depends on context.
- The classifier is utterance-level; fear and disgust are rare in MELD and score low for every model.
- Numbers are single-seed and MPS kernels are not bit-deterministic; small differences between runs are noise.
- The reply model is 0.5 B parameters: replies are on-tone but often generic. Quality is judged by the counterfactual check and a small manual sample, not an automatic score.
- When a reply fails validation, the fallback line replaces the tokens that were already streamed; consumers must treat `done.response` as final.

## Repository layout

```
emo/config.py          paths, labels, model ids, hyperparameters, thresholds, latency targets (single source of truth)
emo/data.py            download, manifests (previous-turn context, clip join), Dataset + collate
emo/audio.py           16 kHz loading, frozen WavLM encoder, feature cache
emo/model.py           EmotionModel: text encoder, audio branch, fused head, temperature; save/load
emo/train.py           training loop, dev selection, temperature fit, logits + summary per run
emo/evaluate.py        test tables, bootstrap CI for the audio gain, disagreement, calibration -> runs/report.md
emo/responder.py       persona, stage note, streamed Qwen generation, validation + fallback
emo/session.py         Classifier (one utterance -> state fields) and Session (memory + event stream)
emo/demo.py            replay a MELD dialogue or your own clip; human-readable output
emo/serve.py           JSON Lines over stdin/stdout
emo/bench.py           latency, memory, parameter budget -> runs/bench_<device>.json
emo/eval_responses.py  reply checks on dev: fallback rate, length, counterfactual sensitivity
tests/                 unit tests with stubs (data manifests, model heads, session contract)
runs/                  committed results (json / md); checkpoints and logits stay local
```

## Use of AI tools

The design, code and this write-up were produced with Claude (Claude Code) as a pair programmer; every
decision, number and limitation above was checked by running the code on the machine described in the
hardware section. Pretrained components are credited in the components table; everything under `emo/` is
original to this repository.
