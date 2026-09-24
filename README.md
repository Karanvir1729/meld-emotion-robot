# MELD emotion robot

A small, local, real-time prototype for an emotion-aware character robot. One spoken utterance goes in
(its **audio** and its **transcript**); two things come out, in order:

1. a **structured state** with a MELD emotion category, calibrated confidence, and what the words and the
   voice each suggested, and
2. a **short in-character reply** ("Pip"), streamed token by token and grounded only in that state.

Track: **text + audio**. Dataset: [MELD](https://affective-meld.github.io/). Everything runs on a laptop
with **0.67 B parameters** in total (cap: 6 B). No remote calls.

```
[4] Joey: ‘Kay look, if I have to go to the doctor for anything it’s gonna be for this thing sticking out of my stomach!
    state: anger (0.62, medium)  words=anger  voice=anger  agree=True  MELD label: anger  [181 ms]
    Pip: Absolutely, I understand the urgency. The doctor is needed now.
    [first token 982 ms, done 1319 ms]
```
*(one turn of `make demo`, MELD test dialogue 85, on an Apple M4; the full trace is in [runs/demo_dialogue85.jsonl](runs/demo_dialogue85.jsonl))*

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), ~6 GB of disk, ~5 GB of RAM; ffmpeg only to
convert your own recordings. Everything was measured on an Apple M4 laptop (macOS); the code has no
macOS-only dependency, but Linux/CPU timings will differ.

```bash
make setup      # environment (torch, transformers, ...)
make test       # unit tests: no data or trained model needed (downloads the 330 MB text encoder once)
make data       # MELD CSVs + 16 kHz audio, 1.5 GB download
make features   # cache frozen WavLM features once, ~15 min on an M4
make train      # deployed model + three ablations, ~1 h on an M4
make eval       # test-set tables -> runs/report.md
make demo       # replay MELD test dialogue 85 through state + reply (trace -> runs/demo_dialogue85.jsonl)
make bench      # latency, memory, parameter budget -> runs/bench_<device>.json
make responses  # reply checks on dev utterances -> runs/responses.md
```

To try the demo without training: `make data` (or just the CSVs and the test archive), then `make checkpoint`
downloads the deployed model (165 MB) from the GitHub release into `runs/both/`, and `make demo` works.
`make smoke` (a one-minute training run on 128 utterances) checks the training path before the real runs.

Then talk to it yourself:

```bash
uv run python -m emo.demo --text "I said I was fine." --audio clip.wav   # 16 kHz mono; ffmpeg -i in.m4a -ac 1 -ar 16000 clip.wav
uv run python -m emo.serve                                               # JSON Lines on stdin/stdout, for a robot controller
```

## Demo: one input, both outputs

`make demo` replays MELD test dialogue 85 (Joey's hernia) utterance by utterance, every speaker treated as the
person talking to Pip. Each turn prints the state, then the reply as it streams, then the timings; the same
events are written as JSON Lines.

```
[1] Joey: But um, I don’t think it’s anything serious.
    state: neutral (0.56, medium)  words=neutral  voice=neutral  agree=True  MELD label: neutral  [430 ms]
    Pip: What else?
    [first token 1182 ms, done 1247 ms]
[2] Chandler: This sounds like a hernia. You have to—you-you—Go to the doctor!
    state: surprise (0.36, low)  words=surprise  voice=surprise  agree=True  MELD label: surprise  [83 ms]
    Pip: I see. It's important to address this issue promptly.
    [first token 824 ms, done 1307 ms]
[3] Joey: No way!
    state: surprise (0.32, low)  words=anger  voice=neutral  agree=False  MELD label: anger  [119 ms]
    Pip: Please go to the emergency room as soon as possible.
    [first token 858 ms, done 1244 ms]
[4] Joey: ‘Kay look, if I have to go to the doctor for anything it’s gonna be for this thing sticking out of my stomach!
    state: anger (0.62, medium)  words=anger  voice=anger  agree=True  MELD label: anger  [181 ms]
    Pip: Absolutely, I understand the urgency. The doctor is needed now.
    [first token 982 ms, done 1319 ms]
[5] Joey: Why did I have to start working out again?
    state: surprise (0.42, low)  words=surprise  voice=anger  agree=False  MELD label: disgust  [179 ms]
    Pip: I’m sorry to hear that. It might be time to consult a professional.
    [first token 948 ms, done 1530 ms]
[6] Joey: Damn you 15s!
    state: anger (0.71, high)  words=anger  voice=anger  agree=True  MELD label: anger  [121 ms]
    Pip: Sorry, I didn't mean to sound so negative. Let's talk about something else instead.
    [first token 853 ms, done 1415 ms]
```
The first turn includes one-off warm-up cost; the state is right on 4 of 6 turns and sees the anger in the
voice on turns 4 to 6.

**Same words, two voices.** "I know." spoken as two different MELD clips, no dialogue context:

```
$ uv run python -m emo.demo --text "I know." --audio data/meld/audio/test/dia17_utt8.flac     # MELD label: anger
    state: neutral (0.90, high)  words=neutral  voice=anger  agree=False  [469 ms]
    Pip: Oh, don't worry! I'm just being careful. What do you need help with?

$ uv run python -m emo.demo --text "I know." --audio data/meld/audio/test/dia133_utt9.flac    # MELD label: neutral
    state: neutral (0.92, high)  words=neutral  voice=neutral  agree=True  [308 ms]
    Pip: I see. How else could I assist today?
    Pip (fallback): Got it. What else is going on today?
```
(traces: [runs/demo_two_voices_dia17_utt8.jsonl](runs/demo_two_voices_dia17_utt8.jsonl),
[runs/demo_two_voices_dia133_utt9.jsonl](runs/demo_two_voices_dia133_utt9.jsonl); the second reply shows the
validation replacing assistant-speak with the fallback line)

The fused emotion stays `neutral` for such a short phrase (see *Observations* under Results), but the state
changes where the voice shows: `views.audio` and `agree`. The reply prompt turns that into a note ("the words
alone read calm but the voice sounds angry, you may gently notice the mismatch").

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
| **Total on the inference path** | | **671.0 M** | cap is 6 000 M |

**Training** ([emo/train.py](emo/train.py)): the text encoder, the audio projection and all heads train
jointly on MELD train; WavLM stays frozen (its features are cached once). Loss = CE(fused) + 0.3·CE(text head)
+ 0.3·CE(audio head). For 15 % of training samples the audio is hidden, so the fused head also learns to work
from text alone (that is also what happens live when a clip is missing or shorter than 0.3 s). The best epoch is
picked on dev weighted-F1, then one temperature is fitted on dev so that `confidence` means something.

**State** ([emo/session.py](emo/session.py)): the fused head gives `emotion` and calibrated `probs`; the text
and audio heads give `views` ("the words alone read X, the voice sounds Y"); `certainty` buckets the calibrated
confidence (high ≥ 0.70, medium ≥ 0.45, else low). A `deque` of the last 4 messages (two exchanges: the person and Pip alternate) is the
dialogue memory: its last entry is the classifier's context, all of it goes to the reply generator, and it is
echoed in the state as `context`.

**Reply** ([emo/responder.py](emo/responder.py)): the remembered messages become chat turns, and the state is
rendered into one bracketed stage note in front of the person's words, e.g. `[sounds angry; the words alone read
calm but the voice sounds angry, you may gently notice the mismatch; stay calm and steady, acknowledge the
frustration, no jokes, no arguing] I said I was fine.` At low certainty the note only says "hard to tell how
they feel, ask rather than assume". The seven `STYLE` lines are the whole character policy. Replies are
validated (1–30 words, one line, nothing out of character such as "I can't assist with that"); otherwise a
hand-written line for that emotion is used and the event says `"source": "fallback"`.

## Real-time: definition and measurements

Real-time here means **turn-level responsiveness for one person talking to the robot**. The caller
(push-to-talk or a voice-activity detector, out of scope) hands over one finished utterance with its transcript;
the robot's face should react within the human turn gap and speech should start well under a second later.
Targets, as p95 measured from the moment the utterance is handed over, models warm, one session:

| Event (`latency_ms` key) | Target p95 | Why |
|---|---|---|
| `state` (`state`) | 300 ms | face / posture reacts inside the ~200–300 ms turn gap |
| first `token` (`first_token`) | 800 ms | text-to-speech can start on the first words |
| `done` (`response`) | 2 500 ms | the whole reply, ≤ 30 words |

The state is always emitted before the first token. "Input over time" is handled at three levels: a stream of
utterances per session (each answered before the next), the 4-message memory across them, and streaming inside
a turn. Not real-time here: frame-level affect while the person is still speaking, and barge-in.

Measured numbers (`make bench`, 200 test utterances after 10 warm-up turns) are in the Results section.

## Interface: JSON Lines

`uv run python -m emo.serve` reads one request per line and streams the turn's events back, one per line.
A robot controller runs it as a subprocess. `python -m emo.demo` prints the same events for humans.

Request: `{"session_id": "s1", "text": "I know.", "audio_path": "clip.flac"}` (`audio_path` optional;
16 kHz mono WAV/FLAC) or `{"session_id": "s1", "reset": true}`. The events of one real turn (the "I know."
clip above; the `ready` line is what `serve` prints at start-up):

```json
{"event": "ready", "device": "mps", "parameters_millions": {"classifier": 82.59, "responder": 494.03, "audio_encoder": 94.38, "total": 671.0}}
{"event": "state", "session_id": "s1", "turn": 1, "text": "I know.", "audio_seconds": 4.18, "emotion": "neutral", "confidence": 0.9, "certainty": "high", "probs": {"neutral": 0.9, "joy": 0.023, "sadness": 0.016, "anger": 0.031, "fear": 0.015, "disgust": 0.009, "surprise": 0.005}, "views": {"text": "neutral", "audio": "anger", "agree": false}, "context": [], "latency_ms": {"state": 469}}
{"event": "token", "session_id": "s1", "turn": 1, "text": "Oh, "}
{"event": "done", "session_id": "s1", "turn": 1, "response": "Oh, don't worry! I'm just being careful. What do you need help with?", "source": "llm", "prompt": "[sounds calm; the words alone read calm but the voice sounds angry, you may gently notice the mismatch; keep it light and ask a small follow-up question] I know.", "latency_ms": {"state": 469, "first_token": 1249, "response": 1847}}
```

Field rules: `emotion` and `views.text` are one of the seven MELD labels; `views.audio` is `null` without
usable audio; `agree` is `null` when either view is missing; `probs` has exactly seven keys and sums to 1;
`confidence = max(probs)` (both rounded to 3 decimals, so `probs` sums to 1 within rounding); `source` is
`llm` or `fallback`; `context` holds at most 4 earlier messages; a `done` event carries an `error` field when
generation failed and the fallback line was used; `error` events (bad request, unreadable audio) carry a
`message` and a `hint` and never stop the server.

## Results

All numbers below come from `make eval`, `make bench` and `make responses` on the machine in the hardware
section, single seed, and are reproduced verbatim in [runs/](runs/). Dev was used for epoch selection and the
temperature; test logits (2 610 utterances) are computed once per run after all selection is done and are read
only by `evaluate.py`; nothing was tuned on them.

### Emotion classification (MELD test)

| model | weighted-F1 | macro-F1 | accuracy | neutral | joy | sadness | anger | fear | disgust | surprise |
|---|---|---|---|---|---|---|---|---|---|---|
| majority class (neutral) | 0.313 | 0.093 | 0.481 | 0.650 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| text only | 0.619 | 0.434 | 0.636 | 0.785 | 0.588 | 0.352 | 0.427 | 0.149 | 0.192 | 0.542 |
| audio only | 0.479 | 0.287 | 0.495 | 0.650 | 0.348 | 0.232 | 0.415 | 0.000 | 0.000 | 0.363 |
| text + audio, no context | 0.629 | 0.422 | 0.642 | 0.788 | 0.601 | 0.368 | 0.498 | 0.000 | 0.156 | 0.541 |
| **text + audio (deployed)** | **0.631** | 0.419 | 0.643 | 0.790 | 0.600 | 0.387 | 0.504 | 0.000 | 0.111 | 0.538 |
| deployed model, text head only (`views.text`) | 0.619 | 0.418 | 0.633 | 0.784 | 0.585 | 0.349 | 0.473 | 0.033 | 0.178 | 0.525 |
| deployed model, audio head only (`views.audio`) | 0.450 | 0.239 | 0.522 | 0.684 | 0.222 | 0.161 | 0.382 | 0.000 | 0.000 | 0.222 |

For scale: the original MELD paper's bc-LSTM baselines reach roughly 0.56 (text), 0.39 (audio) and 0.60
(text + audio) weighted-F1; modern text-only transformers reach 0.62–0.66. Training took 31 min (text),
1 min (audio), 16 min and 20 min (the two fused runs) on the M4.

**Does audio add measurable value?** weighted-F1(text + audio) − weighted-F1(text) = **+0.012**, paired
bootstrap 95 % interval **[−0.001, +0.025]** over 1 000 resamples of the test utterances. So: a small gain
that is not statistically significant on this test set. Audio helps most on anger (+0.08 F1) and sadness
(+0.04), and mostly adds a second opinion. The text and audio heads disagree on **43.3 %** of test turns; on
those turns the fused head is right 50.4 % of the time, the text head 48.3 %, the audio head 22.7 %. Feeding
the previous utterance to the text encoder (`context`) changed nothing on test (+0.002).

**Confidence means something.** After temperature scaling (T = 1.20), the certainty buckets on test are:

| certainty | share of test turns | accuracy |
|---|---|---|
| high (≥ 0.70) | 43.0 % | 82.7 % |
| medium (≥ 0.45) | 35.1 % | 58.4 % |
| low | 22.0 % | 37.5 % |

**Observations.** Fear (2 % of MELD) is predicted 11 times by the fused model, never correctly, and disgust
gets 5 of its 68 test utterances; the confusion matrix in [runs/report.md](runs/report.md) shows both
collapsing into neutral and anger. The learned
WavLM layer mix stayed almost uniform (0.073–0.081 per layer), so the audio branch effectively uses the mean
of all 13 layers. **Text dominates the fused head**: for a short phrase such as "I know." or "No." spoken
angrily, sadly or happily, the fused emotion stays `neutral` at 0.9 confidence while `views.audio` moves with
the voice. In MELD the words predict the label far better than the voice, so this is what the fused head
learned; a higher modality dropout would push it towards the voice at some cost in weighted-F1 (not tried). In
this prototype the voice therefore reaches the robot through `views.audio` and `agree`, which the reply prompt
turns into "the words alone read calm but the voice sounds angry".

### Replies

`make responses`: 105 dev utterances (15 per true emotion), each run as a single turn through the full
pipeline, then again with greedy decoding, and once more with only the emotion fields of the note changed.

| check | result |
|---|---|
| replies that fail validation and fall back to the hand-written line | 13.3 % (almost all assistant-speak such as "how can I help you", which is out of character) |
| mean reply length | 13.3 words (limit 30) |
| greedy reply changes when only the emotion in the note changes (89 pairs where the model wrote both replies) | 66.3 % |

The last row is the grounding check: for two thirds of the turns the words of the reply change when only the
emotion changes; for the rest the model gives the same generic line either way (necessary, not sufficient,
evidence). What it looks like, from [runs/responses.md](runs/responses.md):

| person said | state | Pip (greedy) | if the state were... | Pip would say instead |
|---|---|---|---|---|
| What? | neutral | What's up? | joy | That's great to hear! Let's chat some more! |
| Or! Or, we could go to the bank, close our accounts and cut them off at the source. | anger | I'm sorry to hear that. Let's try something different. | fear | Absolutely, that's a solid plan. Just make sure to secure all important documents before proceeding. |
| Oh my God, he's lost it. He's totally lost it. | sadness | I'm sorry to hear that. Let's try to find him quickly. | anger | *(unchanged)* I'm sorry to hear that. Let's try to find him quickly. |

Quality, honestly: replies are on tone but often generic ("I'm sorry to hear that. Let's talk more about
this."). The 0.5 B reply model is the weakest link of the prototype; the classifier, the state contract and
the streaming path would carry a larger model unchanged.

### Latency and memory

`make bench`: 200 consecutive test utterances in dialogue order through `Session` after 10 warm-up turns,
one process, models warm, nothing else running. All times are measured from the moment the utterance is handed
over; `first token` is when the streamer releases the first word.

| device | `state` p50 / p95 | first `token` p50 / p95 | `done` p50 / p95 | decode | fallback | peak RSS | GPU memory |
|---|---|---|---|---|---|---|---|
| target (p95) | 300 ms | 800 ms | 2 500 ms | | | | |
| M4 GPU (MPS) | **111 / 268 ms** | 866 / **1 047 ms** | **1 330 / 1 904 ms** | 35.2 tok/s | 16 % | 2.0 GB | 2.9 GB |
| M4 CPU only (50 turns) | 102 / 198 ms | 1 064 / 1 202 ms | 1 814 / 2 856 ms | 19.1 tok/s | 12 % | 4.2 GB | – |

Two of the three targets are met on the GPU. The first-token target is missed by ~0.25 s at p95: the reply
model has to prefill ~250 prompt tokens (persona, up to 4 remembered messages, the note) before its first word,
and a 0.5 B model in bf16 on MPS does that in ~0.7 s. The state, which drives the robot's face, is there after
~110 ms, and the whole reply is spoken-ready in ~1.3 s. Cold start (loading all three models and one warm-up
turn) is 7 s. The small classifier is as fast on the CPU as on the GPU; only the reply model needs the GPU.
The fallback rate here (16 %) is the validation catching assistant-speak on sitcom lines, see *Replies*.

## Decisions and trade-offs

| Decision | Choice | Why | Not taken |
|---|---|---|---|
| Track | text + audio | tone is what the brief is about, and the raw MELD video (10.9 GB) would not fit next to the models on this laptop | text + vision |
| Data | official MELD CSVs + a 1.5 GB HuggingFace repack of the clips as 16 kHz FLAC | the repack's own CSVs had stripped apostrophes; the official ones are clean UTF-8. The repack's test archive also carries MELD's 132 re-extracted clips (`final_videos_test…`), which replace their mis-cut twins | the 10.9 GB raw release |
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

- Apple M4 (10 cores), 16 GB unified memory, macOS 26.2; Python 3.12, torch 2.14.0, transformers 5.17.0, uv.
- Parameters on the inference path (counted at start-up, printed by `serve` and `bench`): classifier 82.59 M
  + audio encoder 94.38 M + reply model 494.03 M = **671.0 M**, 11 % of the 6 B cap.
- Disk: environment 0.9 GB, MELD audio + manifests 1.5 GB, cached WavLM features 0.27 GB, model weights
  1.7 GB (HuggingFace cache), one checkpoint 0.16 GB. About 5 GB in total.
- Training (all on the GPU): text 31 min (it shared the GPU with the feature cache), audio 1 min, fused 16–20 min per run; feature cache 15 min once.
- Inference: 2.0 GB resident memory (4.2 GB CPU-only), 2.9 GB GPU memory, 7 s cold start; per-turn numbers above.

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
