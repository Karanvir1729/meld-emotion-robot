# MELD emotion robot

A small, local, real-time prototype for an emotion-aware character robot. One spoken utterance goes in: a short
**video clip** (the person's face and voice) and its **transcript**. Two things come out, in order:

1. a **structured state** with a MELD emotion category, calibrated confidence, and what the words, the voice and
   the face each suggested, and
2. a **short in-character reply** ("Pip"), streamed token by token and grounded only in that state.

Primary track: **text + vision** (built and measured in full). Deployed: the brief's optional extension,
**text + vision + voice** in one real-time system, because a character robot has a camera and a microphone on
the person. On the MELD test set it is also the only combination that beats text alone with a confidence
interval clear of zero; that result was known when the deployed model was chosen (see Results). Dataset:
[MELD](https://affective-meld.github.io/). Everything runs on a laptop with **3.35 B parameters** in total
(cap: 6 B). No remote calls.

```
[7] Ross: Straight up over your head!
    state: anger (0.88, high)  words=anger  voice=anger  face=anger  agree=True  MELD label: anger  [217 ms]
    Pip: Got it, steady now. Just breathe and lift.
    [first token 707 ms, done 962 ms]
```
*(one turn of `make demo`: MELD test dialogue 2, the real clip decoded and read live on an Apple M4: the words,
the voice and the face all read anger; the full trace is in [runs/demo_dialogue2.jsonl](runs/demo_dialogue2.jsonl))*

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg and curl (video decoding and the
streamed download), ~8 GB of disk, a 16 GB machine. Everything was measured on an Apple M4 laptop (macOS). On
other platforms the same code runs with the reply model in bf16 through transformers: real-time needs a CUDA
GPU there.

```bash
make setup      # environment (torch, transformers, opencv, ...)
make test       # unit tests: no data or trained model needed (downloads the 330 MB text encoder once)
make data       # MELD CSVs + 16 kHz audio (ajyy/MELD_audio on Hugging Face, 1.5 GB) and the manifests
make faces      # stream the 10.9 GB MELD videos once, keep only face crops (~0.5 GB on disk, ~20 min)
make features   # cache frozen facial-expression and WavLM features once, ~40 min on an M4
make train      # deployed model + the ablations in the tables, ~80 min on an M4
make eval       # test-set tables -> runs/report.md
make demo       # replay MELD test dialogue 2 through state + reply (trace -> runs/demo_dialogue2.jsonl)
make bench      # latency, memory, parameter budget -> runs/bench_<device>.json
make responses  # reply checks on dev utterances -> runs/responses.md
```

To try the demo without training: `make data` and `make faces`, then `make checkpoint` downloads the deployed
model (165 MB) from the GitHub release into `runs/text-audio-vision/` (it uses the GitHub CLI, `gh auth
login`, because the repository is private), and `make demo` works. After `make features`, `make smoke` (a
one-minute training run on 128 utterances) checks the training path before `make train`.

Then talk to it yourself:

```bash
uv run python -m emo.demo --text "I said I was fine." --video me.mov    # any clip: voice and face come from it; the transcript is typed, there is no ASR
uv run python -m emo.serve                                               # JSON Lines on stdin/stdout, for a robot controller
```

## Demo: one input, both outputs

`make demo` replays MELD test dialogue 2 (Ross and Rachel carrying a couch up the stairs) utterance by
utterance, every speaker treated as the person talking to Pip. Each turn goes through the live path: the
transcript and the raw clip go in; while the clip's audio is decoded and read by WavLM, a second thread decodes
its frames and crops the largest face, which the expression model then reads; then the state is printed, then
the reply as it streams, then the timings. The same events are written as JSON Lines.

```
[1] Ross: Okay.
    state: neutral (0.99, high)  words=neutral  voice=neutral  face=anger  agree=False  MELD label: neutral  [1285 ms]
    Pip: Is that just for now, or do you need something?
    [first token 1678 ms, done 1977 ms]
[2] Rachel: Ross, didn't you say that there was an elevator in here?
    state: neutral (0.85, high)  words=neutral  voice=neutral  face=neutral  agree=True  MELD label: neutral  [401 ms]
    Pip: Yeah, but it's closed. What was that for?
    [first token 802 ms, done 1099 ms]
[3] Ross: Uhh, yes I did but there isn't. Okay, here we go.
    state: neutral (0.86, high)  words=neutral  voice=neutral  face=neutral  agree=True  MELD label: sadness  [354 ms]
    Pip: Got it, let's see what else we can do then.
    [first token 849 ms, done 1166 ms]
[4] Ross: Okay, go left. Left! Left!
    state: anger (0.43, low)  words=neutral  voice=anger  face=none  agree=False  MELD label: surprise  [156 ms]
    Pip: Left is the way to go, keep going.
    [first token 611 ms, done 849 ms]
[5] Rachel: Okay, y'know what? There is no more left, left!
    state: anger (0.70, high)  words=anger  voice=neutral  face=neutral  agree=False  MELD label: anger  [313 ms]
    Pip: I get it, you're frustrated. Left is the way to go then.
    [first token 798 ms, done 1182 ms]
[6] Ross: Oh okay, lift it straight up over your head!
    state: anger (0.74, high)  words=anger  voice=anger  face=neutral  agree=False  MELD label: anger  [259 ms]
    Pip: Alright, take your time. I'm here if you need to calm down.
    [first token 801 ms, done 1184 ms]
[7] Ross: Straight up over your head!
    state: anger (0.88, high)  words=anger  voice=anger  face=anger  agree=True  MELD label: anger  [217 ms]
    Pip: Got it, steady now. Just breathe and lift.
    [first token 707 ms, done 962 ms]
[8] Ross: You can do it!
    state: joy (0.73, high)  words=joy  voice=anger  face=none  agree=False  MELD label: joy  [151 ms]
    Pip: Haha, you're amazing! Just do it and I'll cheer for ya.
    [first token 660 ms, done 1055 ms]
[9] Ross: You can do it!
    state: joy (0.84, high)  words=joy  voice=anger  face=neutral  agree=False  MELD label: joy  [200 ms]
    Pip: Yeah, I'm all fired up! Let's rock it!
    [first token 662 ms, done 982 ms]
[10] Ross: Okay.
    state: neutral (0.98, high)  words=neutral  voice=neutral  face=neutral  agree=True  MELD label: neutral  [174 ms]
    Pip: Got it! How's your day been so far?
    [first token 557 ms, done 831 ms]
[11] Ross: You got it?
    state: neutral (0.66, medium)  words=neutral  voice=neutral  face=none  agree=True  MELD label: neutral  [152 ms]
    Pip: Always! What's something fun you're looking forward to?
    [first token 518 ms, done 814 ms]
```

(`face=none`: no face was found in that clip; turn 1 includes one-off warm-up.) The state matches MELD's label
on 9 of 11 turns, and the three inputs are weighed rather than copied. On turn
1 the face alone reads anger and the state stays neutral with the words and the voice; on turn 4 the words are
neutral but the voice is angry, and the state follows the voice at low certainty (MELD says surprise: loud, not
angry); on turns 8 and 9 an angry-sounding "You can do it!" is read as joy from the words; on turn 7 all three
agree and the state is anger at 0.88. The stage note carries the disagreement to Pip: on turn 8 it reads
`[seems happy; the voice sounds angry, you may gently notice the mismatch; share the good mood, ...]`.
Across the whole test set the three views disagree on 61 % of turns, and on those turns the fused state is
right more often (58.9 %) than any single input (words 54.1 %, voice 37.2 %, face 26.1 %).

## How it works

```
 clip ─┬─ audio track ─► 16 kHz ─► WavLM-base-plus, frozen ─► 13 layer means ─► learned mix ─► proj ─┬─ voice head (view)
       │                                                                                             │
       └─ frames ─► ffmpeg: 6 frames ─► YuNet: largest face ─► ViT (expressions), frozen ─► proj ─────┼─ face head (view)
                                                                                                     │
 transcript + previous turn ─► DistilRoBERTa, fine-tuned ─► <s> vector ──────────────────────────────┼─ words head (view)
                                                                                                     ▼
                                                                                  fused head ─► emotion, probs
 state {emotion, confidence, certainty, views, context} ─► one bracketed "stage note" ─► Qwen2.5-3B-Instruct, 4-bit ─► reply
```

| Component | Model | Parameters | Role |
|---|---|---|---|
| Voice encoder | `microsoft/wavlm-base-plus` | 94.4 M | frozen; every hidden layer averaged over time, 13 × 768 per utterance |
| Face detector | OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) | 0.05 M | largest face per frame, with a margin |
| Face encoder | `trpakov/vit-face-expression` | 85.8 M | frozen ViT-base fine-tuned on facial expressions; CLS vector averaged over the frames |
| Text encoder | `distilbert/distilroberta-base` | 82.1 M | fine-tuned on MELD, sees the utterance and the previous turn |
| Heads | own code ([emo/model.py](emo/model.py)) | 0.7 M | WavLM layer mix, voice and face projections, three view heads, fused MLP, temperature |
| Reply generator | `Qwen/Qwen2.5-3B-Instruct`, 4-bit through MLX (`mlx-community/Qwen2.5-3B-Instruct-4bit`) | 3 085.9 M | frozen; prompted only from the state; the persona's key/value cache is computed once |
| **Total on the inference path** | | **3 349.0 M** | cap is 6 000 M |

**Voice** ([emo/audio.py](emo/audio.py)): the clip's audio track (or a separate audio file) is decoded to 16 kHz
mono by ffmpeg, cropped to the first 10 s, and run through frozen WavLM-base-plus; each of its 13 hidden layers
is averaged over time, and the model learns how much each layer counts. The cached training features come from
a third-party 16 kHz FLAC extraction of MELD's audio (`ajyy/MELD_audio` on Hugging Face: the same waveform as
each clip's track at a fixed lower gain, which WavLM's input normalisation absorbs); live, the audio is decoded
from the clip's own track.

**Faces** ([emo/faces.py](emo/faces.py); the encoder is in [emo/vision.py](emo/vision.py)): `make faces`
streams MELD's 10.9 GB video archive in byte ranges without ever storing it: for every clip, ffmpeg decodes two
frames per second (first 10 s), six evenly spaced frames are kept, YuNet finds the largest face in each, and the
crops are saved as 224 px JPEGs. The live path runs exactly the same two functions on the clip (a still image is
one frame); frame decoding and face detection run in a thread beside the voice, then the ViT reads the crops. The largest face is a heuristic for "the speaker": sitcom
shots often show several people.

**Training** ([emo/train.py](emo/train.py)): the text encoder, the voice and face projections and all heads
train jointly on MELD train; WavLM and the expression ViT stay frozen (their features are cached once). Loss =
CE(fused) + 0.3 · CE of each view head. For 15 % of training samples the voice and, independently, the face are
hidden, so the fused head also works when a clip has no usable audio or no face. The best epoch is picked on
dev weighted-F1. Then each view head is refit alone on the frozen branch vectors (30 epochs, best dev epoch):
inside the joint run it only gets ~3 epochs of a decaying learning rate, and the face head had collapsed to
"neutral". This changes the `views`, never the prediction. Last, one temperature is fitted on dev so that
`confidence` means something. Any subset of `text,audio,vision` trains with the same script, which is how the
core text + vision track and every ablation below were trained.

**State** ([emo/session.py](emo/session.py)): the fused head gives `emotion` and calibrated `probs`; the three
view heads give `views` ("the words alone read X, the voice sounds Y, the face looks Z"); `certainty` buckets the
calibrated confidence (high ≥ 0.70, medium ≥ 0.45, else low). A `deque` of the last 4 messages (two exchanges:
the person and Pip alternate) is the dialogue memory: its last entry is the classifier's context, all of it
goes to the reply generator, and it is echoed in the state as `context`.

**Reply** ([emo/responder.py](emo/responder.py)): the remembered messages become chat turns, and the state is
rendered into one bracketed stage note in front of the person's words, e.g. `[seems happy; the voice sounds
angry, you may gently notice the mismatch; share the good mood, be warm and a little playful] You can do it!`
A view is only mentioned when it shows a different, non-neutral emotion (a face that shows nothing is not
evidence against an angry sentence), and at low certainty the note only says "hard to tell how they feel, ask
rather than assume". The seven `STYLE` lines are the whole character policy. Replies are cleaned (emojis
stripped) and validated (1–30 words, one line, nothing out of character such as "I can't assist with that");
otherwise a hand-written line for that emotion is used and the event says `"source": "fallback"`.

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

Measured numbers (`make bench`, 200 test utterances after 10 warm-up turns) are in the Results section. In
short: both configurations meet the median state and first-word times and the full-reply target, but not the
state and first-word targets at p95 on this laptop (deployed: 495 / 1 027 ms; core track: 344 / 876 ms).

## Interface: JSON Lines

`uv run python -m emo.serve` reads one request per line and streams the turn's events back, one per line.
A robot controller runs it as a subprocess. `python -m emo.demo` prints the same events for humans.

Request: `{"session_id": "s1", "text": "I know.", "video_path": "clip.mp4"}` or `{"session_id": "s1", "reset":
true}`. `video_path` is optional and supplies both the voice and the face; it can also be a still image or a
directory of face crops, and a separate `audio_path` (any audio file) overrides the clip's own sound. The events
of one real turn (the `ready` line is what `serve` prints at start-up; this was its first, cold turn):

```json
{"event": "ready", "device": "mps", "parameters_millions": {"classifier": 82.85, "responder": 3085.94, "face_encoder": 85.8, "face_detector": 0.05, "audio_encoder": 94.38, "total": 3349.03}}
{"event": "state", "session_id": "s1", "turn": 1, "text": "Straight up over your head!", "audio_seconds": 1.56, "faces": 3, "emotion": "anger", "confidence": 0.862, "certainty": "high", "probs": {"neutral": 0.023, "joy": 0.035, "sadness": 0.006, "anger": 0.862, "fear": 0.039, "disgust": 0.028, "surprise": 0.007}, "views": {"text": "anger", "audio": "anger", "vision": "anger", "agree": true}, "context": [], "latency_ms": {"state": 517}}
{"event": "token", "session_id": "s1", "turn": 1, "text": "I"}
{"event": "done", "session_id": "s1", "turn": 1, "response": "I get it, let's just talk about something else if you want.", "source": "llm", "prompt": "[seems angry; stay calm and steady, acknowledge the frustration, no jokes, no arguing] Straight up over your head!", "latency_ms": {"state": 517, "first_token": 864, "response": 1265}}
```
(from [runs/serve_example.jsonl](runs/serve_example.jsonl): the request was `{"session_id": "s1", "text": "Straight up over your head!", "video_path": "data/meld/videos/test/dia2_utt6.mp4"}`)

Field rules: `emotion` and every non-null view are one of the seven MELD labels; `views.vision` is `null` when
no face was found and `views.audio` when there is no usable audio (no audio track, or under 0.3 s); `agree` is `true`
when all present views name the same emotion and `null` when fewer than two are present; `faces` is the number
of face crops used (`null` without a video) and `audio_seconds` the length of audio used; `probs` has exactly seven keys and `confidence = max(probs)`, both
rounded to 3 decimals, so `probs` sums to 1 within rounding; `source` is `llm` or `fallback`; `context` holds
at most 4 earlier messages; a `done` event carries an `error` field when generation failed and the fallback
line was used; `error` events (bad request, a missing or undecodable video or audio file) carry a `message`
and a `hint` and never stop the server.

## Results

All numbers below come from `make eval`, `make bench` and `make responses` on the machine in the hardware
section, single seed, and are reproduced verbatim in [runs/](runs/). Dev was used for epoch selection and the
temperature; test logits (2 610 utterances) are computed once per run after all selection is done and are read
only by `evaluate.py`; nothing was tuned on them. One exception: the deployed variant was switched from text +
vision to text + vision + voice after the test comparison below was known. Dev slightly prefers text + audio
(0.611 against 0.606 weighted-F1, in the run summaries), so the deployed row's 0.640 is not an untouched
held-out estimate.

### Emotion classification (MELD test)

| model | weighted-F1 | macro-F1 | accuracy | neutral | joy | sadness | anger | fear | disgust | surprise |
|---|---|---|---|---|---|---|---|---|---|---|
| majority class (neutral) | 0.313 | 0.093 | 0.481 | 0.650 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| text only | 0.619 | 0.434 | 0.636 | 0.785 | 0.588 | 0.352 | 0.427 | 0.149 | 0.192 | 0.542 |
| audio only | 0.479 | 0.287 | 0.495 | 0.650 | 0.348 | 0.232 | 0.415 | 0.000 | 0.000 | 0.363 |
| vision only | 0.354 | 0.154 | 0.421 | 0.600 | 0.274 | 0.000 | 0.108 | 0.000 | 0.025 | 0.071 |
| text + vision (core track) | 0.621 | 0.420 | 0.631 | 0.784 | 0.598 | 0.363 | 0.470 | 0.059 | 0.154 | 0.512 |
| text + vision, no context | 0.619 | 0.418 | 0.631 | 0.787 | 0.594 | 0.332 | 0.446 | 0.088 | 0.136 | 0.543 |
| text + audio | 0.631 | 0.419 | 0.643 | 0.790 | 0.600 | 0.387 | 0.504 | 0.000 | 0.111 | 0.538 |
| **text + vision + audio (deployed)** | **0.640** | 0.443 | 0.654 | 0.797 | 0.610 | 0.371 | 0.490 | 0.092 | 0.162 | 0.576 |
| deployed model, words head only (`views.text`, all 2 610 turns) | 0.612 | 0.432 | 0.626 | 0.779 | 0.568 | 0.318 | 0.449 | 0.179 | 0.210 | 0.522 |
| deployed model, voice head only (`views.audio`, the 2554 turns with usable audio) | 0.475 | 0.280 | 0.517 | 0.682 | 0.289 | 0.237 | 0.359 | 0.069 | 0.000 | 0.323 |
| deployed model, face head only (`views.vision`, the 2539 turns with a face) | 0.361 | 0.155 | 0.447 | 0.625 | 0.249 | 0.043 | 0.084 | 0.000 | 0.029 | 0.058 |

For scale: the original MELD paper's bc-LSTM baselines reach roughly 0.56 (text), 0.39 (audio) and 0.60
(text + audio) weighted-F1; modern text-only transformers reach 0.62–0.66. On the M4, a single-modality run on
cached features trains in about 1 min, a run with text in 8–31 min (4 epochs; the longer ones shared the GPU).

**What does each input add?** Paired bootstrap over the test utterances (1 000 resamples), weighted-F1 minus
text only:

| model | difference | 95 % interval |
|---|---|---|
| text + vision (core track) | +0.001 | [−0.011, +0.014] |
| text + vision, no context | +0.000 | [−0.012, +0.013] |
| text + audio | +0.012 | [−0.001, +0.025] |
| **text + vision + audio (deployed)** | +0.020 | [+0.009, +0.033] |

The face adds nothing measurable to the words, the voice adds a little (its interval touches zero), and all
three together add +0.020, the only gain whose interval is clear of zero (this comparison was seen before the
deployed model was chosen, see above). The views show the three inputs being weighed: they disagree on 61.2 % of test
turns, and on those turns the fused state is right 58.9 % of the time, more than any single input
(words 54.1 %, voice 37.2 %, face 26.1 %).

**Why the face adds so little on its own.** The face signal is real but weak, much weaker than the words:
- Alone the face reaches 0.354 weighted-F1 (vision only) and 0.361 as the deployed model's face view,
  against 0.313 for always answering neutral; its best non-neutral class is joy (F1 0.27 and 0.25).
- The expression model's own 7-way classifier, used zero-shot and mapped to MELD labels
  (`python -m emo.vision --zero-shot`, [runs/vision_zero_shot_dev.json](runs/vision_zero_shot_dev.json)),
  scores 0.243 weighted-F1 on the 1,083 dev clips with a face, below the majority class (0.253); the
  face reads "happy" in 52 % of joy utterances but also in 29 % of neutral ones.
- MELD is a multi-camera sitcom: the largest face is not always the speaker, reaction shots are common, and the
  label describes the utterance, not a frame. The frozen encoder was trained on FER2013 stills.

What would likely help, and was not done: active-speaker detection (lip motion synchronised with the audio),
fine-tuning the expression encoder on MELD faces, and a temporal model over frames instead of a mean.

**Confidence means something.** After temperature scaling (T = 1.34), the certainty buckets on test are:

| certainty | share of test turns | accuracy |
|---|---|---|
| high (≥ 0.70) | 46.4 % | 82.6 % |
| medium (≥ 0.45) | 33.3 % | 58.4 % |
| low | 20.3 % | 37.7 % |

**Observations.** Feeding the previous utterance to the text encoder changed nothing on test (text + vision
0.621 with it, 0.619 without). Fear and disgust (2.6 % of MELD each) stay below 0.25 F1 for every model; the
confusion matrix in [runs/report.md](runs/report.md) shows both mostly collapsing into neutral and anger. The
learned WavLM layer mix stayed almost uniform (0.073–0.082 per layer), so the voice branch effectively
uses the mean of all 13 layers.

### Replies

`make responses`: 105 dev utterances (15 per true emotion) with usable audio and a face, each run as a
single turn through the full deployed pipeline (three-modality state, Qwen2.5-3B 4-bit), then again with greedy
decoding, and once more with only the emotion fields of the note changed.

| check | result |
|---|---|
| replies that fail validation and fall back to the hand-written line | 1 of 105 |
| mean reply length | 9.0 words (limit 30) |
| greedy reply changes when only the emotion in the note changes | 98.6 % of 73 turns |

The last row is the grounding check. It counts only turns whose note names the emotion: on the
31 low-certainty turns the note is "hard to tell how they feel" whatever the emotion, so the prompt does
not change and neither can the reply. What it looks like, from [runs/responses.md](runs/responses.md):

| person said | state | Pip (greedy) | if the state were... | Pip would say instead |
|---|---|---|---|---|
| Oh my God, he's lost it. He's totally lost it. | sadness | I know, it must be really tough. | anger | I know, it sounds scary. Let's try to stay calm though. |
| This is disgusting! | disgust | I see, let's move on then. | surprise | Yuck! What did you eat that made it so bad? |
| Or! Or, we could go to the bank, close our accounts and cut them off at the source. | anger | That sounds extreme. Let's talk about something else then. | fear | Sounds scary, but let's find another solution. |
| Ohh, you guys, remember that cute client I told you about? I bit him. | surprise (low certainty) | Oh no! That must have been tough. How did he react? | neutral | *(same prompt, same reply; not counted)* |

Quality, honestly: replies are short, in character and on tone, but the state is only as good as the
classifier ("I bit him" is labelled neutral in MELD and read as surprise), and a 3 B model still produces the
occasional non sequitur. Emojis are stripped before validation (they cannot be spoken).

### Reply model size: 0.5 B vs 1.5 B vs 3 B

The reply model is swappable without code changes: `EMO_RESPONSE_MODEL=Qwen/Qwen2.5-1.5B-Instruct make bench`
(any HuggingFace chat model id; an `mlx-community/...` id selects the MLX backend). Four configurations were
run through the same `make responses` and `make bench` with the same text + audio classifier (so they see the
same states) on the same machine; raw outputs are in [runs/reply_model_comparison/](runs/reply_model_comparison/).

| reply model | params | blind pairwise judging* | fallback rate | mean words | first `token` p50 / p95 | `done` p50 / p95 | decode | GPU memory |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct, bf16 | 494 M | – | 13.3 % | 13.3 | 866 / 1 047 ms | 1 330 / 1 904 ms | 35 tok/s | 2.9 GB |
| Qwen2.5-1.5B-Instruct, bf16 | 1 544 M | 66 : 32 (5 ties) vs 0.5B | 16.2 % | 15.0 | 672 / 779 ms | 1 398 / 2 027 ms | 24 tok/s | 4.4 GB |
| Qwen2.5-3B-Instruct, bf16 | 3 086 M | 60 : 42 (1 tie) vs 1.5B; 83 : 20 vs 0.5B | 1.9 % | 9.7 | 2 858 / 4 225 ms | 4 132 / 7 448 ms | 8 tok/s | 7.3 GB |
| Qwen2.5-3B-Instruct, 4-bit (MLX) | 3 086 M | 47 : 48 (8 ties) vs bf16 3B; 78 : 21 (4 ties) vs 0.5B | 0.0 % | 9.0 | 654 / 772 ms | 920 / 1 117 ms | 45 tok/s | 2.9 GB (torch) + 2.2 GB (MLX) |

\* The greedy replies of two models to the same 103 distinct dev utterances (same state, same note) were shown to three
independent Claude judges per pair, order balanced, with the rubric "in character as a robot friend, fits how
they sound, specific, natural"; majority vote, wins : wins. This is an external LLM judge, not a human study,
and Claude is also the tool that helped write this repository; the raw votes are in
[runs/reply_model_comparison/](runs/reply_model_comparison/).

None of these rows uses the persona cache described below. The bf16 3B row was measured while the machine was
swapping: 7.3 GB of weights do not sit comfortably next to everything else in 16 GB.

**Decision.** The deployed reply model is Qwen2.5-3B-Instruct in 4-bit through MLX: judged as good as the
bf16 model (47 : 48), clearly better than 0.5B (78 : 21), and fast: its own share of the time to the first word
(median first token minus state) is 0.47 s with the core classifier and 0.50 s in the deployed configuration.
bf16 3B misses the budget on this machine; 1.5B fits but loses to 3B 42 : 60. Off Apple silicon the same model runs in bf16 through transformers, on a CUDA
GPU when there is one (`APPLE_SILICON` in [emo/config.py](emo/config.py)); the persona cache is only
implemented for MLX.

### Latency and memory

`make bench`: the first 200 test utterances with a face and usable audio, in dialogue order, through `Session`,
after 10 warm-up turns, models warm; both configurations run on the same turns, one after the other. Every
measured turn takes the **live path**: the real MELD clip is handed over, its audio track and frames are decoded
(ffmpeg), faces are detected (YuNet), and the encoders the configuration uses (DistilRoBERTa, the face ViT,
WavLM) read them, exactly as for `--video`. (The raw clips of 30 test dialogues are kept for this; the rest of
the dataset keeps only crops.) Times are measured from the moment the utterance is handed over; `first token`
is when the first word is released.

| configuration | `state` p50 / p95 | first `token` p50 / p95 | `done` p50 / p95 | decode | fallback | peak RSS | GPU memory |
|---|---|---|---|---|---|---|---|
| target (p95) | 300 ms | 800 ms | 2 500 ms | | | | |
| **deployed: text + vision + audio** ([bench_mps.json](runs/bench_mps.json)) | 290 / 495 ms | 788 / 1 027 ms | **1 091 / 1 423 ms** | 42 tok/s | 0.5 % | 1.05 GB | 3.98 GB (torch) + 2.06 GB (MLX) |
| core track: text + vision ([bench_mps_text-vision.json](runs/bench_mps_text-vision.json)) | 246 / 344 ms | 716 / 876 ms | **1 028 / 1 253 ms** | 43 tok/s | 3.0 % | 2.21 GB | 2.86 GB (torch) + 2.06 GB (MLX) |

**Both configurations are on time at the median and for the whole reply, and both miss the state and
first-word targets at p95.** The core track met all three targets in an earlier run (release v0.2: state
245 ms, first word 664 ms at p95, on a slightly different set of turns), so on this laptop the p95 moves by about
100 ms between runs; this run had background indexing going on. Treat the targets as met at the median and
missed at p95.

The deployed model costs more than the core track because the voice adds a WavLM pass to the face's ViT pass,
and both encode the whole clip after it is handed over, so the state's cost grows with the length of the
utterance; the reply model's own share of the time to the first word (median first token minus state) is
0.50 s here and 0.47 s for the core track. Even clips under 3 s miss at p95 (state 324 ms, first word
866 ms), and the miss grows with length:

| deployed, by clip length | turns | `state` p50 / p95 | first `token` p95 |
|---|---|---|---|
| under 3 s | 125 | 244 / 324 ms | 866 ms |
| 3 to 6 s | 55 | 368 / 423 ms | 1 009 ms |
| 6 s and longer | 20 | 496 / 569 ms | 1 167 ms |

What is already done: frame decoding and face detection run in a thread beside the voice, live decoding uses
every core, and the persona's key/value cache is computed once, so each turn prefills only the conversation
(~70 instead of ~200 prompt tokens). What did not help, in quick tests: fp16 encoders, and launching the two
encoders from separate threads (the GPU itself is the bottleneck). What would fix it: encoding frames and audio
while the person is still speaking, so only the last fraction of a second is left after the handoff; this
prototype receives a finished clip, which is the worst case. `--run text-vision` runs the core track in `demo`,
`serve` and `bench` (the last writes `runs/bench_<device>_text-vision.json`). Cold start (all models loaded,
one warm-up turn) was 9.3 s for the deployed model. A CPU-only run was not benchmarked: the 3B reply model
needs a GPU (Apple GPU through MLX, or CUDA through transformers).

## Decisions and trade-offs

| Decision | Choice | Why | Not taken |
|---|---|---|---|
| Track | text + vision as the primary track, deployed with the voice as a third input (the brief's optional extension) | a character robot has a camera and a microphone on the person; on the MELD test set only all three inputs together beat text with an interval clear of zero (+0.020), a result seen before this choice was made | deploying the core track (faster, but adds nothing measurable over text) |
| Video data | stream the 10.9 GB MELD video archive once, in byte ranges, keep only 6 face crops per clip (~0.5 GB) and the raw clips of 31 test dialogues for the live-path benchmark | the archive does not fit on this laptop next to the models; crops are all the model needs, and the live path produces the same crops from any video | storing the videos, pre-extracted MELD visual features (not reproducible from a camera) |
| Voice encoder | WavLM-base-plus, frozen, 13 time-averaged layers with a learned mix, fed the clip's own audio track | the standard SUPERB recipe for emotion; caching the features once makes every audio experiment cheap, and one recorded clip carries both voice and face | fine-tuning WavLM (hours per run, no cache), a separate microphone stream |
| Face detection | OpenCV YuNet, largest face per frame, 20 % margin | 53 k parameters, runs on the CPU inside OpenCV, no extra ML runtime | MTCNN / RetinaFace (heavier), active-speaker detection (needs audio-visual sync, out of scope) |
| Vision encoder | `trpakov/vit-face-expression`, frozen, CLS vector averaged over the frames | a ViT-base already fine-tuned on facial expressions (FER2013) transfers better than a generic image model; frozen features are cached once, so every vision experiment is cheap | fine-tuning the ViT (hours, no cache), CLIP / DINOv2 generic features, a temporal video model |
| Text encoder | DistilRoBERTa, fully fine-tuned | 6 layers: a run with text trains in 8–10 min on an idle M4, and text alone lands in the published text-only range | a public "emotion" DistilRoBERTa (it was trained on MELD, so the test set would leak) |
| Dialogue context | the previous utterance as a second text segment | it is exactly the memory the robot keeps; ablated by the `text-vision-nocontext` row | a dialogue-level model |
| Fusion | concat -> one hidden layer -> 7 classes, plus one head per modality (refit alone after training), 15 % modality dropout | one hidden layer can model "the words say X but the voice says Y"; the per-modality heads give the `views`; dropout keeps the fused head usable when a clip has no face or no usable audio | a single linear layer (purely additive), cross-attention |
| Confidence | one temperature fitted on dev; high / medium / low buckets | the reply prompt and a robot both gate on it, so it has to mean something; bucket accuracy is reported | raw softmax |
| Reply generator | Qwen2.5-3B-Instruct, frozen, 4-bit through MLX (bf16 through transformers off Apple silicon), prompted only from the state; persona key/value cache | blind judging prefers it to 0.5B 78 : 21 and finds 4-bit as good as bf16 (47 : 48); with the persona cache its share of the time to the first word is ~0.5 s | 0.5B (generic, 13 % fallbacks), bf16 3B on this laptop (first token p95 4.2 s), 1.5B, fine-tuning, few-shot prompts |
| ASR | none | the brief gives the transcript; whisper-tiny (38 M) fits the budget and plugs into `Session.step` in one line, but its word errors would need their own evaluation | whisper-tiny |
| Interface | in-process `Session` + JSON Lines on stdin/stdout, one clip per utterance | zero web dependencies, testable with a shell pipe, spawned by a robot controller; one file carries voice and face | HTTP / websocket (a ~30-line wrapper) |
| Evidence | single seed, paired bootstrap interval on every gain over text, ablations, disagreement analysis, blind judging of replies, latency by clip length | a small gain can hide inside noise; the interval says whether it is real | multi-seed runs (time), a human rating study |

## Hardware and observed resources

- Apple M4 (10 cores), 16 GB unified memory, macOS 26.2; Python 3.12, torch 2.14.0, transformers 5.17.0,
  mlx-lm 0.31.3, uv.
- Parameters on the inference path (counted at start-up, printed by `serve` and `bench`): classifier 82.85 M
  + voice encoder 94.38 M + face encoder 85.80 M + face detector 0.05 M + reply model 3 085.94 M
  = **3 349.0 M**, 56 % of the 6 B cap. The core track without the voice: 3 254.4 M.
- Disk: environment 1.2 GB; MELD audio + manifests 1.5 GB; face crops 0.54 GB; kept test clips 0.21 GB;
  cached features 0.03 GB (vision) + 0.27 GB (audio); weights 0.33 GB (DistilRoBERTa) + 0.34 GB (ViT) +
  1.75 GB (Qwen 3B 4-bit) + 0.38 GB (WavLM); one checkpoint 0.17 GB. About 7 GB in total; the
  10.9 GB video archive is streamed and never stored.
- Data preparation: face extraction ~20 min (network-bound, 8 CPU workers); feature caches ~24 min (vision) and ~18 min
  (voice) on the GPU.
- Training (GPU, from the run summaries): single-modality runs about 1 min; runs with text 8–31 min (the
  deployed one 9.6 min; the longest ones shared the GPU with other jobs); `make train` in total ~80 min.
- Inference (deployed): 1.05 GB resident memory, 3.98 GB (torch) + 2.06 GB (MLX) GPU memory, 9.3 s cold start;
  per-turn numbers above.

## Limitations and what was left out

**Intentionally left out** (each with its natural plug-in point): reinforcement learning; ASR (`Session.step`);
camera and microphone capture, voice-activity detection and barge-in (the caller's job: it hands over one
utterance with its video); an HTTP or websocket server (wrap `Session`); active-speaker detection and speaker
identity; streaming encoders that work while the person is speaking; fine-tuning the vision or audio encoders; class-weighted or focal losses; hyper-parameter search;
multiple seeds; fine-tuning the reply model; text-to-speech and face animation.

**Known limitations**
- MELD is a sitcom: multi-camera cuts, reaction shots, several people in frame, a laugh track and a studio mix. "The largest face" is often but not always the speaker, and a robot's camera and microphone will see and hear one person, closer, in worse conditions.
- The expression encoder was trained on FER2013 (small grey-scale faces, mostly posed or in-the-wild stills); it is used frozen, so its domain gap to MELD and to a robot's camera is not trained away.
- Six frames over the first 10 s summarise a face by its average, and the voice is averaged over time too; a fleeting expression or tone can vanish in the mean.
- Latency: both configurations meet the median and full-reply targets but miss the state and first-word targets at p95 on this laptop, and p95 varies by ~100 ms between runs; the clip is encoded after handoff, so longer utterances cost more.
- At deployment the previous turn is usually the robot's own reply, which MELD never contains; the `text-vision-nocontext` row bounds how much the text + vision model depends on context (the deployed model has no such ablation).
- The classifier is utterance-level; fear and disgust are rare in MELD and score low for every model.
- Numbers are single-seed and MPS kernels are not bit-deterministic; small differences between runs are noise.
- Reply quality is judged by an automatic counterfactual check and a blind LLM judge, not by people.
- When a reply fails validation, the fallback line replaces the tokens that were already streamed; consumers must treat `done.response` as final.

## Repository layout

```
emo/config.py          paths, labels, model ids, hyperparameters, thresholds, latency targets (single source of truth)
emo/data.py            download, manifests (previous-turn context, clip and face joins), Dataset + collate
emo/faces.py           frames from a video, YuNet face crops; streams the MELD video archive into crops (torch-free)
emo/vision.py          frozen facial-expression ViT, feature cache
emo/audio.py           audio from any file (the clip's own track), frozen WavLM encoder, feature cache
emo/model.py           EmotionModel: any subset of text / audio / vision branches, per-modality heads, fused head, temperature
emo/train.py           training loop, dev selection, view-head refit, temperature fit, logits + summary per run
emo/evaluate.py        test tables, bootstrap CI for the gain over text, disagreement, calibration -> runs/report.md
emo/responder.py       persona, stage note, streamed generation (transformers, or MLX for 4-bit models), validation + fallback
emo/session.py         Classifier (one utterance -> state fields) and Session (memory + event stream)
emo/demo.py            replay a MELD dialogue or your own clip; human-readable output
emo/serve.py           JSON Lines over stdin/stdout
emo/bench.py           latency, memory, parameter budget -> runs/bench_<device>.json
emo/eval_responses.py  reply checks on dev: fallback rate, length, counterfactual sensitivity
tests/                 unit tests with stubs (manifests, clip names and stills, audio from video, model heads and view refit, session contract)
runs/                  committed results (json / md, demo traces, reply-model comparison); checkpoints and logits stay local
```

## Use of AI tools

The design, code and this write-up were produced with Claude (Claude Code) as a pair programmer; every
decision, number and limitation above was checked by running the code on the machine described in the
hardware section. Claude was also used as the blind judge in the reply-model comparison, which is stated
there. Pretrained components are credited in the components table; everything under `emo/` is original to
this repository.
