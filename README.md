# MELD emotion robot

A small, local, real-time prototype for an emotion-aware character robot. One spoken utterance goes in
(the **video of the person's face** and the **transcript**); two things come out, in order:

1. a **structured state** with a MELD emotion category, calibrated confidence, and what the words and the
   face each suggested, and
2. a **short in-character reply** ("Pip"), streamed token by token and grounded only in that state.

Primary track: **text + vision**. Optional extension: the **voice** as a third input (built and measured,
see *Adding the voice*). Dataset: [MELD](https://affective-meld.github.io/). Everything runs on a laptop
with **3.25 B parameters** in total (cap: 6 B). No remote calls.

```
[6] Joey: Damn you 15s!
    state: anger (0.58, medium)  words=anger  face=joy  agree=False  MELD label: anger  [162 ms]
    Pip: I see, let's try to stay calm. What’s up?
    [first token 567 ms, done 871 ms]
```
*(one turn of `make demo`: MELD test dialogue 85, the real clip decoded and read live on an Apple M4; the full trace is in [runs/demo_dialogue85.jsonl](runs/demo_dialogue85.jsonl))*

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg and curl (video decoding and the
streamed download), ~8 GB of disk, a 16 GB machine. Everything was measured on an Apple M4 laptop (macOS). On
other platforms the same code runs with the reply model in bf16 through transformers: real-time needs a CUDA
GPU there.

```bash
make setup      # environment (torch, transformers, opencv, ...)
make test       # unit tests: no data or trained model needed (downloads the 330 MB text encoder once)
make data       # MELD CSVs + 16 kHz audio (1.5 GB) and the manifests
make faces      # stream the 10.9 GB MELD videos once, keep only face crops (~0.5 GB on disk, ~20 min)
make features   # cache frozen facial-expression features once, ~25 min on an M4
make train      # deployed model + three ablations, ~30 min on an M4
make eval       # test-set tables -> runs/report.md
make demo       # replay MELD test dialogue 85 through state + reply (trace -> runs/demo_dialogue85.jsonl)
make bench      # latency, memory, parameter budget -> runs/bench_<device>.json
make responses  # reply checks on dev utterances -> runs/responses.md
```

To try the demo without training: `make data` and `make faces`, then `make checkpoint` downloads the deployed
model (165 MB) from the GitHub release into `runs/text-vision/` (it uses the GitHub CLI, `gh auth login`,
because the repository is private), and `make demo` works. `make smoke` (a one-minute training run on 128
utterances) checks the training path before the real runs. The voice extension has its own targets,
`make features-audio` and `make train-audio`.

Then talk to it yourself:

```bash
uv run python -m emo.demo --text "I said I was fine." --video me.mov    # any video or image; the transcript is typed, there is no ASR
uv run python -m emo.serve                                               # JSON Lines on stdin/stdout, for a robot controller
```

## Demo: one input, both outputs

`make demo` replays MELD test dialogue 85 (Joey's hernia) utterance by utterance, every speaker treated as the
person talking to Pip. Each turn goes through the live path: the transcript and the raw clip go in, the clip is
decoded, the largest face is cropped from six frames and read by the expression model, the state is printed,
then the reply as it streams, then the timings. The same events are written as JSON Lines.

```
[1] Joey: But um, I don’t think it’s anything serious.
    state: neutral (0.58, medium)  words=neutral  face=neutral  agree=True  MELD label: neutral  [327 ms]
    Pip: Got you, that's good to know then.
    [first token 534 ms, done 738 ms]
[2] Chandler: This sounds like a hernia. You have to—you-you—Go to the doctor!
    state: anger (0.30, low)  words=surprise  face=neutral  agree=False  MELD label: surprise  [196 ms]
    Pip: Got it, you should definitely see a doctor about this.
    [first token 469 ms, done 731 ms]
[3] Joey: No way!
    state: anger (0.44, low)  words=surprise  face=surprise  agree=True  MELD label: anger  [127 ms]
    Pip: Alright, you decide. Let me know if you change your mind.
    [first token 470 ms, done 774 ms]
[4] Joey: ‘Kay look, if I have to go to the doctor for anything it’s gonna be for this thing sticking out of my stomach!
    state: anger (0.53, medium)  words=anger  face=neutral  agree=False  MELD label: anger  [210 ms]
    Pip: I understand, but sometimes it's necessary to be seen. Let's keep calm.
    [first token 614 ms, done 981 ms]
[5] Joey: Why did I have to start working out again?
    state: surprise (0.38, low)  words=surprise  face=neutral  agree=False  MELD label: disgust  [193 ms]
    Pip: Not sure yet, but working out can help in lots of ways.
    [first token 531 ms, done 807 ms]
[6] Joey: Damn you 15s!
    state: anger (0.58, medium)  words=anger  face=joy  agree=False  MELD label: anger  [162 ms]
    Pip: I see, let's try to stay calm. What’s up?
    [first token 567 ms, done 871 ms]
```

The state matches MELD's label on 4 of 6 turns. The face view is weak (see Results) but not silent: on
turn 3 the words and the face both read surprise while the fused state says anger, and on turn 6 the face reads
happy under angry words, so the note tells Pip that the face looks happy and it may gently notice the mismatch.
Turns 2, 3 and 5 are low-certainty, so their note only says "hard to tell how they feel, ask rather than
assume".

## How it works

```
 video (or image) ─► ffmpeg: up to 6 frames ─► YuNet: largest face ─► ViT (facial expressions), frozen ─► CLS mean ─► proj ─┐
                                                                                                        vision head (view) ┘  │
 text + previous turn ─► DistilRoBERTa, fine-tuned ─► <s> vector ───────────────────────────────────────────────────────────┼─► fused head ─► emotion, probs
                                                                                                          text head (view) ┘  │
 (optional) audio 16 kHz ─► WavLM-base-plus, frozen ─► 13 layer means ─► learned mix ─► proj ─► audio head (view) ─────────┘  ▼
 state {emotion, confidence, certainty, views, context} ─► one bracketed "stage note" ─► Qwen2.5-3B-Instruct, 4-bit ─► reply
```

| Component | Model | Parameters | Role |
|---|---|---|---|
| Face detector | OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) | 0.05 M | largest face per frame, with a margin |
| Vision encoder | `trpakov/vit-face-expression` | 85.8 M | frozen ViT-base fine-tuned on facial expressions; CLS vector averaged over the frames |
| Text encoder | `distilbert/distilroberta-base` | 82.1 M | fine-tuned on MELD, sees the utterance and the previous turn |
| Heads | own code ([emo/model.py](emo/model.py)) | 0.5 M | vision projection, text head, vision head, fused MLP, temperature |
| Reply generator | `Qwen/Qwen2.5-3B-Instruct`, 4-bit through MLX (`mlx-community/Qwen2.5-3B-Instruct-4bit`) | 3 085.9 M | frozen; prompted only from the state; the persona's key/value cache is computed once |
| **Total on the inference path** | | **3 254.4 M** | cap is 6 000 M |
| *(extension)* Audio encoder | `microsoft/wavlm-base-plus` | 94.4 M | frozen; only in the three-modality model |

**Faces** ([emo/faces.py](emo/faces.py); the encoder is in [emo/vision.py](emo/vision.py)): MELD ships each
utterance as a short video clip. `make faces` streams the 10.9 GB archive in byte ranges without ever storing
it: for every clip, ffmpeg decodes two frames per second (first 10 s), six evenly spaced frames are kept, YuNet
finds the largest face in each, and the crops are saved as 224 px JPEGs. The live path (`--video`) runs exactly the same two functions on your file (a still
image is one frame). The largest face is a heuristic for "the speaker": sitcom shots often show several people.

**Training** ([emo/train.py](emo/train.py)): the text encoder, the vision projection and all heads train
jointly on MELD train; the expression ViT stays frozen (its features are cached once). Loss = CE(fused) +
0.3·CE(text head) + 0.3·CE(vision head). For 15 % of training samples the face features are hidden, so the
fused head also learns to work from text alone (that is also what happens live when no face is found). The
best epoch is picked on dev weighted-F1. Then each per-modality head is refit alone on the frozen branch
vectors (30 epochs, best dev epoch): inside the joint run it only gets ~3 epochs of a decaying learning rate
and the vision head collapsed to "neutral". This changes the `views`, never the prediction. Last, one
temperature is fitted on dev so that `confidence` means something. Any subset of `text,audio,vision` trains
with the same script.

**State** ([emo/session.py](emo/session.py)): the fused head gives `emotion` and calibrated `probs`; the text
and vision heads give `views` ("the words alone read X, the face looks Y"); `certainty` buckets the calibrated
confidence (high ≥ 0.70, medium ≥ 0.45, else low). A `deque` of the last 4 messages (two exchanges: the
person and Pip alternate) is the dialogue memory: its last entry is the classifier's context, all of it goes to
the reply generator, and it is echoed in the state as `context`.

**Reply** ([emo/responder.py](emo/responder.py)): the remembered messages become chat turns, and the state is
rendered into one bracketed stage note in front of the person's words, e.g. `[seems angry; the words alone read
surprised, you may gently notice the mismatch; stay calm and steady, acknowledge the frustration, no jokes, no
arguing] No way!` A view is only mentioned when it shows a different, non-neutral emotion (a face that shows
nothing is not evidence against an angry sentence), and at low certainty the note only says "hard to tell how
they feel, ask rather than assume". The seven `STYLE` lines are the whole character policy. Replies are
cleaned (emojis stripped) and validated (1–30 words, one line, nothing out of character such as "I can't assist
with that"); otherwise a hand-written line for that emotion is used and the event says `"source": "fallback"`.

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

Request: `{"session_id": "s1", "text": "I know.", "video_path": "clip.mp4"}` (`video_path`: any video or
image, or a directory of face crops, optional; `audio_path`: 16 kHz mono WAV/FLAC, only used by a model with an
audio branch) or `{"session_id": "s1", "reset": true}`. The events of one real turn (the `ready` line is what
`serve` prints at start-up):

```json
{"event": "ready", "device": "mps", "parameters_millions": {"classifier": 82.59, "responder": 3085.94, "face_encoder": 85.8, "face_detector": 0.05, "total": 3254.38}}
{"event": "state", "session_id": "s1", "turn": 1, "text": "No way!", "audio_seconds": null, "faces": 3, "emotion": "anger", "confidence": 0.467, "certainty": "medium", "probs": {"neutral": 0.054, "joy": 0.027, "sadness": 0.058, "anger": 0.467, "fear": 0.053, "disgust": 0.073, "surprise": 0.267}, "views": {"text": "surprise", "vision": "surprise", "agree": true}, "context": [], "latency_ms": {"state": 212}}
{"event": "token", "session_id": "s1", "turn": 1, "text": "I"}
{"event": "done", "session_id": "s1", "turn": 1, "response": "I get it, let's talk when you're ready.", "source": "llm", "prompt": "[seems angry; the words alone read surprised, the face looks surprised, you may gently notice the mismatch; stay calm and steady, acknowledge the frustration, no jokes, no arguing] No way!", "latency_ms": {"state": 212, "first_token": 424, "response": 657}}
```
(from [runs/serve_example.jsonl](runs/serve_example.jsonl): the request was `{"session_id": "s1", "text": "No way!", "video_path": "data/meld/videos/test/dia85_utt2.mp4"}`)

Field rules: `emotion` and every non-null view are one of the seven MELD labels; `views.vision` is `null` when
no face was found (and `views.audio` without usable audio, for a model with an audio branch); `agree` is `true`
when all present views name the same emotion and `null` when fewer than two are present; `faces` is the number
of face crops used (`null` without a video); `probs` has exactly seven keys and `confidence = max(probs)`, both
rounded to 3 decimals, so `probs` sums to 1 within rounding; `source` is `llm` or `fallback`; `context` holds
at most 4 earlier messages; a `done` event carries an `error` field when generation failed and the fallback
line was used; `error` events (bad request, missing video or audio file, unreadable audio) carry a `message`
and a `hint` and never stop the server.

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
| vision only | 0.354 | 0.154 | 0.421 | 0.600 | 0.274 | 0.000 | 0.108 | 0.000 | 0.025 | 0.071 |
| **text + vision (deployed)** | **0.621** | 0.420 | 0.631 | 0.784 | 0.598 | 0.363 | 0.470 | 0.059 | 0.154 | 0.512 |
| text + vision, no context | 0.619 | 0.418 | 0.631 | 0.787 | 0.594 | 0.332 | 0.446 | 0.088 | 0.136 | 0.543 |
| deployed model, text head only (`views.text`, 2 610 turns) | 0.623 | 0.441 | 0.635 | 0.781 | 0.582 | 0.332 | 0.474 | 0.125 | 0.243 | 0.549 |
| deployed model, vision head only (`views.vision`, the 2 539 turns with a face) | 0.357 | 0.152 | 0.454 | 0.628 | 0.217 | 0.036 | 0.090 | 0.040 | 0.000 | 0.050 |
| *extension:* audio only | 0.479 | 0.287 | 0.495 | 0.650 | 0.348 | 0.232 | 0.415 | 0.000 | 0.000 | 0.363 |
| *extension:* text + audio | 0.631 | 0.419 | 0.643 | 0.790 | 0.600 | 0.387 | 0.504 | 0.000 | 0.111 | 0.538 |
| *extension:* text + vision + audio | **0.640** | 0.443 | 0.654 | 0.797 | 0.610 | 0.371 | 0.490 | 0.092 | 0.162 | 0.576 |

For scale: the original MELD paper's bc-LSTM baselines reach roughly 0.56 (text), 0.39 (audio) and 0.60
(text + audio) weighted-F1; modern text-only transformers reach 0.62–0.66. On the M4, a vision-only run trains
in 1 min (frozen features, 30 epochs of a small head), a text + vision run in 8–10 min (4 epochs).

**Does vision add measurable value? Not with this encoder.** Paired bootstrap over the test utterances
(1 000 resamples), weighted-F1 minus text only:

| model | difference | 95 % interval |
|---|---|---|
| text + vision (deployed) | +0.001 | [−0.011, +0.014] |
| text + vision, no context | +0.000 | [−0.012, +0.013] |
| text + audio | +0.012 | [−0.001, +0.025] |
| text + vision + audio | **+0.020** | **[+0.009, +0.033]** |

Adding the face to the words changes nothing measurable; the only combination that beats text with an interval
clear of zero is all three modalities. The deployed model is still text + vision, because that is the track this
prototype was asked to build; switching to the three-modality model is one line (`DEPLOYED_RUN`), costs the
94 M-parameter WavLM and one pass over the audio per turn, and needs the utterance audio.

**Why the face adds so little here.** The face signal is real but weak, and much weaker than the words:
- On its own the face reaches 0.354 weighted-F1 (vision only) and 0.357 as the deployed model's refit vision
  view, against 0.313 for always answering neutral; its best class is joy (F1 0.27 and 0.22).
- The expression model's own 7-way classifier, used zero-shot and mapped to MELD labels
  (`python -m emo.vision --zero-shot`, [runs/vision_zero_shot_dev.json](runs/vision_zero_shot_dev.json)),
  scores 0.243 weighted-F1 on the 1 083 dev clips with a face, below the majority class (0.253); the face
  reads "happy" in 52 % of joy utterances but also in 29 % of neutral ones.
- The words and the face disagree on 49.5 % of test turns; on those turns the words are right 52.6 % of the
  time and the face 17.7 %, so the fused head learned to follow the words.
- MELD is a multi-camera sitcom: the largest face is not always the speaker, reaction shots are common, and the
  label describes the utterance, not a frame. The frozen encoder was trained on FER2013 stills.

What would likely help, and was not done: active-speaker detection (lip motion synchronised with the audio),
fine-tuning the expression encoder on MELD faces, and a temporal model over frames instead of a mean.

**Confidence means something.** After temperature scaling (T = 1.23), the certainty buckets on test are:

| certainty | share of test turns | accuracy |
|---|---|---|
| high (≥ 0.70) | 41.1 % | 82.9 % |
| medium (≥ 0.45) | 35.4 % | 58.3 % |
| low | 23.5 % | 35.6 % |

**Observations.** Feeding the previous utterance to the text encoder changed nothing on test (0.621 vs 0.619).
Fear and disgust (2.6 % of MELD each) stay below 0.25 F1 for every model; the confusion matrix in
[runs/report.md](runs/report.md) shows both mostly collapsing into neutral and anger.

### Replies

`make responses`: 105 dev utterances (15 per true emotion) with a face, each run as a single turn through the
full deployed pipeline (text + vision state, Qwen2.5-3B 4-bit), then again with greedy decoding, and once more
with only the emotion fields of the note changed.

| check | result |
|---|---|
| replies that fail validation and fall back to the hand-written line | 0 of 105 |
| mean reply length | 8.8 words (limit 30) |
| greedy reply changes when only the emotion in the note changes | 98.6 % of 72 turns |

The last row is the grounding check. It counts only turns whose note names the emotion: on the
32 low-certainty turns the note is "hard to tell how they feel" whatever the emotion, so the prompt
does not change and neither can the reply. What it looks like, from [runs/responses.md](runs/responses.md):

| person said | state | Pip (greedy) | if the state were... | Pip would say instead |
|---|---|---|---|---|
| Oh my God, he's lost it. He's totally lost it. | sadness | I know, it must be really tough. | anger | I know, it sounds scary. Let's try to stay calm though. |
| What? | neutral | Sounds good, what do you like to do in your free time? | joy | Just wanted to share, everything's okay! |
| Where?! | surprise | Where what happened? Tell me! | neutral | Anywhere you like, just tell me! |
| Ohh, you guys, remember that cute client I told you about? I bit him. | disgust (low certainty) | Oh no! That must have been tough. How did he react? | surprise | *(same prompt, same reply; not counted)* |

Quality, honestly: replies are short, in character and on tone, but the state is only as good as the
classifier (the "I bit him" line above is labelled neutral in MELD and read as disgust; "What?" is labelled
surprise and read as neutral), and a 3 B model still produces the occasional non sequitur. Emojis are stripped
before validation (they cannot be spoken).

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
bf16 model (47 : 48), clearly better than 0.5B (78 : 21), no fallbacks in 105 turns, and in the deployed
configuration it meets every latency target (next section). bf16 3B misses the budget on this machine; 1.5B
fits but loses to 3B 42 : 60. Off Apple silicon the same model runs in bf16 through transformers, on a CUDA
GPU when there is one (`APPLE_SILICON` in [emo/config.py](emo/config.py)); the persona cache is only
implemented for MLX.

### Latency and memory

`make bench`: 200 consecutive test utterances in dialogue order through `Session`, after 10 warm-up turns,
models warm, nothing else running. Every measured turn takes the **live video path**: the real MELD clip is
decoded (ffmpeg), faces are detected (YuNet) and embedded (ViT), exactly as for `--video` (the raw clips of 31
test dialogues are kept for this; the rest of the dataset keeps only crops). Times are measured from the
moment the utterance is handed over; `first token` is when the first word is released.

| | `state` p50 / p95 | first `token` p50 / p95 | `done` p50 / p95 | decode | fallback | peak RSS | GPU memory |
|---|---|---|---|---|---|---|---|
| target (p95) | 300 ms | 800 ms | 2 500 ms | | | | |
| M4, text + vision, Qwen2.5-3B 4-bit | **176 / 245 ms** | **523 / 664 ms** | **780 / 1 009 ms** | 46 tok/s | 1.5 % | 2.66 GB | 2.86 GB (torch) + 2.07 GB (MLX) |

All three targets are met ([runs/bench_mps.json](runs/bench_mps.json)). Two measures got there: live frame
decoding uses every core (it had inherited the one-thread setting of the dataset workers, which run eight at a
time), and the persona's key/value cache is computed once, so each turn prefills only the conversation (~70
instead of ~200 prompt tokens). The video path (decode, detect, embed) is part of every `state` number above; a
robot that processes frames while the person is still speaking would take most of it off the critical path,
so these numbers are the worst case. Cold start (all models loaded, one warm-up turn) is 2.6 s. A
CPU-only run was not benchmarked for this configuration: the 3B reply model needs a GPU (Apple GPU through MLX,
or CUDA through transformers).

### Adding the voice (optional extension)

The audio branch was the first version of this prototype (release v0.1, text + audio) and is kept as the
optional third input: `make features-audio` caches frozen WavLM-base-plus features (13 time-averaged hidden
layers with a learned mix), and `make train-audio` trains `audio`, `text-audio` and `text-audio-vision` with the
same recipe. Their rows are in the tables above: audio alone 0.479, text + audio 0.631 (+0.012 over text,
interval [−0.001, +0.025]), and all three modalities 0.640, the only combination whose gain over text is clear
of zero (+0.020, [+0.009, +0.033]).

To deploy it, set `DEPLOYED_RUN = "text-audio-vision"` in [emo/config.py](emo/config.py) and send an
`audio_path` (16 kHz mono) with each request. It adds WavLM (94.4 M parameters) and one pass over the audio per
turn; that configuration was not benchmarked here.

## Decisions and trade-offs

| Decision | Choice | Why | Not taken |
|---|---|---|---|
| Track | text + vision (face), voice as an optional third input | a character robot has a camera on the person; the face shows what the words hide. The voice branch was built first and is kept as the measured extension | text + audio only |
| Video data | stream the 10.9 GB MELD video archive once, in byte ranges, keep only 6 face crops per clip (~0.5 GB) and the raw clips of 31 test dialogues for the live-path benchmark | the archive does not fit on this laptop next to the models; crops are all the model needs, and the live path produces the same crops from any video | storing the videos, pre-extracted MELD visual features (not reproducible from a camera) |
| Face detection | OpenCV YuNet, largest face per frame, 20 % margin | 53 k parameters, runs on the CPU inside OpenCV, no extra ML runtime | MTCNN / RetinaFace (heavier), active-speaker detection (needs audio-visual sync, out of scope) |
| Vision encoder | `trpakov/vit-face-expression`, frozen, CLS vector averaged over the frames | a ViT-base already fine-tuned on facial expressions (FER2013) transfers better than a generic image model; frozen features are cached once, so every vision experiment is cheap | fine-tuning the ViT (hours, no cache), CLIP / DINOv2 generic features, a temporal video model |
| Text encoder | DistilRoBERTa, fully fine-tuned | 6 layers: a text + vision run trains in 8–10 min here, and text alone lands in the published text-only range | a public "emotion" DistilRoBERTa (it was trained on MELD, so the test set would leak) |
| Dialogue context | the previous utterance as a second text segment | it is exactly the memory the robot keeps; ablated by the `text-vision-nocontext` row | a dialogue-level model |
| Fusion | concat -> one hidden layer -> 7 classes, plus one head per modality, 15 % modality dropout | one hidden layer can model "words say X but the face says Y"; the per-modality heads give the `views` for free; dropout keeps the fused head usable when no face is found | a single linear layer (purely additive), cross-attention |
| Confidence | one temperature fitted on dev; high / medium / low buckets | the reply prompt and a robot both gate on it, so it has to mean something; bucket accuracy is reported | raw softmax |
| Reply generator | Qwen2.5-3B-Instruct, frozen, 4-bit through MLX (bf16 through transformers off Apple silicon), prompted only from the state; persona key/value cache | blind judging prefers it to 0.5B 78 : 21 and finds 4-bit as good as bf16 (47 : 48); with the persona cache it meets every latency target | 0.5B (generic, 13 % fallbacks), bf16 3B on this laptop (first token p95 4.2 s), 1.5B, fine-tuning, few-shot prompts |
| ASR | none | the brief gives the transcript; whisper-tiny (38 M) fits the budget and plugs into `Session.step` in one line, but its word errors would need their own evaluation | whisper-tiny |
| Interface | in-process `Session` + JSON Lines on stdin/stdout | zero web dependencies, testable with a shell pipe, spawned by a robot controller | HTTP / websocket (a ~30-line wrapper) |
| Evidence | single seed, paired bootstrap interval on the vision gain, ablations, disagreement analysis, blind judging of replies | a small gain can hide inside noise; the interval says whether it is real | multi-seed runs (time), a human rating study |

## Hardware and observed resources

- Apple M4 (10 cores), 16 GB unified memory, macOS 26.2; Python 3.12, torch 2.14.0, transformers 5.17.0,
  mlx-lm 0.31.3, uv.
- Parameters on the inference path (counted at start-up, printed by `serve` and `bench`): classifier 82.59 M
  + face encoder 85.80 M + face detector 0.05 M + reply model 3 085.94 M = **3 254.4 M**, 54 % of the 6 B cap
  (+94.4 M for WavLM in the three-modality extension).
- Disk: environment 1.2 GB; MELD audio + manifests 1.5 GB; face crops 0.54 GB; kept test clips 0.21 GB;
  cached features 0.03 GB (vision) + 0.27 GB (audio); weights 0.33 GB (DistilRoBERTa) + 0.34 GB (ViT) +
  1.75 GB (Qwen 3B 4-bit) + 0.38 GB (WavLM, extension); one checkpoint 0.17 GB. About 7 GB in total; the
  10.9 GB video archive is streamed and never stored.
- Data preparation: face extraction ~20 min (network-bound, 8 CPU workers); vision feature cache ~24 min on the GPU.
- Training (GPU): vision-only 1 min; text + vision 8–10 min per run; text only 31 min (it shared the GPU).
- Inference: 2.66 GB resident memory, 2.86 GB (torch) + 2.07 GB (MLX) GPU memory, 2.6 s cold start;
  per-turn numbers above.

## Limitations and what was left out

**Intentionally left out** (each with its natural plug-in point): reinforcement learning; ASR (`Session.step`);
camera and microphone capture, voice-activity detection and barge-in (the caller's job: it hands over one
utterance with its video); an HTTP or websocket server (wrap `Session`); active-speaker detection and speaker
identity; fine-tuning the vision or audio encoders; class-weighted or focal losses; hyper-parameter search;
multiple seeds; fine-tuning the reply model; text-to-speech and face animation.

**Known limitations**
- MELD is a sitcom: multi-camera cuts, reaction shots, several people in frame, studio lighting. "The largest face" is often but not always the speaker, and a robot's camera will see one person, closer, in worse light.
- The expression encoder was trained on FER2013 (small grey-scale faces, mostly posed or in-the-wild stills); it is used frozen, so its domain gap to MELD and to a robot's camera is not trained away.
- Six frames over the first 10 s summarise a face by its average; a fleeting expression can vanish in the mean.
- At deployment the previous turn is usually the robot's own reply, which MELD never contains; the `text-vision-nocontext` row bounds how much the model depends on context.
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
emo/audio.py           16 kHz loading, frozen WavLM encoder, feature cache (optional voice extension)
emo/model.py           EmotionModel: any subset of text / audio / vision branches, per-modality heads, fused head, temperature
emo/train.py           training loop, dev selection, view-head refit, temperature fit, logits + summary per run
emo/evaluate.py        test tables, bootstrap CI for the gain over text, disagreement, calibration -> runs/report.md
emo/responder.py       persona, stage note, streamed generation (transformers, or MLX for 4-bit models), validation + fallback
emo/session.py         Classifier (one utterance -> state fields) and Session (memory + event stream)
emo/demo.py            replay a MELD dialogue or your own clip; human-readable output
emo/serve.py           JSON Lines over stdin/stdout
emo/bench.py           latency, memory, parameter budget -> runs/bench_<device>.json
emo/eval_responses.py  reply checks on dev: fallback rate, length, counterfactual sensitivity
tests/                 unit tests with stubs (manifests, clip names and stills, model heads and view refit, session contract)
runs/                  committed results (json / md, demo traces, reply-model comparison); checkpoints and logits stay local
```

## Use of AI tools

The design, code and this write-up were produced with Claude (Claude Code) as a pair programmer; every
decision, number and limitation above was checked by running the code on the machine described in the
hardware section. Claude was also used as the blind judge in the reply-model comparison, which is stated
there. Pretrained components are credited in the components table; everything under `emo/` is original to
this repository.
