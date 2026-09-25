# Every step of the prototype, in order. `uv run` creates/updates the environment on first use.
RUN = uv run python -m

setup:      ## install the environment (Python 3.12, torch, transformers, ...)
	uv sync

data:       ## download MELD CSVs + 16 kHz audio (1.5 GB) and build the manifests
	$(RUN) emo.data

faces:      ## stream the 10.9 GB MELD videos once and keep face crops (~0.5 GB, ~20 min)
	$(RUN) emo.faces

features:   ## cache frozen facial-expression features for every clip (~25 min on an M4)
	$(RUN) emo.vision

features-audio: ## cache frozen WavLM features (only for the optional audio extension, ~15 min)
	$(RUN) emo.audio

checkpoint: ## download the trained deployed model from the GitHub release instead of running faces + features + train
	mkdir -p runs/text-vision
	gh release download v0.2 --repo Karanvir1729/meld-emotion-robot --dir runs/text-vision --clobber

smoke:      ## one-minute end-to-end training check on 128 utterances
	$(RUN) emo.train --modalities text,vision --limit 128 --epochs 1 --name smoke

train:      ## the deployed model and its ablations (~30 min on an M4)
	$(RUN) emo.train --modalities text
	$(RUN) emo.train --modalities vision
	$(RUN) emo.train --modalities text,vision --context 0
	$(RUN) emo.train --modalities text,vision

train-audio: ## the optional audio extension: needs features-audio (~25 min)
	$(RUN) emo.train --modalities audio
	$(RUN) emo.train --modalities text,audio
	$(RUN) emo.train --modalities text,audio,vision

eval:       ## test-set tables -> runs/report.md
	$(RUN) emo.evaluate

demo:       ## replay one MELD dialogue through state + reply; the trace is committed under runs/
	$(RUN) emo.demo --split test --dialogue 85 --jsonl runs/demo_dialogue85.jsonl

serve:      ## JSON Lines over stdin/stdout for a robot controller
	$(RUN) emo.serve

bench:      ## latency, memory and parameter budget -> runs/bench_<device>.json
	$(RUN) emo.bench

responses:  ## reply checks on dev utterances -> runs/responses.md
	$(RUN) emo.eval_responses

test:       ## unit tests (no trained model needed)
	$(RUN) pytest -q

.PHONY: setup data faces features features-audio checkpoint smoke train train-audio eval demo serve bench responses test
