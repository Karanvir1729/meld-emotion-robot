# Every step of the prototype, in order. `uv run` creates/updates the environment on first use.
RUN = uv run python -m

setup:      ## install the environment (Python 3.12, torch, transformers, ...)
	uv sync

data:       ## download MELD CSVs + 16 kHz audio (1.5 GB) and build the manifests
	$(RUN) emo.data

features:   ## cache frozen WavLM features for every clip (~15 min on an M4)
	$(RUN) emo.audio

checkpoint: ## download the trained deployed model from the GitHub release instead of running features + train
	mkdir -p runs/both
	gh release download v0.1 --repo Karanvir1729/meld-emotion-robot --dir runs/both --clobber

smoke:      ## one-minute end-to-end training check on 128 utterances
	$(RUN) emo.train --modalities both --limit 128 --epochs 1 --name smoke

train:      ## the deployed model and its ablations (~1 h on an M4)
	$(RUN) emo.train --modalities text
	$(RUN) emo.train --modalities audio
	$(RUN) emo.train --modalities both --context 0
	$(RUN) emo.train --modalities both

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

.PHONY: setup data features checkpoint smoke train eval demo serve bench responses test
