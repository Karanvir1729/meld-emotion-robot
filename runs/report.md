# Results on the MELD test split

## Classification

| model | weighted-F1 | macro-F1 | accuracy | neutral | joy | sadness | anger | fear | disgust | surprise |
|---|---|---|---|---|---|---|---|---|---|---|
| majority class (neutral) | 0.313 | 0.093 | 0.481 | 0.650 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| text only | 0.619 | 0.434 | 0.636 | 0.785 | 0.588 | 0.352 | 0.427 | 0.149 | 0.192 | 0.542 |
| audio only | 0.479 | 0.287 | 0.495 | 0.650 | 0.348 | 0.232 | 0.415 | 0.000 | 0.000 | 0.363 |
| text + audio, no context | 0.629 | 0.422 | 0.642 | 0.788 | 0.601 | 0.368 | 0.498 | 0.000 | 0.156 | 0.541 |
| text + audio (deployed) | 0.631 | 0.419 | 0.643 | 0.790 | 0.600 | 0.387 | 0.504 | 0.000 | 0.111 | 0.538 |
| deployed model, text head only (views.text) | 0.619 | 0.418 | 0.633 | 0.784 | 0.585 | 0.349 | 0.473 | 0.033 | 0.178 | 0.525 |
| deployed model, audio head only (views.audio) | 0.450 | 0.239 | 0.522 | 0.684 | 0.222 | 0.161 | 0.382 | 0.000 | 0.000 | 0.222 |

## Audio gain over text (paired bootstrap, 1000 resamples)

weighted-F1(both) - weighted-F1(text) = +0.0118, 95% CI [-0.0008, +0.0254]

## Turns where the text head and the audio head disagree

43.3% of test turns. Accuracy on them: fused 0.504, text head 0.483, audio head 0.227

## Learned WavLM layer weights (0 = CNN output, 12 = last transformer layer)

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weight | 0.077 | 0.073 | 0.073 | 0.075 | 0.075 | 0.077 | 0.080 | 0.079 | 0.081 | 0.080 | 0.077 | 0.076 | 0.077 |

## Certainty buckets (temperature 1.198)

| certainty | share of test | accuracy |
|---|---|---|
| high | 0.430 | 0.827 |
| medium | 0.351 | 0.584 |
| low | 0.220 | 0.375 |

## Confusion matrix (rows = true, columns = predicted)

| true \ predicted | neutral | joy | sadness | anger | fear | disgust | surprise |
|---|---|---|---|---|---|---|---|
| neutral | 1021 | 58 | 58 | 48 | 4 | 5 | 62 |
| joy | 90 | 223 | 8 | 41 | 1 | 1 | 38 |
| sadness | 69 | 12 | 73 | 32 | 4 | 3 | 15 |
| anger | 70 | 18 | 13 | 183 | 2 | 6 | 53 |
| fear | 18 | 3 | 7 | 15 | 0 | 0 | 7 |
| disgust | 21 | 1 | 7 | 23 | 0 | 5 | 11 |
| surprise | 39 | 26 | 3 | 39 | 0 | 2 | 172 |
