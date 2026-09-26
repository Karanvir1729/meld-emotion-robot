# Results on the MELD test split

## Classification

| model | weighted-F1 | macro-F1 | accuracy | neutral | joy | sadness | anger | fear | disgust | surprise |
|---|---|---|---|---|---|---|---|---|---|---|
| majority class (neutral) | 0.313 | 0.093 | 0.481 | 0.650 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| text only | 0.619 | 0.434 | 0.636 | 0.785 | 0.588 | 0.352 | 0.427 | 0.149 | 0.192 | 0.542 |
| audio only | 0.479 | 0.287 | 0.495 | 0.650 | 0.348 | 0.232 | 0.415 | 0.000 | 0.000 | 0.363 |
| vision only | 0.354 | 0.154 | 0.421 | 0.600 | 0.274 | 0.000 | 0.108 | 0.000 | 0.025 | 0.071 |
| text + vision | 0.621 | 0.420 | 0.631 | 0.784 | 0.598 | 0.363 | 0.470 | 0.059 | 0.154 | 0.512 |
| text + vision, no context | 0.619 | 0.418 | 0.631 | 0.787 | 0.594 | 0.332 | 0.446 | 0.088 | 0.136 | 0.543 |
| text + audio | 0.631 | 0.419 | 0.643 | 0.790 | 0.600 | 0.387 | 0.504 | 0.000 | 0.111 | 0.538 |
| text + vision + audio (deployed) | 0.640 | 0.443 | 0.654 | 0.797 | 0.610 | 0.371 | 0.490 | 0.092 | 0.162 | 0.576 |
| deployed model, text head only (views.text; 2610 turns with that input) | 0.612 | 0.432 | 0.626 | 0.779 | 0.568 | 0.318 | 0.449 | 0.179 | 0.210 | 0.522 |
| deployed model, audio head only (views.audio; 2554 turns with that input) | 0.475 | 0.280 | 0.517 | 0.682 | 0.289 | 0.237 | 0.359 | 0.069 | 0.000 | 0.323 |
| deployed model, vision head only (views.vision; 2539 turns with that input) | 0.361 | 0.155 | 0.447 | 0.625 | 0.249 | 0.043 | 0.084 | 0.000 | 0.029 | 0.058 |

## Gain over text only (weighted-F1, paired bootstrap over test utterances, 1000 resamples)

| model | difference | 95% interval |
|---|---|---|
| text + vision | +0.001 | [-0.011, +0.014] |
| text + vision, no context | +0.000 | [-0.012, +0.013] |
| text + audio | +0.012 | [-0.001, +0.025] |
| text + vision + audio (deployed) | +0.020 | [+0.009, +0.033] |

## Turns where the views disagree

61.2% of test turns. Accuracy on them: fused 0.589, the words 0.541, the voice 0.372, the face 0.261

## Learned WavLM layer weights (0 = CNN output, 12 = last transformer layer)

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weight | 0.077 | 0.074 | 0.074 | 0.074 | 0.073 | 0.076 | 0.078 | 0.080 | 0.081 | 0.082 | 0.079 | 0.075 | 0.077 |

## Certainty buckets (temperature 1.344)

| certainty | share of test | accuracy |
|---|---|---|
| high | 0.464 | 0.826 |
| medium | 0.333 | 0.584 |
| low | 0.203 | 0.377 |

## Confusion matrix (rows = true, columns = predicted)

| true \ predicted | neutral | joy | sadness | anger | fear | disgust | surprise |
|---|---|---|---|---|---|---|---|
| neutral | 1051 | 68 | 39 | 30 | 4 | 8 | 56 |
| joy | 91 | 246 | 8 | 20 | 0 | 5 | 32 |
| sadness | 80 | 18 | 63 | 24 | 4 | 4 | 15 |
| anger | 76 | 34 | 8 | 152 | 3 | 15 | 57 |
| fear | 21 | 4 | 4 | 13 | 3 | 0 | 5 |
| disgust | 25 | 2 | 7 | 16 | 0 | 9 | 9 |
| surprise | 39 | 32 | 3 | 20 | 1 | 2 | 184 |
