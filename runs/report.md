# Results on the MELD test split

## Classification

| model | weighted-F1 | macro-F1 | accuracy | neutral | joy | sadness | anger | fear | disgust | surprise |
|---|---|---|---|---|---|---|---|---|---|---|
| majority class (neutral) | 0.313 | 0.093 | 0.481 | 0.650 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| text only | 0.619 | 0.434 | 0.636 | 0.785 | 0.588 | 0.352 | 0.427 | 0.149 | 0.192 | 0.542 |
| audio only | 0.479 | 0.287 | 0.495 | 0.650 | 0.348 | 0.232 | 0.415 | 0.000 | 0.000 | 0.363 |
| vision only | 0.354 | 0.154 | 0.421 | 0.600 | 0.274 | 0.000 | 0.108 | 0.000 | 0.025 | 0.071 |
| text + vision (deployed) | 0.621 | 0.420 | 0.631 | 0.784 | 0.598 | 0.363 | 0.470 | 0.059 | 0.154 | 0.512 |
| text + vision, no context | 0.619 | 0.418 | 0.631 | 0.787 | 0.594 | 0.332 | 0.446 | 0.088 | 0.136 | 0.543 |
| text + audio | 0.631 | 0.419 | 0.643 | 0.790 | 0.600 | 0.387 | 0.504 | 0.000 | 0.111 | 0.538 |
| text + vision + audio | 0.640 | 0.443 | 0.654 | 0.797 | 0.610 | 0.371 | 0.490 | 0.092 | 0.162 | 0.576 |
| deployed model, text head only (views.text; 2610 turns with that input) | 0.623 | 0.441 | 0.635 | 0.781 | 0.582 | 0.332 | 0.474 | 0.125 | 0.243 | 0.549 |
| deployed model, vision head only (views.vision; 2539 turns with that input) | 0.357 | 0.152 | 0.454 | 0.628 | 0.217 | 0.036 | 0.090 | 0.040 | 0.000 | 0.050 |

## Gain over text only (weighted-F1, paired bootstrap over test utterances, 1000 resamples)

| model | difference | 95% interval |
|---|---|---|
| text + vision (deployed) | +0.001 | [-0.011, +0.014] |
| text + vision, no context | +0.000 | [-0.012, +0.013] |
| text + audio | +0.012 | [-0.001, +0.025] |
| text + vision + audio | +0.020 | [+0.009, +0.033] |

## Turns where the views disagree

49.5% of test turns. Accuracy on them: fused 0.526, the words 0.526, the face 0.177

## Certainty buckets (temperature 1.227)

| certainty | share of test | accuracy |
|---|---|---|
| high | 0.411 | 0.829 |
| medium | 0.354 | 0.583 |
| low | 0.235 | 0.356 |

## Confusion matrix (rows = true, columns = predicted)

| true \ predicted | neutral | joy | sadness | anger | fear | disgust | surprise |
|---|---|---|---|---|---|---|---|
| neutral | 1015 | 53 | 56 | 54 | 7 | 6 | 65 |
| joy | 88 | 221 | 6 | 41 | 2 | 2 | 42 |
| sadness | 76 | 11 | 67 | 32 | 4 | 2 | 16 |
| anger | 73 | 21 | 15 | 167 | 2 | 5 | 62 |
| fear | 19 | 1 | 5 | 15 | 2 | 0 | 8 |
| disgust | 21 | 2 | 7 | 20 | 0 | 7 | 11 |
| surprise | 42 | 28 | 5 | 37 | 1 | 1 | 167 |
