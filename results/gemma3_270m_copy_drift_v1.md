# Copy-or-drift probe v1 — gemma-3-270m-it

| set | level | target | class | asked | first drifting decimal |
|---|---|---|---|---|---|
| dev | L1 | cox/clinical/harrell_c | copy | 0.647 | — |
| dev | L2 | cox/clinical/harrell_c | copy | 0.647 | — |
| dev | L3 | cox/clinical/harrell_c | copy | 0.647 | — |
| dev | L4 | cox/clinical/harrell_c | misattribution | 0.647 | — |
| dev | L1 | rsf/clinical_expression/uno_c | omission | 0.615 | — |
| dev | L2 | rsf/clinical_expression/uno_c | omission | 0.615 | — |
| dev | L3 | rsf/clinical_expression/uno_c | omission | 0.615 | — |
| dev | L4 | rsf/clinical_expression/uno_c | omission | 0.615 | — |
| dev | L5 | all | omission | — | — |
| dev | L5 | all | omission | — | — |
| heldout | L1 | cox/clinical_expression/harrell_c | omission | 0.643 | — |
| heldout | L2 | cox/clinical_expression/harrell_c | copy | 0.643 | — |
| heldout | L3 | cox/clinical_expression/harrell_c | misattribution | 0.643 | — |
| heldout | L4 | cox/clinical_expression/harrell_c | copy | 0.643 | — |
| heldout | L1 | rsf/clinical_expression/harrell_c | omission | 0.632 | — |
| heldout | L2 | rsf/clinical_expression/harrell_c | copy | 0.632 | — |
| heldout | L3 | rsf/clinical_expression/harrell_c | misattribution | 0.632 | — |
| heldout | L4 | rsf/clinical_expression/harrell_c | misattribution | 0.632 | — |
| heldout | L1 | rsf/clinical_expression/auc_36m | copy | 0.609 | — |
| heldout | L2 | rsf/clinical_expression/auc_36m | copy | 0.609 | — |
| heldout | L3 | rsf/clinical_expression/auc_36m | misattribution | 0.609 | — |
| heldout | L4 | rsf/clinical_expression/auc_36m | misattribution | 0.609 | — |
| heldout | L1 | rsf/clinical_expression/integrated_brier_6_36m | copy | 0.153 | — |
| heldout | L2 | rsf/clinical_expression/integrated_brier_6_36m | copy | 0.153 | — |
| heldout | L3 | rsf/clinical_expression/integrated_brier_6_36m | copy | 0.153 | — |
| heldout | L4 | rsf/clinical_expression/integrated_brier_6_36m | copy | 0.153 | — |
| heldout | L5 | all | omission | — | — |

## Per-level summary

| set | level | copy | drift | omission | misattribution | partial |
|---|---|---|---|---|---|---|
| dev | L1 | 1 | 0 | 1 | 0 | 0 |
| dev | L2 | 1 | 0 | 1 | 0 | 0 |
| dev | L3 | 1 | 0 | 1 | 0 | 0 |
| dev | L4 | 0 | 0 | 1 | 1 | 0 |
| dev | L5 | 0 | 0 | 2 | 0 | 0 |
| heldout | L1 | 2 | 0 | 2 | 0 | 0 |
| heldout | L2 | 4 | 0 | 0 | 0 | 0 |
| heldout | L3 | 1 | 0 | 0 | 3 | 0 |
| heldout | L4 | 2 | 0 | 0 | 2 | 0 |
| heldout | L5 | 0 | 0 | 1 | 0 | 0 |
