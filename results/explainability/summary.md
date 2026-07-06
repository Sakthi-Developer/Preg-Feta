# Explainability Summary

Generated from the fixed test split using trained FETA-Transformer and PREG-Net checkpoints.

## FETA Temporal Attention

| Outcome | Week | Scans | Mean Temporal Attention | Mean Maternal-to-Time Attention |
| --- | ---: | ---: | ---: | ---: |
| ongoing | 5 | 8 | 0.0900 | 0.3758 |
| ongoing | 6 | 42 | 0.0428 | 0.4154 |
| ongoing | 7 | 49 | 0.0506 | 0.3893 |
| ongoing | 8 | 43 | 0.4901 | 0.3540 |
| ongoing | 9 | 47 | 0.4122 | 0.3230 |
| ongoing | 10 | 28 | 0.5227 | 0.3098 |
| ongoing | 11 | 30 | 0.6936 | 0.2778 |
| ongoing | 12 | 12 | 0.6411 | 0.2644 |
| ongoing | 13 | 9 | 0.3694 | 0.1944 |
| ongoing | 14 | 5 | 0.2187 | 0.2288 |
| loss | 6 | 17 | 0.1007 | 0.3919 |
| loss | 7 | 12 | 0.0871 | 0.2988 |
| loss | 8 | 12 | 0.3445 | 0.3639 |
| loss | 9 | 14 | 0.4107 | 0.2956 |
| loss | 10 | 12 | 0.4567 | 0.3202 |
| loss | 11 | 8 | 0.5392 | 0.3882 |
| loss | 12 | 6 | 0.5576 | 0.2904 |
| loss | 13 | 2 | 0.5612 | 0.3338 |
| loss | 14 | 3 | 0.6991 | 0.2965 |

## PREG-Net Node Importance

| Rank | Node Feature | Family | Mean Importance |
| ---: | --- | --- | ---: |
| 1 | FHR | temporal | 0.2283 |
| 2 | YSD | temporal | 0.0613 |
| 3 | singleton | maternal | 0.0400 |
| 4 | gravidity | maternal | 0.0301 |
| 5 | GS | temporal | 0.0189 |
| 6 | conception_ivf | maternal | 0.0036 |
| 7 | previous_loss | maternal | 0.0032 |
| 8 | bmi | maternal | 0.0030 |
| 9 | CRL | temporal | 0.0029 |
| 10 | age | maternal | 0.0022 |
| 11 | parity | maternal | 0.0014 |

## PREG-Net Edge Attention

| Rank | Relationship | Clinical Group | Mean Attention |
| ---: | --- | --- | ---: |
| 1 | CRL_to_GS | additional_code_edge | 0.2167 |
| 2 | FHR_to_YSD | additional_code_edge | 0.1952 |
| 3 | CRL_temporal_backward | temporal_same_variable | 0.1680 |
| 4 | GS_to_YSD | additional_code_edge | 0.1679 |
| 5 | GS_temporal_backward | temporal_same_variable | 0.1660 |
| 6 | previous_loss_to_YSD | primary_research_rationale | 0.1626 |
| 7 | CRL_to_FHR | additional_code_edge | 0.1602 |
| 8 | age_to_YSD | additional_code_edge | 0.1599 |
| 9 | YSD_temporal_backward | temporal_same_variable | 0.1550 |
| 10 | FHR_temporal_forward | temporal_same_variable | 0.1416 |
| 11 | FHR_to_CRL | primary_research_rationale | 0.1382 |
| 12 | CRL_temporal_forward | temporal_same_variable | 0.1380 |

## Selected Case Studies

| Case Type | Patient | Label | PREG Probability | FETA Probability |
| --- | --- | ---: | ---: | ---: |
| true_positive | P00548 | 1 | 1.0000 | 0.9996 |
| true_negative | P00320 | 0 | 0.0000 | 0.0005 |
| false_positive | P00089 | 0 | 0.9978 | 0.8495 |
| false_negative | P00099 | 1 | 0.1369 | 0.9766 |

Note: FETA maternal-to-time maps are post-hoc maps derived from the trained maternal query and temporal keys.
The original model forward path uses maternal cross-attention over the pooled temporal token, so these maps should be reported as interpretability aids rather than as a retrained architectural claim.
