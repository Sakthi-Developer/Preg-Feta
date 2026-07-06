# Explainability Summary (synthetic cohort)

- Test patients analysed: 122
- FETA most-attended gestational week (overall): 13
- PREG-Net most important variable: **YSD** (temporal, mean importance 0.1761)
- Mean attention on primary-rationale physiological edges: 0.1616

## Node importance (top 5)

| node_family   | node_feature   |   mean_importance |   sd_importance |   n |
|:--------------|:---------------|------------------:|----------------:|----:|
| temporal      | YSD            |            0.1761 |          0.1851 | 359 |
| temporal      | FHR            |            0.0927 |          0.0875 | 359 |
| temporal      | GS             |            0.0405 |          0.0576 | 359 |
| maternal      | age            |            0.0185 |          0.0212 | 122 |
| maternal      | gravidity      |            0.0174 |          0.0192 | 122 |

## Edge attention by clinical group

| clinical_group         |   mean_attention |    n |
|:-----------------------|-----------------:|-----:|
| primary_rationale      |           0.1616 | 1077 |
| temporal_same_variable |           0.135  | 1896 |
| additional_physio      |           0.1339 | 1795 |
| maternal_to_us         |           0.1123 | 6821 |

## Case studies

| case_type      | patient_id   |   label |   feta_prob |   preg_prob |   FETA_top_week | PREG_top_nodes         | PREG_top_edges              |
|:---------------|:-------------|--------:|------------:|------------:|----------------:|:-----------------------|:----------------------------|
| true_positive  | P00014       |       1 |       0.959 |       0.994 |              11 | FHR_t3, GS_t3, YSD_t3  | CRL->GS, previous_loss->YSD |
| true_negative  | P00002       |       0 |       0.003 |       0.003 |              12 | YSD_t5, YSD_t4, FHR_t5 | CRL->GS, CRL->FHR           |
| false_positive | P00089       |       0 |       0.12  |       0.942 |              12 | FHR_t4, GS_t4, FHR_t3  | CRL->GS, CRL->FHR           |
| false_negative | P00099       |       1 |       0.421 |       0.056 |               6 | YSD_t1, YSD_t2, FHR_t2 | CRL->GS, YSD->YSD           |

> All figures derive from synthetic data; they demonstrate model interpretability, not clinical findings.