# Explainability Case Studies

## P00548: True Positive

- Label: 1
- PREG-Net probability: 1.0000
- FETA-Transformer probability: 0.9996
- Maternal profile: age 43.4, BMI 27.7, parity 0, gravidity 3, previous loss 1, conception IVF, singleton 1.0.
- Scan trajectory:
  - GA 6.1: FHR 134.8, CRL 1.0, GS 2.0, YSD 1.8
  - GA 8.5: FHR 134.8, CRL 1.6, GS 2.0, YSD 2.5
  - GA 11.2: FHR 151.5, CRL 5.3, GS 11.0, YSD 2.5
- Top PREG-Net nodes:
  - YSD_t3: 0.2181
  - YSD_t1: 0.1891
  - FHR_t3: 0.1489
  - YSD_t2: 0.1176
  - FHR_t1: 0.0923
- Top PREG-Net edges:
  - age_to_YSD (age -> YSD_t1): 0.3318
  - age_to_YSD (age -> YSD_t2): 0.3270
  - age_to_GS (age -> GS_t1): 0.3127
  - age_to_YSD (age -> YSD_t3): 0.3073
  - age_to_GS (age -> GS_t3): 0.3015
- Top FETA temporal attention scans:
  - GA 11.2 weeks: 0.9758
  - GA 8.5 weeks: 0.0167
  - GA 6.1 weeks: 0.0076

## P00320: True Negative

- Label: 0
- PREG-Net probability: 0.0000
- FETA-Transformer probability: 0.0005
- Maternal profile: age 29.2, BMI 21.6, parity 0, gravidity 1, previous loss 0, conception Natural, singleton 1.0.
- Scan trajectory:
  - GA 6.4: FHR 121.5, CRL 1.0, GS 2.0, YSD 3.5
  - GA 9.1: FHR 177.9, CRL 6.9, GS 3.7, YSD 4.8
- Top PREG-Net nodes:
  - FHR_t2: 0.9105
  - singleton: 0.0238
  - YSD_t1: 0.0141
  - gravidity: 0.0124
  - YSD_t2: 0.0108
- Top PREG-Net edges:
  - CRL_to_GS (CRL_t1 -> GS_t1): 0.2632
  - YSD_temporal_backward (YSD_t2 -> YSD_t1): 0.2347
  - FHR_to_YSD (FHR_t2 -> YSD_t2): 0.2298
  - CRL_to_GS (CRL_t2 -> GS_t2): 0.2212
  - FHR_to_YSD (FHR_t1 -> YSD_t1): 0.2041
- Top FETA temporal attention scans:
  - GA 9.1 weeks: 0.9842
  - GA 6.4 weeks: 0.0158

## P00089: False Positive

- Label: 0
- PREG-Net probability: 0.9978
- FETA-Transformer probability: 0.8495
- Maternal profile: age 30.5, BMI 32.4, parity 0, gravidity 1, previous loss 1, conception Natural, singleton 1.0.
- Scan trajectory:
  - GA 6.5: FHR 108.6, CRL 1.3, GS 2.5, YSD 3.1
  - GA 8.4: FHR 144.3, CRL 4.8, GS 2.0, YSD 3.8
  - GA 10.3: FHR 157.9, CRL 8.5, GS 11.0, YSD 3.9
  - GA 12.4: FHR 162.2, CRL 16.1, GS 13.4, YSD 3.9
- Top PREG-Net nodes:
  - FHR_t4: 0.2519
  - YSD_t4: 0.1446
  - YSD_t3: 0.1052
  - GS_t4: 0.0830
  - FHR_t2: 0.0809
- Top PREG-Net edges:
  - FHR_to_YSD (FHR_t1 -> YSD_t1): 0.2848
  - GS_to_YSD (GS_t4 -> YSD_t4): 0.2655
  - previous_loss_to_YSD (previous_loss -> YSD_t1): 0.2510
  - CRL_to_GS (CRL_t4 -> GS_t4): 0.2418
  - CRL_to_GS (CRL_t1 -> GS_t1): 0.2321
- Top FETA temporal attention scans:
  - GA 12.4 weeks: 1.0000
  - GA 10.3 weeks: 0.0000
  - GA 8.4 weeks: 0.0000

## P00099: False Negative

- Label: 1
- PREG-Net probability: 0.1369
- FETA-Transformer probability: 0.9766
- Maternal profile: age 25.0, BMI 21.1, parity 1, gravidity 1, previous loss 0, conception Natural, singleton 1.0.
- Scan trajectory:
  - GA 6.1: FHR 111.9, CRL 2.9, GS 7.0, YSD 4.3
  - GA 8.5: FHR 111.9, CRL 5.2, GS 2.0, YSD 4.3
- Top PREG-Net nodes:
  - singleton: 0.2049
  - FHR_t1: 0.1631
  - FHR_t2: 0.1299
  - gravidity: 0.1065
  - GS_t2: 0.0969
- Top PREG-Net edges:
  - CRL_to_GS (CRL_t2 -> GS_t2): 0.2326
  - YSD_temporal_forward (YSD_t1 -> YSD_t2): 0.2192
  - YSD_temporal_backward (YSD_t2 -> YSD_t1): 0.2108
  - CRL_to_GS (CRL_t1 -> GS_t1): 0.1981
  - FHR_to_YSD (FHR_t2 -> YSD_t2): 0.1965
- Top FETA temporal attention scans:
  - GA 8.5 weeks: 0.9953
  - GA 6.1 weeks: 0.0047

