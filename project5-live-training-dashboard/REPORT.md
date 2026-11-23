# City Hospital: screening model report

## Experiments (validation set, threshold 0.5)
| run | diagnosis | epochs run | best val loss | cancers found | false alarms |
|---|---|---|---|---|---|
| 1 · 256, no fixes | overfitting | 300 | 0.125 @ 23 | 41 of 43 | 3 |
| 2 · + dropout & decay | overfitting | 300 | 0.131 @ 24 | 40 of 43 | 2 |
| 3 · + early stopping | healthy | 44 | 0.131 @ 24 | 40 of 43 | 3 |
| 4 · 16 neurons, all fixes | healthy | 177 | 0.129 @ 157 | 40 of 43 | 1 |
| 5 · same, NOT scaled | unstable | 27 | 0.430 @ 7 | 33 of 43 | 2 |

## Chosen settings
neurons 16, dropout 0.3, weight decay 0.05,
learning rate 0.001, early stopping patience 20, features scaled: True

## Final test (run once)
accuracy 0.947 · 41 of 42 cancers found · 5 false alarms

## Recommendation
Use the run 4 settings: **16 neurons, dropout 0.3, weight decay 0.05, learning rate 0.001, early stopping (patience 20), features scaled.**

- It is the only setup where training and validation loss fall **together** and stay close (a healthy curve).
  The 256-neuron runs drive training loss to almost 0 while validation loss climbs back up (runs 1 and 2).
- Dropout and weight decay alone did **not** fix the overfitting (run 2). Early stopping and the smaller model did (runs 3 and 4).
- Its best validation loss (0.129) is almost the same as the 256-neuron model's (0.125), with the **fewest false alarms** (1 of 71 healthy patients).
- Scaling is required: without it the same model is unstable and finds only 33 of 43 cancers (run 5).
- For screening, the doctors should agree on a **lower decision threshold**. On the validation set, 0.30 instead of 0.50
  finds 41 of 43 cancers instead of 40, at the cost of 4 false alarms instead of 1 (dashboard run 6).
  The final test was run once at 0.50 and was not repeated at other thresholds.

## Why it is not ready for real patients
1. **Too little and too narrow data.** 569 patients from one public dataset, and the final test has only 114 patients (42 cancers).
   One patient more or less moves the cancer-detection rate by about 2.4 points, and the model has never seen patients from another hospital or machine.
2. **It still misses cancers.** It missed 1 of 42 cancers on the test set and 3 of 43 on the validation set at threshold 0.5.
   For a screening tool, every result still needs a doctor to review it; it can support a doctor, not replace one.
3. **It is fragile to how the data is prepared.** Without scaling it breaks (run 5), it expects exactly these 30 measurements
   from the same lab process, and there is no monitoring, probability calibration or explanation of its decisions for the doctors.
