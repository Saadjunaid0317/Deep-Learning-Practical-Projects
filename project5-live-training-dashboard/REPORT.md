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
I recommend the run 4 settings: **16 neurons, dropout 0.3, weight decay 0.05, learning rate 0.001, early stopping with patience 20, and scaled features.**

- In my experiments this is the setup with a healthy curve: training loss and validation loss go down together and stay close.
  With 256 neurons, training loss drops to almost 0 but validation loss starts going up again (runs 1 and 2), which is overfitting.
- Adding dropout and weight decay alone was not enough (run 2 still overfits). Early stopping and a smaller model are what fixed it (runs 3 and 4).
- Its best validation loss (0.129) is almost as good as the big model's (0.125), and it gave the fewest false alarms: only 1 out of 71 healthy patients.
- Scaling the features is a must. Without it the same model jumps around and finds only 33 of 43 cancers (run 5).
- Because missing a cancer is much worse than a false alarm, I would ask the doctors to use a lower threshold.
  On the validation set, a threshold of 0.30 finds 41 of 43 cancers instead of 40, but gives 4 false alarms instead of 1 (dashboard run 6).
  I ran the final test only once, at 0.50, and did not repeat it with other thresholds.

## Why it is not ready for real patients
1. **The data is too small and comes from one place.** There are only 569 patients from one public dataset, and the test set has just 42 cancers,
   so a single patient changes the detection rate by about 2.4 points. The model has never been checked on patients from another hospital.
2. **It still misses cancers.** On the test set it missed 1 of 42 cancers, and on the validation set 3 of 43 (threshold 0.5).
   A doctor has to check every result. This model can help a doctor, but it cannot replace one.
3. **It depends heavily on how the data is prepared.** It fails without scaling, it needs exactly these 30 measurements from the same lab process,
   and there is no monitoring, calibration or explanation of its answers that a doctor could look at.
