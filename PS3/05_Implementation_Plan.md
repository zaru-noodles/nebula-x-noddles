# Train Condition Monitoring Hackathon Implementation Plan

## Strategy

Attempt all four subsystems, while allocating most modelling time to Door and SHM. Breadth directly improves the Overall Score, and a valid modest model contributes more than omitting a subsystem.

| Priority | Subsystem | Rationale |
|---|---|---|
| 1 | Door | Segmentation is unusually tractable and offers the fastest route to a strong score. |
| 2 | SHM | The target is physics-derived and the supplied signals have a strong relationship with damage. |
| 3 | ACV | There are only six labelled cases, but the ranking metric gives useful partial credit. |
| 4 | Rail corrugation | This has the largest dataset, severe class imbalance, and the most expensive signal processing. |

## 1. Door

### Key observation

The challenge is joint temporal segmentation and Normal/Abnormal-resistance classification, scored using IoU-weighted F1.

In the supplied training data:

- All 18,036 rows belong to the 110 labelled cycles.
- The normal sampling interval is 20 ms.
- There are exactly 109 timestamp gaps greater than one second, matching the 110 labelled segments.
- Test has 37 equivalent gaps, implying 38 candidate cycles.

Timestamp-gap segmentation is therefore a strong baseline and may produce near-exact boundaries if the test-generation process is consistent.

### Pipeline

1. Parse the custom timestamp format.
2. Split the stream where the timestamp delta exceeds a data-derived threshold, initially one second.
3. Cross-check boundaries using command, movement, and door-position signals.
4. Infer Open/Close internally from position direction or controller flags.
5. Extract cycle-level features:
   - Current mean, minimum, peak, RMS, quantiles, and area
   - Voltage and back-EMF statistics
   - Current versus door position
   - Current versus normalized cycle progress
   - Peak current during acceleration, steady movement, and end-stop phases
   - Cycle duration, stalls, and position derivatives
6. Compare:
   - Logistic regression or SVM
   - Random forest or gradient boosting
   - Nearest-template or dynamic-time-warping classification
7. Either train separate Open and Close classifiers or include operation as a feature.
8. Select the model using the exact end-to-end IoU-weighted F1 calculation.

### Validation

- Use contiguous chronological blocks instead of random row splits.
- Evaluate segmentation and classification together.
- Track segment count, mean IoU, Normal/Abnormal precision and recall, and the official score.
- Check that a boundary rule learned from Train also produces plausible segment durations and counts on Test.

## 2. Structural Health Monitoring

### Approach

The reference target was produced using rainflow counting and Miner's rule. Reproduce that mechanism before trying a generic machine-learning model.

1. Remove offsets and obvious acquisition artifacts.
2. Extract turning points from the stress series.
3. Perform rainflow counting.
4. For a grid of S-N exponents `m`, calculate a damage proxy:

   ```text
   D_m = sum(n_i * sigma_a_i ^ m)
   ```

5. Fit the exponent and scale constant against the 64 training labels.
6. Test an optional mean-stress correction such as Goodman correction.
7. Build a statistical fallback model using:
   - RMS and standard deviation
   - Maximum absolute stress
   - High stress quantiles
   - Crest factor and kurtosis
   - Spectral-band energy
   - High-order stress moments
8. Blend the physical and statistical models only if out-of-fold MAPE improves.

Maximum absolute stress has approximately 0.92 Spearman correlation with damage in the supplied training set, so a calibrated physics-informed model is promising.

### Validation

- Use leave-one-file-out or repeated grouped cross-validation.
- Select models using MAPE rather than RMSE or R-squared.
- Keep predictions positive.
- Consider fitting in log space, but always judge the final predictions in the original scale using the official score:

  ```text
  score = max(0, 1 - MAPE)
  ```

- Inspect percentage error by low-, medium-, and high-damage ranges because low true values can dominate MAPE.

## 3. ACV

### Approach

With only six labelled cases, avoid a conventional eight-class supervised classifier. Each file contains seven healthy peer cars that can serve as an internal reference for the faulty car.

1. Parse car identifiers and parameter names dynamically from each workbook's headers.
2. Do not assume every case has the same columns.
3. Respect information-valid flags and exclude invalid observations.
4. At every timestamp, calculate each car's robust residual from the other cars.
5. Extract per-car features:
   - Indoor temperature minus cooling target
   - Indoor temperature relative to peer cars
   - Cooling recovery after control-mode transitions
   - Time spent above peer-temperature thresholds
   - Running-mode and load-halving duty cycles
   - Persistent deviations rather than isolated spikes
   - Trend and change-point features
   - Missing or invalid telemetry rates
6. Produce a weighted anomaly score for every car.
7. Tune feature weights with leave-one-case-out validation using the official rank-decay score.
8. Output every car identifier from most to least likely faulty.

### Validation

- Hold out one entire case at a time.
- Never split timestamps from the same case between training and validation.
- Report mean faulty-car rank, top-1 rate, top-3 rate, and the official ranking score.
- Ensure the output uses identifiers exactly as found in the file headers, such as `03`, rather than `Car 3`.

## 4. Rail Corrugation

### Approach

Avoid feeding the raw `10,000 x 129` arrays directly into a large neural network. Only 38 of the 272 training files are faults: 14 Side I and 24 Side II.

1. Estimate train speed from rotational pulse transitions and wheel diameter.
2. Detrend and window each vibration and shock channel.
3. Compute Welch power spectral densities and time-domain shock features.
4. Convert frequency signatures into spatial wavelength using:

   ```text
   wavelength = speed / frequency
   ```

5. Aggregate axle-box signals independently by side:
   - Side I: positions 1, 3, 5, and 7
   - Side II: positions 2, 4, 6, and 8
6. Use robust median, upper quantiles, and maximum response across cars.
7. Add explicit Side-I-minus-Side-II contrast features.
8. Extract features such as:
   - Speed-normalized spectral bandpower
   - Dominant wavelength and spectral peak prominence
   - Spectral entropy
   - RMS, kurtosis, crest factor, and shock counts
   - Cross-car consistency on each side
   - Side-to-side energy and peak ratios
9. Compare a direct three-class model with a hierarchical model:
   - Stage A: Normal versus corrugation
   - Stage B: Side I versus Side II
10. Use class weighting and optimize out-of-fold thresholds for macro F1.

### Validation

- Use repeated stratified validation because the fault classes are small.
- Group or audit folds by speed and operating-condition clusters.
- Never use file number as a feature; the documentation states that identifiers are random.
- Report the confusion matrix and per-class precision, recall, and F1, not accuracy alone.

## Shared Evaluation Layer

Implement the official metrics before substantial model tuning:

- Door: greedy same-label segment matching and IoU-weighted F1
- ACV: linear rank-decay score
- Rail corrugation: macro F1 across Normal, Side I, and Side II
- SHM: `max(0, 1 - MAPE)`

Use these implementations for every experiment so local model selection matches leaderboard scoring.

## Application Architecture

Build one inference core with four subsystem adapters:

```text
Uploaded files
    |
    v
Schema detection and validation
    |
    v
Door / ACV / Rail / SHM preprocessing
    |
    v
Subsystem model
    |
    v
Prediction and explanation
    |
    v
Strict output-schema validator
    |
    v
Download CSV or predictions.zip
```

The same Python inference functions should serve:

- A Streamlit application for non-technical users
- A `predict.py` command-line interface for reproducibility
- Local notebooks or training scripts

### App workflow

1. Select a subsystem.
2. Upload or drag and drop one or more input files.
3. Validate the files and explain any schema errors.
4. Run inference with visible progress.
5. Display the result and a subsystem-specific diagnostic plot.
6. Allow the user to download the result CSV.
7. Allow all attempted subsystem outputs to be packaged as `predictions.zip`.

### Useful explanations

- Door: detected cycle boundaries, current/position trace, and abnormality score
- ACV: ranked cars and the telemetry deviations driving each score
- Rail: side-specific spectra or wavelength-band response
- SHM: predicted damage and the stress-cycle or high-stress features driving it

## Output Validation

Before creating `predictions.zip`, enforce:

- Exact required filenames
- Exact output column names
- Source filenames including extensions
- No duplicate file IDs
- Only permitted class labels
- Numeric, finite SHM predictions
- Every ACV car included exactly once in `ranked_cars`
- Valid and ordered Door timestamps
- At least one plausible Door segment
- CSV files placed at the top level of the zip with no subfolders

## Execution Order

1. Implement all four official metrics.
2. Build schema loaders and submission validators.
3. Finish the Door end-to-end baseline.
4. Implement rainflow-based SHM prediction.
5. Add ACV peer-anomaly ranking.
6. Add the Rail spectral baseline.
7. Integrate all four pipelines into the application.
8. Generate predictions through the application itself.
9. Spend remaining time improving the weakest out-of-fold metric.
10. Freeze models and run a clean end-to-end submission rehearsal.
11. Record the demo video after the final application and packaging flow work correctly.

## Suggested Work Allocation

### Phase 1: Foundation

- Repository structure and configuration
- Reusable loaders
- Official metrics
- Output-schema validation
- Experiment logging

### Phase 2: Fast high-value baselines

- Door segmentation and classification
- SHM physical baseline

### Phase 3: Breadth

- ACV peer-ranking baseline
- Rail side-aware spectral baseline

### Phase 4: Productization

- Streamlit interface
- Diagnostic visualizations
- Downloadable predictions
- `predictions.zip` generation
- Error handling

### Phase 5: Final tuning and submission

- Out-of-fold comparison and model selection
- Threshold and ensemble tuning
- Held-out input inference
- Submission-schema checks
- Reproducibility rehearsal
- Demo recording and short write-up

## Submission Story

Present the solution as one condition-monitoring interface with four deliberately different modelling strategies chosen to match each physical system:

- State-aware cycle analysis for doors
- Physics-informed fatigue estimation for SHM
- Peer-based anomaly ranking for ACV
- Speed-normalized, side-aware spectral diagnosis for rail corrugation

This framing demonstrates problem fit, methodological discipline, explainability, and usefulness for predictive maintenance rather than presenting one generic model for unrelated sensor problems.

## Source Material

- `01_Problem_Statement_3_Specifications.md`
- `03_References/Door/Door_Subsystem_Info_Kit.md`
- `03_References/ACV/ACV_Subsystem_Info_Kit.md`
- `03_References/Rail_Corrugation/Rail_Corrugation_Info_Kit.md`
- `03_References/SHM/SHM_Info_Kit.md`
