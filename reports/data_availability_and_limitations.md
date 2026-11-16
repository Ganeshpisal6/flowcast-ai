# FlowCast AI - Data Availability and Limitations

## 1. Available datasets

### PEMS04
- Raw file: data/raw/pems04/pems04.npz
- Sensors: 307
- Original sampling interval: 5 minutes
- Processed interval: 30 minutes
- Forecast horizons: 30, 60, 90 and 120 minutes
- Available measurements: flow, occupancy and speed

### PEMS-BAY
- Files: speed.csv and occupancy.csv
- Measurements are organized by sensor ID and timestamp.
- The inspected files do not provide road distance or road capacity.

### UCI Metro Interstate Traffic Volume
- Available as an hourly traffic dataset.
- It does not represent the PRD's 25-segment, 30-minute corridor structure.

## 2. Missing data

The currently inspected project datasets do not provide verified:
- Road-segment distances
- Road-capacity values
- Weather observations linked to the traffic sensors
- Calendar and local-event records
- Accident records or accident-risk labels
- Verified road geometry or sensor-to-road mapping

## 3. Impact on PRD requirements

| PRD requirement | Current status |
|---|---|
| 30-minute traffic forecasting | Implemented using PEMS04 |
| 30/60/90/120-minute flow forecasting | Implemented |
| 30/60/90/120-minute speed forecasting | Implemented |
| Congestion classification | Implemented using provisional speed-based labels |
| Travel-time estimation | Blocked: distance and capacity data unavailable |
| Weather integration | Blocked: linked weather data unavailable |
| Calendar/event integration | Blocked: event data unavailable |
| Accident-risk prediction | Blocked: accident labels unavailable |

## 4. Important limitations

PEMS04 contains 307 sensors rather than the PRD's specified 25 road segments.
The congestion labels are proxy labels based on provisional speed thresholds,
not independently verified ground-truth congestion labels.

The available data does not support a reliable travel-time or accident-risk
model without additional validated source data.

## 5. Recommended next action

Obtain verified road distance, capacity, weather, calendar/event and accident
data before implementing the corresponding PRD modules.

Do not fabricate missing values or present proxy outputs as verified results.
