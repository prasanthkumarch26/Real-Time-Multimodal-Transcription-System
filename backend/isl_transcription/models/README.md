# Place your trained LSTM model file here:
#   action.h5
#
# The model should be trained for ISL gesture recognition with:
#   - Input shape: (30, 1629) — 30 frames of 1629-dim MediaPipe Holistic features
#   - Output: softmax over 7 ISL words: [hello, thanks, iloveyou, please, yes, no, good]
#
# If action.h5 is not present, the service runs in stub/demo mode.
