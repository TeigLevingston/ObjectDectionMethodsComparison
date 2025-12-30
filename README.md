# ObjectDectionMethodsComparison
An exercise in using Machine Learning on a Raspberry PI on the edge to detect objects in images from high resolution cameras.
I tested five ways of feeding a 4K video stream into a fixed YOLO11m model running on a Raspberry Pi 5 with a Hailo8 accelerator. The examples run on a Pi with or without an accelerator, or a host with or without CUDA.
Motion-based Region-of-Interest (ROI) selection improved F1 score by ~0.26 over full-frame resizing, even after controlling for confidence thresholds, duplicate suppression, and target distance.
Target distance dominated performance: far targets reduced F1 by ~0.45 compared to near targets, regardless of method.
The takeaway: how you prepare images matters as much as the model itself, especially on constrained hardware.

