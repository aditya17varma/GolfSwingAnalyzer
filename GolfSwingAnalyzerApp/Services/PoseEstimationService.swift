import Vision
import CoreVideo
import CoreImage
import os

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "PoseEstimation")

enum PoseEstimationError: Error, LocalizedError {
    case noPoseDetected
    case imageConversionFailed

    var errorDescription: String? {
        switch self {
        case .noPoseDetected: return "No body pose detected in frame"
        case .imageConversionFailed: return "Failed to convert image format"
        }
    }
}

struct PoseEstimationService {

    private let minimumConfidence: Float = 0.3

    /// Detect body pose from a CGImage.
    func detectPose(in image: CGImage) throws -> PoseLandmarks {
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])

        guard let observation = request.results?.first else {
            throw PoseEstimationError.noPoseDetected
        }

        return extractLandmarks(from: observation)
    }

    /// Detect body pose from a CVPixelBuffer.
    func detectPose(in pixelBuffer: CVPixelBuffer) throws -> PoseLandmarks {
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])

        guard let observation = request.results?.first else {
            throw PoseEstimationError.noPoseDetected
        }

        return extractLandmarks(from: observation)
    }

    /// Detect poses for multiple event frames at once.
    func detectPoses(in frames: [SwingEvent: CGImage]) -> [SwingEvent: PoseLandmarks] {
        logger.info("Detecting poses in \(frames.count) event frames...")
        var results: [SwingEvent: PoseLandmarks] = [:]
        for (event, image) in frames.sorted(by: { $0.key.classIndex < $1.key.classIndex }) {
            do {
                let landmarks = try detectPose(in: image)
                results[event] = landmarks
                logger.info("  \(event.shortName): \(landmarks.joints.count)/\(PoseLandmarks.comparisonJoints.count) joints detected")
            } catch {
                logger.warning("  \(event.shortName): pose detection failed — \(error.localizedDescription)")
            }
        }
        logger.info("Pose detection complete: \(results.count)/\(frames.count) events with poses")
        return results
    }

    // MARK: - Private

    private func extractLandmarks(from observation: VNHumanBodyPoseObservation) -> PoseLandmarks {
        var joints: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:]
        var confidences: [VNHumanBodyPoseObservation.JointName: Float] = [:]

        for jointName in PoseLandmarks.comparisonJoints {
            guard let point = try? observation.recognizedPoint(jointName),
                  point.confidence > minimumConfidence else {
                continue
            }
            // Vision coordinates: origin at bottom-left, normalized [0,1]
            joints[jointName] = point.location
            confidences[jointName] = point.confidence
        }

        return PoseLandmarks(joints: joints, confidences: confidences)
    }
}
