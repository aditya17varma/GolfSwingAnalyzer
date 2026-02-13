import SwiftUI
import Combine
import os
import CoreVideo

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "Analysis")

@MainActor
class AnalysisViewModel: ObservableObject {
    @Published var stage: AnalysisStage = .idle
    @Published var result: AnalysisResult?

    private let videoService = VideoProcessingService()
    private let eventDetectionService = EventDetectionService()
    private let poseService = PoseEstimationService()
    private let comparisonService = ComparisonService()

    func analyze(videoURL: URL, perspective: Perspective) async {
        logger.info("Starting analysis: video=\(videoURL.lastPathComponent), perspective=\(perspective.rawValue)")
        let startTime = CFAbsoluteTimeGetCurrent()

        do {
            // Step 1: Extract frames
            stage = .extractingFrames
            logger.info("[1/4] Extracting frames from video...")
            let stepStart = CFAbsoluteTimeGetCurrent()
            let frames = try await videoService.extractFrames(from: videoURL)
            logger.info("[1/4] Extracted \(frames.count) frames in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - stepStart))s")

            // Step 2: Detect swing events
            stage = .detectingEvents
            logger.info("[2/4] Detecting swing events via CoreML...")
            let stepStart2 = CFAbsoluteTimeGetCurrent()
            let eventFrameIndices = try await eventDetectionService.detectEvents(frames: frames)
            logger.info("[2/4] Event detection complete in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - stepStart2))s")
            for (event, frameIdx) in eventFrameIndices.sorted(by: { $0.value < $1.value }) {
                logger.info("  \(event.rawValue): frame \(frameIdx)")
            }

            // Extract event frame CGImages
            var eventFrames: [SwingEvent: CGImage] = [:]
            for (event, frameIndex) in eventFrameIndices {
                guard frameIndex < frames.count,
                      let cgImage = VideoProcessingService.cgImage(from: frames[frameIndex]) else {
                    logger.warning("  Could not extract frame \(frameIndex) for \(event.rawValue)")
                    continue
                }
                eventFrames[event] = cgImage
            }
            logger.info("[2/4] Extracted \(eventFrames.count)/8 event frame images")

            // Pixel buffers are released automatically by ARC

            // Step 3: Analyze poses
            stage = .analyzingPoses
            logger.info("[3/4] Analyzing poses with Apple Vision...")
            let stepStart3 = CFAbsoluteTimeGetCurrent()
            let userPoses = poseService.detectPoses(in: eventFrames)
            logger.info("[3/4] Pose estimation complete in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - stepStart3))s")
            for (event, landmarks) in userPoses.sorted(by: { $0.key.classIndex < $1.key.classIndex }) {
                logger.info("  \(event.shortName): \(landmarks.joints.count) joints detected")
            }

            // Step 4: Compare with pro golfers
            stage = .comparingWithPros
            logger.info("[4/4] Comparing with pro golfers...")
            let stepStart4 = CFAbsoluteTimeGetCurrent()
            let proDatabase = try comparisonService.loadProData()
            logger.info("[4/4] Loaded pro data: \(proDatabase.front.count) front, \(proDatabase.side.count) side entries")

            let matches = comparisonService.findClosestMatch(
                userPoses: userPoses,
                proDatabase: proDatabase,
                perspective: perspective
            )
            logger.info("[4/4] Comparison complete in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - stepStart4))s")

            guard let bestMatch = matches.first else {
                throw ComparisonError.proDataNotFound
            }

            logger.info("Best match: \(bestMatch.playerName) (distance: \(String(format: "%.4f", bestMatch.totalDistance)))")
            for match in matches.prefix(5) {
                logger.info("  \(match.playerName): \(String(format: "%.4f", match.totalDistance))")
            }

            result = AnalysisResult(
                perspective: perspective,
                bestMatch: bestMatch,
                allMatches: matches,
                userEventFrames: eventFrames,
                userPoses: userPoses
            )
            stage = .complete

            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            logger.info("Analysis complete in \(String(format: "%.1f", totalTime))s")

        } catch {
            logger.error("Analysis failed: \(error.localizedDescription)")
            stage = .failed(error.localizedDescription)
        }
    }

    func reset() {
        logger.info("Resetting analysis state")
        stage = .idle
        result = nil
    }
}
