import AVFoundation
import CoreImage
import UIKit
import os

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "VideoProcessing")

enum VideoProcessingError: Error, LocalizedError {
    case noVideoTrack
    case readerFailed(String)
    case noFramesExtracted

    var errorDescription: String? {
        switch self {
        case .noVideoTrack: return "No video track found"
        case .readerFailed(let msg): return "Video reader failed: \(msg)"
        case .noFramesExtracted: return "No frames could be extracted"
        }
    }
}

struct VideoProcessingService {

    /// Extract all frames from a video file as CVPixelBuffers.
    func extractFrames(from url: URL) async throws -> [CVPixelBuffer] {
        logger.info("Loading video: \(url.lastPathComponent)")
        let asset = AVURLAsset(url: url)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            logger.error("No video track found in \(url.lastPathComponent)")
            throw VideoProcessingError.noVideoTrack
        }

        let duration = try await asset.load(.duration)
        let fps = try await videoTrack.load(.nominalFrameRate)
        let size = try await videoTrack.load(.naturalSize)
        logger.info("Video: \(String(format: "%.1f", duration.seconds))s, \(String(format: "%.0f", fps))fps, \(Int(size.width))x\(Int(size.height))")

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        let reader = try AVAssetReader(asset: asset)
        let trackOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        reader.add(trackOutput)

        guard reader.startReading() else {
            let err = reader.error?.localizedDescription ?? "Unknown"
            logger.error("Reader failed to start: \(err)")
            throw VideoProcessingError.readerFailed(err)
        }

        var frames: [CVPixelBuffer] = []
        while reader.status == .reading {
            guard let sampleBuffer = trackOutput.copyNextSampleBuffer(),
                  let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                continue
            }
            CVPixelBufferRetain(pixelBuffer)
            frames.append(pixelBuffer)
        }

        if reader.status == .failed {
            let err = reader.error?.localizedDescription ?? "Unknown"
            logger.error("Reader failed during extraction: \(err)")
            throw VideoProcessingError.readerFailed(err)
        }

        guard !frames.isEmpty else {
            logger.error("No frames extracted from video")
            throw VideoProcessingError.noFramesExtracted
        }

        logger.info("Extracted \(frames.count) frames (\(CVPixelBufferGetWidth(frames[0]))x\(CVPixelBufferGetHeight(frames[0])))")
        return frames
    }

    /// Extract a single frame at a specific index from a video.
    func extractFrame(from url: URL, at frameIndex: Int) async throws -> CGImage {
        let asset = AVURLAsset(url: url)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoProcessingError.noVideoTrack
        }

        let duration = try await asset.load(.duration)
        let frameCount = try await Double(videoTrack.load(.nominalFrameRate)) * duration.seconds
        let time = CMTime(seconds: Double(frameIndex) / frameCount * duration.seconds, preferredTimescale: 600)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let (image, _) = try await generator.image(at: time)
        return image
    }

    /// Convert a CVPixelBuffer to CGImage.
    static func cgImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        return context.createCGImage(ciImage, from: ciImage.extent)
    }

    /// Generate a thumbnail for a video URL.
    func generateThumbnail(for url: URL) async -> UIImage? {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 300, height: 300)

        do {
            let (cgImage, _) = try await generator.image(at: .zero)
            return UIImage(cgImage: cgImage)
        } catch {
            return nil
        }
    }
}
