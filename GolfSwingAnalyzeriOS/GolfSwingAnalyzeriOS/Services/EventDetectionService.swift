import CoreML
import CoreVideo
import Accelerate
import UIKit
import os

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "EventDetection")

enum EventDetectionError: Error, LocalizedError {
    case modelLoadFailed(String)
    case preprocessingFailed
    case inferenceFailed(String)
    case noEventsDetected

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let msg): return "Failed to load model: \(msg)"
        case .preprocessingFailed: return "Frame preprocessing failed"
        case .inferenceFailed(let msg): return "Inference failed: \(msg)"
        case .noEventsDetected: return "No swing events detected"
        }
    }
}

struct EventDetectionService {

    private let seqLength = 64

    // ImageNet normalization (must match Python pipeline exactly)
    // Note: Python code uses std [0.299, 0.224, 0.225] — the 0.299 appears to be a typo
    // in the original code but we match it for consistency with the trained weights.
    private let imagenetMean: [Float] = [0.485, 0.456, 0.406]
    private let imagenetStd: [Float] = [0.299, 0.224, 0.225]
    private let inputSize = 160

    /// Detect 8 swing events from video frames.
    /// Returns a dictionary mapping each SwingEvent to its frame index in the original video.
    func detectEvents(frames: [CVPixelBuffer]) async throws -> [SwingEvent: Int] {
        logger.info("Loading CoreML models...")
        let cnnModel = try loadCNNModel()
        let lstmModel = try loadLSTMModel()
        logger.info("Models loaded successfully")

        // Step 1: Extract CNN features for each frame
        logger.info("Extracting CNN features for \(frames.count) frames...")
        var features: [[Float]] = []
        for (i, frame) in frames.enumerated() {
            let preprocessed = try preprocessFrame(frame)
            let featureVector = try runCNN(model: cnnModel, input: preprocessed)
            features.append(featureVector)
            if (i + 1) % 50 == 0 || i == frames.count - 1 {
                logger.debug("  CNN features: \(i + 1)/\(frames.count)")
            }
        }
        logger.info("CNN feature extraction complete: \(features.count) vectors of dim \(features.first?.count ?? 0)")

        // Step 2: Run LSTM on feature sequences in batches of seqLength
        logger.info("Running LSTM inference (seq_length=\(self.seqLength))...")
        var allLogits: [[Float]] = []
        var batch = 0
        while batch * seqLength < features.count {
            let start = batch * seqLength
            let end = min((batch + 1) * seqLength, features.count)
            let batchFeatures = Array(features[start..<end])

            let logits = try runLSTM(model: lstmModel, features: batchFeatures)
            allLogits.append(contentsOf: logits)
            logger.debug("  LSTM batch \(batch): frames \(start)..<\(end)")
            batch += 1
        }
        logger.info("LSTM inference complete: \(allLogits.count) frame logits, \(batch) batches")

        // Step 3: Apply softmax and find event frames
        let probs = softmax2D(allLogits)
        let events = extractEvents(from: probs)

        for (event, frameIdx) in events.sorted(by: { $0.key.classIndex < $1.key.classIndex }) {
            let confidence = probs[frameIdx][event.classIndex]
            logger.info("  \(event.rawValue): frame \(frameIdx) (confidence: \(String(format: "%.3f", confidence)))")
        }

        return events
    }

    // MARK: - Model Loading

    private func loadCNNModel() throws -> MLModel {
        guard let modelURL = Bundle.main.url(forResource: "SwingNetCNN", withExtension: "mlmodelc") else {
            throw EventDetectionError.modelLoadFailed("SwingNetCNN.mlmodelc not found in bundle")
        }
        do {
            return try MLModel(contentsOf: modelURL)
        } catch {
            throw EventDetectionError.modelLoadFailed(error.localizedDescription)
        }
    }

    private func loadLSTMModel() throws -> MLModel {
        guard let modelURL = Bundle.main.url(forResource: "SwingNetLSTM", withExtension: "mlmodelc") else {
            throw EventDetectionError.modelLoadFailed("SwingNetLSTM.mlmodelc not found in bundle")
        }
        do {
            return try MLModel(contentsOf: modelURL)
        } catch {
            throw EventDetectionError.modelLoadFailed(error.localizedDescription)
        }
    }

    // MARK: - Preprocessing

    /// Preprocess a frame to match the Python pipeline:
    /// 1. Resize maintaining aspect ratio to fit 160x160
    /// 2. Letterbox with ImageNet mean fill
    /// 3. Convert BGR to RGB, normalize to [0,1]
    /// 4. Subtract ImageNet means, divide by ImageNet stds
    /// Returns MLMultiArray of shape (1, 3, 160, 160)
    private func preprocessFrame(_ pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw EventDetectionError.preprocessingFailed
        }

        let pixelData = baseAddress.assumingMemoryBound(to: UInt8.self)

        // Calculate resize dimensions (maintain aspect ratio)
        let ratio = Float(inputSize) / Float(max(height, width))
        let newH = Int(Float(height) * ratio)
        let newW = Int(Float(width) * ratio)

        // Letterbox offsets
        let deltaH = inputSize - newH
        let deltaW = inputSize - newW
        let top = deltaH / 2
        let left = deltaW / 2

        // Create output MLMultiArray: (1, 3, 160, 160)
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)], dataType: .float32)

        // Fill with ImageNet mean (letterbox fill)
        for c in 0..<3 {
            let fillValue = imagenetMean[c] // Already normalized
            for y in 0..<inputSize {
                for x in 0..<inputSize {
                    // After normalization: (fillValue - mean) / std = 0.0
                    // So letterbox areas should be 0.0 after full normalization
                    let idx = c * inputSize * inputSize + y * inputSize + x
                    array[idx] = NSNumber(value: Float(0.0))
                }
            }
        }

        // Fill resized image area with normalized pixel values
        for y in 0..<newH {
            for x in 0..<newW {
                // Source pixel (nearest neighbor interpolation)
                let srcY = min(Int(Float(y) / ratio), height - 1)
                let srcX = min(Int(Float(x) / ratio), width - 1)
                let srcOffset = srcY * bytesPerRow + srcX * 4

                // BGRA format
                let b = Float(pixelData[srcOffset]) / 255.0
                let g = Float(pixelData[srcOffset + 1]) / 255.0
                let r = Float(pixelData[srcOffset + 2]) / 255.0

                // RGB channels normalized with ImageNet stats
                let outY = y + top
                let outX = x + left

                if outY < inputSize && outX < inputSize {
                    let rNorm = (r - imagenetMean[0]) / imagenetStd[0]
                    let gNorm = (g - imagenetMean[1]) / imagenetStd[1]
                    let bNorm = (b - imagenetMean[2]) / imagenetStd[2]

                    array[0 * inputSize * inputSize + outY * inputSize + outX] = NSNumber(value: rNorm)
                    array[1 * inputSize * inputSize + outY * inputSize + outX] = NSNumber(value: gNorm)
                    array[2 * inputSize * inputSize + outY * inputSize + outX] = NSNumber(value: bNorm)
                }
            }
        }

        return array
    }

    // MARK: - Inference

    private func runCNN(model: MLModel, input: MLMultiArray) throws -> [Float] {
        let provider = try MLDictionaryFeatureProvider(dictionary: ["frame": MLFeatureValue(multiArray: input)])
        let output = try model.prediction(from: provider)

        guard let features = output.featureValue(for: "features")?.multiArrayValue else {
            throw EventDetectionError.inferenceFailed("CNN output missing 'features'")
        }

        // Convert to Float array (1280 values)
        var result = [Float](repeating: 0, count: features.count)
        let ptr = features.dataPointer.bindMemory(to: Float.self, capacity: features.count)
        for i in 0..<features.count {
            result[i] = ptr[i]
        }
        return result
    }

    private func runLSTM(model: MLModel, features: [[Float]]) throws -> [[Float]] {
        let T = features.count
        let featureSize = 1280

        // Build input: (1, T, 1280)
        let array = try MLMultiArray(shape: [1, NSNumber(value: T), NSNumber(value: featureSize)], dataType: .float32)
        for t in 0..<T {
            for f in 0..<featureSize {
                array[t * featureSize + f] = NSNumber(value: features[t][f])
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: ["features": MLFeatureValue(multiArray: array)])
        let output = try model.prediction(from: provider)

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw EventDetectionError.inferenceFailed("LSTM output missing 'logits'")
        }

        // Parse output: (1, T, 9) -> [[Float]] of shape (T, 9)
        var result: [[Float]] = []
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: T * 9)
        for t in 0..<T {
            var row = [Float](repeating: 0, count: 9)
            for c in 0..<9 {
                row[c] = ptr[t * 9 + c]
            }
            result.append(row)
        }
        return result
    }

    // MARK: - Post-processing

    private func softmax2D(_ logits: [[Float]]) -> [[Float]] {
        logits.map { row in
            let maxVal = row.max() ?? 0
            let exps = row.map { exp($0 - maxVal) }
            let sum = exps.reduce(0, +)
            return exps.map { $0 / sum }
        }
    }

    /// For each of 8 events, find the frame with the highest probability.
    /// Matches Python: `events = np.argmax(probs, axis=0)[:-1]`
    private func extractEvents(from probs: [[Float]]) -> [SwingEvent: Int] {
        var events: [SwingEvent: Int] = [:]

        for eventIdx in 0..<8 {
            var maxProb: Float = -1
            var maxFrame = 0
            for frameIdx in 0..<probs.count {
                if probs[frameIdx][eventIdx] > maxProb {
                    maxProb = probs[frameIdx][eventIdx]
                    maxFrame = frameIdx
                }
            }
            if let event = SwingEvent.from(classIndex: eventIdx) {
                events[event] = maxFrame
            }
        }

        return events
    }
}
