import Foundation
import CoreGraphics
import CoreImage

struct AnalysisResult: Identifiable {
    let id = UUID()
    let perspective: Perspective
    let bestMatch: ProGolferMatch
    let allMatches: [ProGolferMatch]
    let userEventFrames: [SwingEvent: CGImage]
    let userPoses: [SwingEvent: PoseLandmarks]
}

struct ProGolferMatch: Identifiable {
    let id: String
    let playerName: String
    let totalDistance: Double
    let perEventDistances: [SwingEvent: Double]

    /// Normalized score where 0 = perfect match, 1 = worst match.
    var normalizedScore: Double {
        // Empirical: total distance < 1.0 is very good, > 5.0 is poor
        min(totalDistance / 5.0, 1.0)
    }
}

enum AnalysisStage: Equatable {
    case idle
    case extractingFrames
    case detectingEvents
    case analyzingPoses
    case comparingWithPros
    case complete
    case failed(String)

    var description: String {
        switch self {
        case .idle: return "Ready"
        case .extractingFrames: return "Extracting frames..."
        case .detectingEvents: return "Detecting swing events..."
        case .analyzingPoses: return "Analyzing poses..."
        case .comparingWithPros: return "Comparing with pros..."
        case .complete: return "Analysis complete"
        case .failed(let msg): return "Error: \(msg)"
        }
    }

    var progress: Double {
        switch self {
        case .idle: return 0
        case .extractingFrames: return 0.15
        case .detectingEvents: return 0.45
        case .analyzingPoses: return 0.70
        case .comparingWithPros: return 0.90
        case .complete: return 1.0
        case .failed: return 0
        }
    }
}
