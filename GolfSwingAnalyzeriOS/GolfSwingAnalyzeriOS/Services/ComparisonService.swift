import Foundation
import Vision
import os

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "Comparison")

enum ComparisonError: Error, LocalizedError {
    case proDataNotFound
    case decodingFailed(String)

    var errorDescription: String? {
        switch self {
        case .proDataNotFound: return "Pro golfer data file not found"
        case .decodingFailed(let msg): return "Failed to decode pro data: \(msg)"
        }
    }
}

struct ComparisonService {

    /// Load pro golfer data from the app bundle.
    func loadProData() throws -> ProGolferDatabase {
        guard let url = Bundle.main.url(forResource: "proGolferData", withExtension: "json") else {
            throw ComparisonError.proDataNotFound
        }
        let data = try Data(contentsOf: url)
        do {
            return try JSONDecoder().decode(ProGolferDatabase.self, from: data)
        } catch {
            throw ComparisonError.decodingFailed(error.localizedDescription)
        }
    }

    /// Find the closest pro golfer match by comparing pose landmarks across all swing events.
    func findClosestMatch(
        userPoses: [SwingEvent: PoseLandmarks],
        proDatabase: ProGolferDatabase,
        perspective: Perspective
    ) -> [ProGolferMatch] {
        let proEntries = proDatabase.entries(for: perspective)
        logger.info("Comparing against \(proEntries.count) pro golfers (\(perspective.rawValue) view)")
        var matches: [ProGolferMatch] = []

        for pro in proEntries {
            var totalDistance = 0.0
            var perEventDistances: [SwingEvent: Double] = [:]

            for event in SwingEvent.allCases {
                guard let userLandmarks = userPoses[event],
                      let proJoints = pro.events[event.rawValue] else {
                    continue
                }

                let eventDistance = calculateEventDistance(
                    userLandmarks: userLandmarks,
                    proJoints: proJoints
                )
                perEventDistances[event] = eventDistance
                totalDistance += eventDistance
            }

            matches.append(ProGolferMatch(
                id: pro.id,
                playerName: pro.playerName,
                totalDistance: totalDistance,
                perEventDistances: perEventDistances
            ))
        }

        let sorted = matches.sorted { $0.totalDistance < $1.totalDistance }
        for (i, match) in sorted.enumerated() {
            logger.info("  #\(i + 1) \(match.playerName): distance=\(String(format: "%.4f", match.totalDistance)) (\(match.perEventDistances.count) events compared)")
        }
        return sorted
    }

    // MARK: - Distance Calculation

    /// Calculate 2D Euclidean distance between user and pro landmarks for a single event.
    private func calculateEventDistance(
        userLandmarks: PoseLandmarks,
        proJoints: [JointPoint]
    ) -> Double {
        // Build lookup from pro joint data
        var proLookup: [String: CGPoint] = [:]
        var proConfLookup: [String: Double] = [:]
        for joint in proJoints {
            proLookup[joint.name] = CGPoint(x: joint.x, y: joint.y)
            proConfLookup[joint.name] = joint.confidence
        }

        var distance = 0.0
        for jointName in PoseLandmarks.comparisonJoints {
            guard let userPoint = userLandmarks.joints[jointName],
                  let proPoint = proLookup[jointName.rawValue.rawValue],
                  let proConf = proConfLookup[jointName.rawValue.rawValue],
                  proConf > 0.3 else {
                continue
            }

            let dx = userPoint.x - proPoint.x
            let dy = userPoint.y - proPoint.y
            distance += sqrt(dx * dx + dy * dy)
        }

        return distance
    }
}
