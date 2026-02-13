import Foundation
import Vision

struct JointPoint: Codable {
    let name: String
    let x: Double
    let y: Double
    let confidence: Double
}

struct PoseLandmarks {
    let joints: [VNHumanBodyPoseObservation.JointName: CGPoint]
    let confidences: [VNHumanBodyPoseObservation.JointName: Float]

    /// The joints used for comparison — the structurally significant ones for golf swing analysis.
    static let comparisonJoints: [VNHumanBodyPoseObservation.JointName] = [
        .nose,
        .neck,
        .leftShoulder,
        .rightShoulder,
        .leftElbow,
        .rightElbow,
        .leftWrist,
        .rightWrist,
        .leftHip,
        .rightHip,
        .leftKnee,
        .rightKnee,
        .leftAnkle,
        .rightAnkle,
        .root
    ]

    /// Skeleton connections for drawing pose overlay.
    static let skeletonConnections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
        // Torso
        (.neck, .root),
        (.neck, .leftShoulder),
        (.neck, .rightShoulder),
        (.root, .leftHip),
        (.root, .rightHip),
        // Left arm
        (.leftShoulder, .leftElbow),
        (.leftElbow, .leftWrist),
        // Right arm
        (.rightShoulder, .rightElbow),
        (.rightElbow, .rightWrist),
        // Left leg
        (.leftHip, .leftKnee),
        (.leftKnee, .leftAnkle),
        // Right leg
        (.rightHip, .rightKnee),
        (.rightKnee, .rightAnkle),
    ]
}
