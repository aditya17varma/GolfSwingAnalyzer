#!/usr/bin/swift

/// Reprocess pro golfer event images using Apple Vision framework.
///
/// Reads all event JPGs from PoseDetection/proEvents/{front,side}/
/// and extracts body pose landmarks using VNDetectHumanBodyPoseRequest.
/// Outputs proGolferData.json for bundling in the iOS app.
///
/// Usage:
///   swift Scripts/reprocess_pro_data.swift

import Foundation
import Vision
import AppKit

// MARK: - Configuration

let scriptDir = URL(fileURLWithPath: #file).deletingLastPathComponent()
let projectRoot = scriptDir.deletingLastPathComponent()
let proEventsDir = projectRoot.appendingPathComponent("PoseDetection/proEvents")
let outputPath = scriptDir.appendingPathComponent("output/proGolferData.json")

let eventNames = [
    "Address",
    "Toe-up",
    "Mid-backswing (arm parallel)",
    "Top",
    "Mid-downswing (arm parallel)",
    "Impact",
    "Mid-follow-through (shaft parallel)",
    "Finish"
]

let jointNames: [VNHumanBodyPoseObservation.JointName] = [
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

// MARK: - Data Structures

struct JointData: Codable {
    let name: String
    let x: Double
    let y: Double
    let confidence: Double
}

struct GolferEntry: Codable {
    let id: String
    let playerName: String
    let club: String
    let perspective: String
    let events: [String: [JointData]]
}

struct ProGolferDatabase: Codable {
    let front: [GolferEntry]
    let side: [GolferEntry]
}

// MARK: - Pose Extraction

func extractPose(from imagePath: URL) -> [JointData]? {
    guard let image = NSImage(contentsOf: imagePath),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("  Warning: Could not load image at \(imagePath.lastPathComponent)")
        return nil
    }

    let request = VNDetectHumanBodyPoseRequest()
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

    do {
        try handler.perform([request])
    } catch {
        print("  Warning: Vision request failed for \(imagePath.lastPathComponent): \(error)")
        return nil
    }

    guard let observation = request.results?.first else {
        print("  Warning: No pose detected in \(imagePath.lastPathComponent)")
        return nil
    }

    var joints: [JointData] = []
    for jointName in jointNames {
        do {
            let point = try observation.recognizedPoint(jointName)
            joints.append(JointData(
                name: jointName.rawValue.rawValue,
                x: Double(point.location.x),
                y: Double(point.location.y),
                confidence: Double(point.confidence)
            ))
        } catch {
            // Joint not detected — add with zero confidence
            joints.append(JointData(
                name: jointName.rawValue.rawValue,
                x: 0,
                y: 0,
                confidence: 0
            ))
        }
    }

    return joints
}

func parseGolferName(from folderId: String) -> (playerName: String, club: String) {
    // Format: "Adam-Scott_LongIrons_Front1" or "Rory-McIlroy-LongIrons_Side1"
    let parts = folderId.components(separatedBy: "_")
    let namePart = parts[0].replacingOccurrences(of: "-", with: " ")
    let club = parts.count > 1 ? parts[1] : "Unknown"
    return (namePart, club)
}

func processGolfers(perspective: String) -> [GolferEntry] {
    let perspectiveDir = proEventsDir.appendingPathComponent(perspective)
    let fileManager = FileManager.default

    guard let golferFolders = try? fileManager.contentsOfDirectory(
        at: perspectiveDir,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    ) else {
        print("Error: Could not read \(perspectiveDir.path)")
        return []
    }

    var entries: [GolferEntry] = []

    for folder in golferFolders.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: folder.path, isDirectory: &isDir), isDir.boolValue else {
            continue
        }

        let folderId = folder.lastPathComponent
        let (playerName, club) = parseGolferName(from: folderId)
        print("Processing \(perspective)/\(folderId)...")

        var eventData: [String: [JointData]] = [:]
        var eventsFound = 0

        for eventName in eventNames {
            // Filename pattern: {folderId}.mp4_{eventName}.jpg
            let filename = "\(folderId).mp4_\(eventName).jpg"
            let imagePath = folder.appendingPathComponent(filename)

            guard fileManager.fileExists(atPath: imagePath.path) else {
                print("  Missing: \(filename)")
                continue
            }

            if let joints = extractPose(from: imagePath) {
                eventData[eventName] = joints
                eventsFound += 1
            }
        }

        print("  Detected poses in \(eventsFound)/\(eventNames.count) events")

        entries.append(GolferEntry(
            id: folderId,
            playerName: playerName,
            club: club,
            perspective: perspective,
            events: eventData
        ))
    }

    return entries
}

// MARK: - Main

print("Reprocessing pro golfer data with Apple Vision...\n")

let frontEntries = processGolfers(perspective: "front")
print()
let sideEntries = processGolfers(perspective: "side")

let database = ProGolferDatabase(front: frontEntries, side: sideEntries)

// Write JSON
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

do {
    let outputDir = outputPath.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
    let jsonData = try encoder.encode(database)
    try jsonData.write(to: outputPath)
    print("\nWrote \(frontEntries.count) front + \(sideEntries.count) side entries to:")
    print("  \(outputPath.path)")
} catch {
    print("Error writing JSON: \(error)")
    exit(1)
}

print("\nDone!")
