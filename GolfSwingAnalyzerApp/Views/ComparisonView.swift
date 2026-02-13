import SwiftUI
import os

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "ComparisonView")

struct ComparisonView: View {
    let result: AnalysisResult
    @State private var selectedEvent: SwingEvent = .address

    var body: some View {
        VStack(spacing: 0) {
            eventPicker
            eventComparison
        }
        .navigationTitle("Pose Comparison")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            logger.info("Showing comparison view for \(result.bestMatch.playerName)")
        }
    }

    // MARK: - Event Picker

    private var eventPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(SwingEvent.allCases) { event in
                    Button {
                        selectedEvent = event
                    } label: {
                        Text(event.shortName)
                            .font(.caption)
                            .fontWeight(selectedEvent == event ? .semibold : .regular)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(selectedEvent == event ? Color.green : Color.gray.opacity(0.2))
                            .foregroundStyle(selectedEvent == event ? .white : .primary)
                            .clipShape(Capsule())
                    }
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 12)
        }
        .background(.ultraThinMaterial)
    }

    // MARK: - Event Comparison

    private var eventComparison: some View {
        ScrollView {
            VStack(spacing: 20) {
                // User's frame with pose overlay
                VStack(alignment: .leading, spacing: 8) {
                    Text("Your Swing")
                        .font(.headline)

                    if let cgImage = result.userEventFrames[selectedEvent] {
                        ZStack {
                            Image(uiImage: UIImage(cgImage: cgImage))
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .clipShape(RoundedRectangle(cornerRadius: 12))

                            if let pose = result.userPoses[selectedEvent] {
                                PoseOverlayView(
                                    landmarks: pose,
                                    imageSize: CGSize(
                                        width: CGFloat(cgImage.width),
                                        height: CGFloat(cgImage.height)
                                    ),
                                    color: .green
                                )
                            }
                        }
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else {
                        noFramePlaceholder
                    }
                }

                // Distance for this event
                if let dist = result.bestMatch.perEventDistances[selectedEvent] {
                    HStack {
                        Image(systemName: "arrow.left.and.right")
                        Text("Distance: \(String(format: "%.3f", dist))")
                            .font(.subheadline.monospacedDigit())
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                }

                // Pro info
                VStack(alignment: .leading, spacing: 8) {
                    Text("Pro: \(result.bestMatch.playerName)")
                        .font(.headline)

                    Text("Pro event images will appear here once bundled in the app.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(.gray.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
            }
            .padding()
        }
    }

    private var noFramePlaceholder: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(.gray.opacity(0.1))
            .frame(height: 250)
            .overlay {
                VStack {
                    Image(systemName: "photo")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No frame available")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
    }
}

// MARK: - Pose Overlay

struct PoseOverlayView: View {
    let landmarks: PoseLandmarks
    let imageSize: CGSize
    let color: Color

    var body: some View {
        GeometryReader { geometry in
            let scaleX = geometry.size.width / imageSize.width
            let scaleY = geometry.size.height / imageSize.height

            Canvas { context, size in
                // Draw skeleton connections
                for (from, to) in PoseLandmarks.skeletonConnections {
                    guard let fromPt = landmarks.joints[from],
                          let toPt = landmarks.joints[to] else { continue }

                    // Vision coordinates: origin bottom-left, need to flip Y
                    let p1 = CGPoint(
                        x: fromPt.x * imageSize.width * scaleX,
                        y: (1 - fromPt.y) * imageSize.height * scaleY
                    )
                    let p2 = CGPoint(
                        x: toPt.x * imageSize.width * scaleX,
                        y: (1 - toPt.y) * imageSize.height * scaleY
                    )

                    var path = Path()
                    path.move(to: p1)
                    path.addLine(to: p2)
                    context.stroke(path, with: .color(color.opacity(0.8)), lineWidth: 2)
                }

                // Draw joint dots
                for (_, point) in landmarks.joints {
                    let p = CGPoint(
                        x: point.x * imageSize.width * scaleX,
                        y: (1 - point.y) * imageSize.height * scaleY
                    )
                    let rect = CGRect(x: p.x - 4, y: p.y - 4, width: 8, height: 8)
                    context.fill(Path(ellipseIn: rect), with: .color(color))
                }
            }
        }
    }
}
