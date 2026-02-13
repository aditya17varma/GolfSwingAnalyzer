import SwiftUI
import os

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "ResultsView")

struct ResultsView: View {
    let result: AnalysisResult

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                bestMatchCard
                comparisonLink
                rankingsList
                perEventBreakdown
            }
            .padding()
        }
        .navigationTitle("Results")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            logger.info("Showing results: best match = \(result.bestMatch.playerName)")
        }
    }

    // MARK: - Best Match

    private var bestMatchCard: some View {
        VStack(spacing: 12) {
            Text("Closest Match")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Text(result.bestMatch.playerName)
                .font(.largeTitle)
                .fontWeight(.bold)

            HStack(spacing: 4) {
                Image(systemName: "target")
                Text("Similarity: \(String(format: "%.1f", (1 - result.bestMatch.normalizedScore) * 100))%")
            }
            .font(.headline)
            .foregroundStyle(.green)
        }
        .frame(maxWidth: .infinity)
        .padding(24)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Comparison Link

    private var comparisonLink: some View {
        NavigationLink {
            ComparisonView(result: result)
        } label: {
            HStack {
                Image(systemName: "person.2")
                Text("Compare Poses Side-by-Side")
                Spacer()
                Image(systemName: "chevron.right")
            }
            .padding()
            .background(.green.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    // MARK: - Rankings

    private var rankingsList: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("All Rankings")
                .font(.headline)

            ForEach(Array(result.allMatches.enumerated()), id: \.element.id) { index, match in
                HStack {
                    Text("#\(index + 1)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 30)

                    Text(match.playerName)
                        .fontWeight(index == 0 ? .semibold : .regular)

                    Spacer()

                    // Distance bar
                    let maxDist = result.allMatches.last?.totalDistance ?? 1
                    let barWidth = max(0.05, match.totalDistance / maxDist)

                    GeometryReader { geo in
                        RoundedRectangle(cornerRadius: 4)
                            .fill(index == 0 ? Color.green : Color.gray.opacity(0.3))
                            .frame(width: geo.size.width * barWidth)
                    }
                    .frame(width: 80, height: 12)

                    Text(String(format: "%.3f", match.totalDistance))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 60, alignment: .trailing)
                }
            }
        }
        .padding()
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Per-Event Breakdown

    private var perEventBreakdown: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Per-Event Breakdown (Best Match)")
                .font(.headline)

            ForEach(SwingEvent.allCases) { event in
                if let dist = result.bestMatch.perEventDistances[event] {
                    HStack {
                        Text(event.shortName)
                            .frame(width: 100, alignment: .leading)

                        let maxEventDist = result.bestMatch.perEventDistances.values.max() ?? 1
                        let barRatio = dist / maxEventDist

                        GeometryReader { geo in
                            RoundedRectangle(cornerRadius: 4)
                                .fill(barRatio < 0.5 ? Color.green : barRatio < 0.8 ? Color.orange : Color.red)
                                .frame(width: geo.size.width * barRatio)
                        }
                        .frame(height: 12)

                        Text(String(format: "%.3f", dist))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 50, alignment: .trailing)
                    }
                }
            }
        }
        .padding()
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
