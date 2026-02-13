import SwiftUI
import os

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "ProGolferList")

struct ProGolferListView: View {
    @State private var database: ProGolferDatabase?
    @State private var loadError: String?

    private let comparisonService = ComparisonService()

    var body: some View {
        Group {
            if let database {
                List {
                    Section("Front View") {
                        ForEach(database.front) { entry in
                            golferRow(entry)
                        }
                    }
                    Section("Side View") {
                        ForEach(database.side) { entry in
                            golferRow(entry)
                        }
                    }
                }
            } else if let loadError {
                ContentUnavailableView {
                    Label("Data Not Available", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(loadError)
                } actions: {
                    Text("Run the reprocess_pro_data.swift script and bundle proGolferData.json")
                        .font(.caption)
                }
            } else {
                ProgressView("Loading pro golfer data...")
            }
        }
        .navigationTitle("Pro Golfers")
        .task {
            do {
                database = try comparisonService.loadProData()
                logger.info("Loaded pro data: \(database?.front.count ?? 0) front, \(database?.side.count ?? 0) side")
            } catch {
                loadError = error.localizedDescription
                logger.error("Failed to load pro data: \(error.localizedDescription)")
            }
        }
    }

    private func golferRow(_ entry: ProGolferEntry) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.playerName)
                .font(.headline)
            HStack {
                Text(entry.club)
                Text("·")
                Text("\(entry.events.count) events")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    NavigationStack {
        ProGolferListView()
    }
}
