import SwiftUI
import os

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "AnalysisView")

struct AnalysisView: View {
    let videoURL: URL
    let perspective: Perspective
    @ObservedObject var viewModel: AnalysisViewModel

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            progressSection

            stageDescription

            if case .complete = viewModel.stage {
                NavigationLink {
                    if let result = viewModel.result {
                        ResultsView(result: result)
                    }
                } label: {
                    Text("View Results")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.green)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .padding(.horizontal)
            }

            if case .failed(let msg) = viewModel.stage {
                VStack(spacing: 12) {
                    Text(msg)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)

                    Button("Try Again") {
                        startAnalysis()
                    }
                }
            }

            Spacer()
        }
        .padding()
        .navigationTitle("Analyzing")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            startAnalysis()
        }
    }

    private var progressSection: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .stroke(.gray.opacity(0.2), lineWidth: 8)
                    .frame(width: 120, height: 120)

                Circle()
                    .trim(from: 0, to: viewModel.stage.progress)
                    .stroke(.green, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                    .frame(width: 120, height: 120)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.5), value: viewModel.stage.progress)

                if case .complete = viewModel.stage {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.green)
                } else if case .failed = viewModel.stage {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.red)
                } else {
                    Text("\(Int(viewModel.stage.progress * 100))%")
                        .font(.title2.monospacedDigit())
                        .fontWeight(.semibold)
                }
            }
        }
    }

    private var stageDescription: some View {
        VStack(spacing: 8) {
            Text(viewModel.stage.description)
                .font(.headline)

            Text("\(perspective.rawValue) view")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private func startAnalysis() {
        guard case .idle = viewModel.stage else {
            if case .failed = viewModel.stage {
                viewModel.reset()
            } else {
                return
            }
        }
        logger.info("Launching analysis task")
        Task {
            await viewModel.analyze(videoURL: videoURL, perspective: perspective)
        }
    }
}
