import SwiftUI
import PhotosUI
import AVKit
import os

private let logger = Logger(subsystem: "com.golfswinganalyzer", category: "VideoPicker")

struct VideoPickerView: View {
    @StateObject private var viewModel = AnalysisViewModel()
    @State private var selectedVideoURL: URL?
    @State private var selectedPerspective: Perspective = .front
    @State private var showPhotoPicker = false
    @State private var thumbnail: UIImage?
    @State private var navigateToAnalysis = false

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                headerSection
                videoSelectionSection
                perspectiveSection
                analyzeButton
            }
            .padding()
        }
        .navigationTitle("Golf Swing Analyzer")
        .sheet(isPresented: $showPhotoPicker) {
            PHPickerRepresentable(videoURL: $selectedVideoURL)
        }
        .onChange(of: selectedVideoURL) { _, newURL in
            if let url = newURL {
                logger.info("Video selected: \(url.lastPathComponent)")
                loadThumbnail(for: url)
            }
        }
        .navigationDestination(isPresented: $navigateToAnalysis) {
            if let url = selectedVideoURL {
                AnalysisView(
                    videoURL: url,
                    perspective: selectedPerspective,
                    viewModel: viewModel
                )
            }
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "figure.golf")
                .font(.system(size: 48))
                .foregroundStyle(.green)
            Text("Compare your swing with the pros")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 20)
    }

    private var videoSelectionSection: some View {
        VStack(spacing: 12) {
            if let thumbnail {
                Image(uiImage: thumbnail)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(.secondary.opacity(0.3), lineWidth: 1)
                    )
            }

            Button {
                showPhotoPicker = true
            } label: {
                Label(
                    selectedVideoURL == nil ? "Select Video" : "Change Video",
                    systemImage: "video.badge.plus"
                )
                .frame(maxWidth: .infinity)
                .padding()
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    private var perspectiveSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Camera Angle")
                .font(.headline)

            Picker("Perspective", selection: $selectedPerspective) {
                ForEach(Perspective.allCases) { perspective in
                    Text(perspective.rawValue).tag(perspective)
                }
            }
            .pickerStyle(.segmented)

            Text(selectedPerspective == .front
                 ? "Camera facing the golfer head-on"
                 : "Camera to the side of the golfer")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var analyzeButton: some View {
        Button {
            logger.info("Starting analysis: perspective=\(selectedPerspective.rawValue)")
            navigateToAnalysis = true
        } label: {
            Text("Analyze Swing")
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .background(selectedVideoURL != nil ? Color.green : Color.gray)
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .disabled(selectedVideoURL == nil)
    }

    // MARK: - Helpers

    private func loadThumbnail(for url: URL) {
        Task {
            thumbnail = await VideoProcessingService().generateThumbnail(for: url)
        }
    }
}

// MARK: - PHPicker Wrapper

struct PHPickerRepresentable: UIViewControllerRepresentable {
    @Binding var videoURL: URL?
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .videos
        config.selectionLimit = 1
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: PHPickerRepresentable

        init(_ parent: PHPickerRepresentable) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.dismiss()

            guard let result = results.first else { return }

            result.itemProvider.loadFileRepresentation(forTypeIdentifier: "public.movie") { url, error in
                if let error {
                    logger.error("Failed to load video: \(error.localizedDescription)")
                    return
                }
                guard let sourceURL = url else {
                    logger.error("Video URL is nil")
                    return
                }

                // Copy to temp directory for persistent access
                let tempDir = FileManager.default.temporaryDirectory
                let destURL = tempDir.appendingPathComponent(UUID().uuidString + "." + sourceURL.pathExtension)

                do {
                    try FileManager.default.copyItem(at: sourceURL, to: destURL)
                    logger.info("Copied video to: \(destURL.lastPathComponent)")
                    DispatchQueue.main.async {
                        self.parent.videoURL = destURL
                    }
                } catch {
                    logger.error("Failed to copy video: \(error.localizedDescription)")
                }
            }
        }
    }
}

#Preview {
    NavigationStack {
        VideoPickerView()
    }
}
