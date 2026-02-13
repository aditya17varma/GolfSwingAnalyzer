import Foundation

struct ProGolferEntry: Codable, Identifiable {
    let id: String
    let playerName: String
    let club: String
    let perspective: String
    let events: [String: [JointPoint]]
}

struct ProGolferDatabase: Codable {
    let front: [ProGolferEntry]
    let side: [ProGolferEntry]

    func entries(for perspective: Perspective) -> [ProGolferEntry] {
        switch perspective {
        case .front: return front
        case .side: return side
        }
    }
}
