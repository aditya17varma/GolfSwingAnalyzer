import Foundation

enum SwingEvent: String, CaseIterable, Codable, Identifiable {
    case address = "Address"
    case toeUp = "Toe-up"
    case midBackswing = "Mid-backswing (arm parallel)"
    case top = "Top"
    case midDownswing = "Mid-downswing (arm parallel)"
    case impact = "Impact"
    case midFollowThrough = "Mid-follow-through (shaft parallel)"
    case finish = "Finish"

    var id: String { rawValue }

    var shortName: String {
        switch self {
        case .address: return "Address"
        case .toeUp: return "Toe-up"
        case .midBackswing: return "Backswing"
        case .top: return "Top"
        case .midDownswing: return "Downswing"
        case .impact: return "Impact"
        case .midFollowThrough: return "Follow-through"
        case .finish: return "Finish"
        }
    }

    /// Index matching the SwingNet output class ordering (0-7).
    var classIndex: Int {
        switch self {
        case .address: return 0
        case .toeUp: return 1
        case .midBackswing: return 2
        case .top: return 3
        case .midDownswing: return 4
        case .impact: return 5
        case .midFollowThrough: return 6
        case .finish: return 7
        }
    }

    static func from(classIndex: Int) -> SwingEvent? {
        allCases.first { $0.classIndex == classIndex }
    }
}

enum Perspective: String, CaseIterable, Identifiable {
    case front = "Front"
    case side = "Side"

    var id: String { rawValue }
}
