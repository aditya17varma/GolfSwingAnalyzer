import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            NavigationStack {
                VideoPickerView()
            }
            .tabItem {
                Label("Analyze", systemImage: "figure.golf")
            }

            NavigationStack {
                ProGolferListView()
            }
            .tabItem {
                Label("Pro Golfers", systemImage: "person.3")
            }
        }
    }
}

#Preview {
    ContentView()
}
