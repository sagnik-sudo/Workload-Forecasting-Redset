import pandas as pd

class TopClustersFinder:
    def __init__(self, file_path, top_n=5):
        """
        Initialize the class with a Parquet file path and number of top clusters to retrieve.

        Args:
            file_path (str): Path to the Parquet file.
            top_n (int): Number of top clusters to retrieve based on total queries.
        """
        self.file_path = file_path
        self.top_n = top_n
        self.df = None
        self.top_clusters = None

    def load_data(self):
        """Load the Parquet file into a DataFrame."""
        print("📥 Loading Parquet file...")
        self.df = pd.read_parquet(self.file_path)

        # Ensure required columns exist
        required_columns = {"instance_id", "query_count"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"❌ Missing columns: {required_columns - set(self.df.columns)}")

        print("✅ Data loaded successfully!")

    def find_top_clusters(self):
        """Find the top N clusters based on total query count."""
        print(f"🔍 Finding the top {self.top_n} clusters...")

        # Aggregate total queries per instance_id
        query_counts = self.df.groupby("instance_id")["query_count"].sum()

        # Get the top N instances
        self.top_clusters = query_counts.nlargest(self.top_n).reset_index()
        self.top_clusters.columns = ["instance_id", "total_queries"]

        print("✅ Top clusters identified!")
        return self.top_clusters

    def save_results(self, output_file="top_clusters.csv"):
        """Save the top clusters to a CSV file."""
        if self.top_clusters is not None:
            self.top_clusters.to_csv(output_file, index=False)
            print(f"📂 Results saved to '{output_file}'")

    def run(self):
        """Execute the full pipeline to find the top clusters."""
        self.load_data()
        top_clusters = self.find_top_clusters()
        print("\n🏆 Top Clusters:\n", top_clusters)
        return top_clusters

# Example Usage
if __name__ == "__main__":
    finder = TopClustersFinder("provisioned.parquet")
    top_clusters = finder.run()
