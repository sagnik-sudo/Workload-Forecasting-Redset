import pandas as pd


class TopClustersFinder:
    """
    Load query data, identify top clusters based on total query count, and manage output.
    """

    def __init__(self, file_path, top_n=5):
        """
        Initialize the TopClustersFinder.

        Args:
            file_path (str): Path to the Parquet file containing query data.
            top_n (int): Number of top clusters to retrieve.
        """
        self.file_path = file_path
        self.top_n = top_n
        self.df = None
        self.top_clusters = None

    def load_data(self):
        """
        Load data from a Parquet file into a DataFrame.

        Raises:
            ValueError: If required columns ('instance_id', 'query_count') are missing.
        """
        print("Loading data from Parquet file...")
        self.df = pd.read_parquet(self.file_path)

        # Verify necessary columns exist
        required_columns = {"instance_id", "query_count"}
        if not required_columns.issubset(self.df.columns):
            missing = required_columns - set(self.df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        print("Data loaded successfully.")

    def find_top_clusters(self):
        """
        Identify the top clusters based on the total query count.

        Returns:
            pandas.DataFrame: DataFrame containing 'instance_id' and 'total_queries' for the top clusters.
        """
        print(f"Finding the top {self.top_n} clusters...")
        # Sum query counts grouped by 'instance_id'
        query_counts = self.df.groupby("instance_id")["query_count"].sum()

        # Retrieve the top N clusters
        self.top_clusters = query_counts.nlargest(self.top_n).reset_index()
        self.top_clusters.columns = ["instance_id", "total_queries"]

        print("Top clusters identified.")
        return self.top_clusters

    def save_results(self, output_file="top_clusters.csv"):
        """
        Save the top clusters DataFrame to a CSV file.

        Args:
            output_file (str): Filename for the output CSV.
        """
        if self.top_clusters is not None:
            self.top_clusters.to_csv(output_file, index=False)
            print(f"Results saved to '{output_file}'")

    def run(self):
        """
        Execute the entire workflow: load data, process top clusters, and display results.

        Returns:
            pandas.DataFrame: DataFrame with the top clusters and their total queries.
        """
        self.load_data()
        top_clusters = self.find_top_clusters()
        print("\nTop Clusters:\n", top_clusters)
        return top_clusters


if __name__ == "__main__":
    finder = TopClustersFinder("provisioned.parquet")
    finder.run()
