import os
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from seaborn import displot, catplot, countplot
import openpyxl
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import customtkinter as ctk
from customtkinter import filedialog
import time
from datetime import datetime

matplotlib.use('TkAgg')

class ClusteringClass:
    def __init__(self):
        super().__init__()
        self.original_to_encoded_columns = {}
        self.initial_columns = []

    def set_data(self):
        filename = filedialog.askopenfilename()
        data = pd.read_excel(filename)
        self.initial_columns = data.columns.tolist()  # Store initial columns
        print("Data is selected")
        return data

    def convert_string_to_numeric(self, data):
        label_encoders = {}
        one_hot_encoders = {}
        for column in data.columns:
            if data[column].dtype == 'object':
                unique_values = data[column].nunique()
                if column == 'Grade':  # Assuming 'Grade' is the only ordinal column
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column].astype(str))
                    label_encoders[column] = le
                elif unique_values < 100:  # Apply one-hot encoding only if unique values are less than 100
                    ohe = OneHotEncoder(sparse_output=False, drop='first')
                    transformed_data = ohe.fit_transform(data[[column]])
                    ohe_df = pd.DataFrame(transformed_data, columns=ohe.get_feature_names_out([column]))
                    data = pd.concat([data.drop(column, axis=1), ohe_df], axis=1)
                    one_hot_encoders[column] = ohe
                    self.original_to_encoded_columns[column] = ohe_df.columns.tolist()
                else:  # Use label encoding for high cardinality columns
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column].astype(str))
                    label_encoders[column] = le
        return data

    def set_feature_column(self, data):
        def on_select():
            self.selected_features = []
            for col, var in checkboxes.items():
                if var.get() == 1:
                    if col in self.original_to_encoded_columns:
                        self.selected_features.extend(self.original_to_encoded_columns[col])
                    else:
                        self.selected_features.append(col)
            root.destroy()

        root = ctk.CTk()
        root.title("Select Feature Columns")

        checkboxes = {}
        for col in self.initial_columns:  # Use initial columns for checkboxes
            var = ctk.IntVar()
            chk = ctk.CTkCheckBox(root, text=col, variable=var)
            chk.pack(anchor='w')
            checkboxes[col] = var

        btn = ctk.CTkButton(root, text="Select", command=on_select)
        btn.pack()

        root.mainloop()

        selected_data = data[self.selected_features]
        selected_data = selected_data.apply(pd.to_numeric, errors='coerce')
        selected_data = selected_data.dropna()

        if selected_data.empty:
            raise ValueError("Selected features resulted in an empty DataFrame after conversion to numeric.")

        print("Feature columns are selected")
        return selected_data

    def set_cluster_count(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        dialog = ctk.CTkInputDialog(text="Enter desired cluster count :", title='Set Cluster Number',
                                    button_fg_color='#01675A', button_hover_color='#AAC980')

        n_cluster = int(dialog.get_input())
        return n_cluster

    def sample_data(self, data, sample_size):
        return data.sample(n=sample_size, random_state=42)

    def plot_silhouette_scores(self, x, method='kmeans', save_dir=None):
        range_n_clusters = [2, 3, 4, 5]
        silhouette_avgs = []

        start_time = time.time()  # Start timing

        x_sampled = self.sample_data(x, 1000)  # Sample 1000 rows

        for n_clusters in range_n_clusters:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            elif method == 'gmm':
                clusterer = GaussianMixture(n_components=n_clusters, random_state=10)
            else:
                raise ValueError("Method must be 'kmeans' or 'gmm'")

            cluster_labels = clusterer.fit_predict(x_sampled)
            silhouette_avg = silhouette_score(x_sampled, cluster_labels)
            silhouette_avgs.append(silhouette_avg)

        end_time = time.time()  # End timing

        plt.figure()  # Ensure a new figure is created
        plt.plot(range_n_clusters, silhouette_avgs, marker='o')
        plt.title(f'Silhouette Scores for {method.upper()}')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.tight_layout()

        # Save the plot in the specified directory
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"silhouette_scores_{method}.png"))

        plt.show()
        plt.close()

    def kmeans_clustering_func(self, data, x, original_data, save_dir):
        self.plot_silhouette_scores(x, method='kmeans', save_dir=save_dir)
        kmeans = KMeans(n_clusters=self.set_cluster_count())
        kmeans.fit(x)
        original_data.loc[x.index, 'Cluster'] = kmeans.labels_
        original_data['Cluster'] = original_data['Cluster'].astype('category')
        original_data.to_excel(os.path.join(save_dir, "kmeans_output.xlsx"))
        print("Kmeans Clustering completed!")
        self.show_clusters(original_data, "kmeans", save_dir)
        return original_data

    def agglomerative_clustering_func(self, data, x, sample_size, original_data, save_dir):
        x_sampled = self.sample_data(x, sample_size)
        agglomerative = AgglomerativeClustering()
        original_data.loc[x_sampled.index, 'Cluster'] = agglomerative.fit_predict(x_sampled)
        original_data['Cluster'] = original_data['Cluster'].astype('category')
        original_data.to_excel(os.path.join(save_dir, "agglomerative_output.xlsx"))
        print("Agglomerative Clustering completed!")
        self.show_clusters(original_data, "agglomerative", save_dir)
        return original_data

    def gmm_clustering_func(self, data, x, sample_size, original_data, save_dir):
        self.plot_silhouette_scores(x, method='gmm', save_dir=save_dir)
        x_sampled = self.sample_data(x, sample_size)
        n_clusters = self.set_cluster_count()
        gmm = GaussianMixture(n_components=n_clusters)
        original_data.loc[x_sampled.index, 'Cluster'] = gmm.fit_predict(x_sampled)
        original_data['Cluster'] = original_data['Cluster'].astype('category')
        original_data.to_excel(os.path.join(save_dir, "gmm_output.xlsx"))
        print("GMM Clustering completed!")
        self.show_clusters(original_data, "gmm", save_dir)
        return original_data

    def show_clusters(self, data, method_name, save_dir):
        # Plot 1
        plt.figure(figsize=(12, 8))
        displot(data=data, x='Cluster', hue="Grade", multiple="stack", aspect=2, height=5)
        plt.title('Amount of Sales in Cluster for Each Grade')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{method_name}_sales_in_cluster_for_each_grade.png"))
        plt.close()

        # Plot 2 - Limit to top 50 categories
        top_categories = data['Kategori'].value_counts().nlargest(50).index
        filtered_data = data[data['Kategori'].isin(top_categories)]
        plt.figure(figsize=(12, 8))
        displot(data=filtered_data, y='Kategori', hue='Cluster', multiple="stack", aspect=1.5, height=10)
        plt.title('Amount of Sales in Categories for Each Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{method_name}_sales_in_categories_for_each_cluster.png"))
        plt.close()

        # Plot 3
        plt.figure(figsize=(12, 8))
        filtered_data = data[data['Perakende'] <= 7000]  # Filter out outliers for plotting
        catplot(data=filtered_data, x='Perakende', y='Cluster', hue='Cluster', kind='violin', aspect=1.5, height=10)
        plt.title('Sales Prices for Each Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{method_name}_sales_prices_for_each_cluster.png"))
        plt.close()

        # Plot 4
        plt.figure(figsize=(12, 8))
        countplot(data=data, y='Lokasyon', hue='Cluster')
        plt.title('Sales Distribution Across Locations and Clusters')
        plt.ylabel('Location')
        plt.xlabel('Number of Sales')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{method_name}_sales_distribution_across_locations_and_clusters.png"))
        plt.close()

        # Plot 5
        plt.figure(figsize=(12, 8))
        countplot(data=data, x='Cinsiyet', hue='Cluster')
        plt.title('Gender Distribution of Sales')
        plt.xlabel('Gender')
        plt.ylabel('Number of Products')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{method_name}_gender_distribution_of_sales.png"))
        plt.close('all')

    def create_save_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(os.getcwd(), f"clustering_results_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def run_all_clustering_methods(self, data):
        original_data = data.copy()  # Keep a copy of the original data with string values
        data = self.convert_string_to_numeric(data)
        x = self.set_feature_column(data)
        save_dir = self.create_save_directory()
        self.kmeans_clustering_func(data.copy(), x, original_data.copy(), save_dir)
        self.agglomerative_clustering_func(data.copy(), x, 30000, original_data.copy(), save_dir)
        self.gmm_clustering_func(data.copy(), x, 30000, original_data.copy(), save_dir)

# Example usage
if __name__ == "__main__":
    clustering_instance = ClusteringClass()
    data = clustering_instance.set_data()
    clustering_instance.run_all_clustering_methods(data)