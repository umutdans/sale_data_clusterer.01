import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from seaborn import displot, catplot, countplot
import openpyxl
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import customtkinter as ctk
from customtkinter import filedialog

matplotlib.use('TkAgg')


class ClusteringClass:
    def __init__(self):
        super().__init__()

    def set_data(self):
        global data
        filename = filedialog.askopenfilename()
        data = pd.read_excel(filename)
        return data

    def convert_string_to_numeric(self, data):
        # Convert string columns to numeric using Label Encoding
        label_encoders = {}
        for column in data.columns:
            if data[column].dtype == 'object':
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                label_encoders[column] = le
        return data

    def set_feature_column(self, data):
        def on_select():
            self.selected_features = [col for col, var in checkboxes.items() if var.get() == 1]
            root.destroy()

        root = ctk.CTk()
        root.title("Select Feature Columns")

        checkboxes = {}
        for col in data.columns:
            var = ctk.IntVar()
            chk = ctk.CTkCheckBox(root, text=col, variable=var)
            chk.pack(anchor='w')
            checkboxes[col] = var

        btn = ctk.CTkButton(root, text="Select", command=on_select)
        btn.pack()

        root.mainloop()

        selected_data = data[self.selected_features]
        print("Selected Features:", self.selected_features)  # Debugging information
        print("Selected Data before conversion:\n", selected_data.head())  # Debugging information

        # Ensure that the selected columns are numeric
        selected_data = selected_data.apply(pd.to_numeric, errors='coerce')
        selected_data = selected_data.dropna()

        print("Selected Data after conversion:\n", selected_data.head())  # Debugging information

        if selected_data.empty:
            raise ValueError("Selected features resulted in an empty DataFrame after conversion to numeric.")

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

    def plot_silhouette_scores(self, data, method='kmeans'):
        x = self.set_feature_column(data)
        range_n_clusters = [2, 3, 4, 5]
        silhouette_avgs = []

        for n_clusters in range_n_clusters:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            elif method == 'gmm':
                clusterer = GaussianMixture(n_components=n_clusters, random_state=10)
            else:
                raise ValueError("Method must be 'kmeans' or 'gmm'")

            cluster_labels = clusterer.fit_predict(x)
            silhouette_avg = silhouette_score(x, cluster_labels)
            silhouette_avgs.append(silhouette_avg)
            print(f"For n_clusters = {n_clusters}, the average silhouette score is: {silhouette_avg}")

        plt.figure()
        plt.plot(range_n_clusters, silhouette_avgs, marker='o')
        plt.title(f'Silhouette Scores for {method.upper()}')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()

    def kmeans_clustering_func(self, data, x, original_data):
        self.plot_silhouette_scores(data, method='kmeans')
        kmeans = KMeans(n_clusters=self.set_cluster_count())
        kmeans.fit(x)
        original_data.loc[x.index, 'Cluster'] = kmeans.labels_
        original_data['Cluster'] = original_data['Cluster'].astype('category')
        original_data.to_excel("kmeans_output.xlsx")
        print("Kmeans Clustering completed!")
        self.show_clusters(original_data, "kmeans")
        return original_data

    def agglomerative_clustering_func(self, data, x, sample_size, original_data):
        x_sampled = self.sample_data(x, sample_size)
        agglomerative = AgglomerativeClustering()
        original_data.loc[x_sampled.index, 'Cluster'] = agglomerative.fit_predict(x_sampled)
        original_data['Cluster'] = original_data['Cluster'].astype('category')
        original_data.to_excel("agglomerative_output.xlsx")
        print("Agglomerative Clustering completed!")
        self.show_clusters(original_data, "agglomerative")
        return original_data

    def gmm_clustering_func(self, data, x, sample_size, original_data):
        self.plot_silhouette_scores(data, method='gmm')
        x_sampled = self.sample_data(x, sample_size)
        n_clusters = self.set_cluster_count()
        gmm = GaussianMixture(n_components=n_clusters)
        original_data.loc[x_sampled.index, 'Cluster'] = gmm.fit_predict(x_sampled)
        original_data['Cluster'] = original_data['Cluster'].astype('category')
        original_data.to_excel("gmm_output.xlsx")
        print("GMM Clustering completed!")
        self.show_clusters(original_data, "gmm")
        return original_data

    def show_clusters(self, data, method_name):
        print(matplotlib.get_backend())

        # Plot 1
        displot(data=data, x='Cluster', hue="Grade", multiple="stack", aspect=2, height=5)
        plt.title('Amount of Sales in Cluster for Each Grade')
        plt.savefig(f"{method_name}_sales_in_cluster_for_each_grade.png")
        plt.close()

        # Plot 2
        displot(data=data, y='Category', hue="Cluster", multiple="stack", aspect=1.5, height=10)
        plt.title('Amount of Sales in Categories for Each Cluster')
        plt.savefig(f"{method_name}_sales_in_categories_for_each_cluster.png")
        plt.close()

        # Plot 3
        catplot(data=data, x='Perakende', y='Cluster', hue='Grade', kind='violin', aspect=1.5, height=10)
        plt.title('Sales Prices for Each Grade')
        plt.savefig(f"{method_name}_sales_prices_for_each_grade.png")
        plt.close()

        # Plot 4
        plt.figure()
        countplot(data=data, y='Lokasyon', hue='Cluster')
        plt.title('Sales Distribution Across Locations and Clusters')
        plt.ylabel('Location')
        plt.xlabel('Number of Sales')
        plt.legend(title='Cluster')
        plt.savefig(f"{method_name}_sales_distribution_across_locations_and_clusters.png")
        plt.close()

        # Plot 5
        plt.figure()
        countplot(data=data, x='Cinsiyet', hue='Cluster')
        plt.title('Gender Distribution of Sales')
        plt.xlabel('Gender')
        plt.ylabel('Number of Products')
        plt.legend(title='Cluster')
        plt.savefig(f"{method_name}_gender_distribution_of_sales.png")
        plt.close()

    def run_all_clustering_methods(self, data):
        original_data = data.copy()  # Keep a copy of the original data with string values
        data = self.convert_string_to_numeric(data)
        x = self.set_feature_column(data)
        self.kmeans_clustering_func(data.copy(), x, original_data.copy())
        self.agglomerative_clustering_func(data.copy(), x, 1000, original_data.copy())
        self.gmm_clustering_func(data.copy(), x, 1000, original_data.copy())


# Example usage
if __name__ == "__main__":
    clustering_instance = ClusteringClass()
    data = clustering_instance.set_data()
    clustering_instance.run_all_clustering_methods(data)