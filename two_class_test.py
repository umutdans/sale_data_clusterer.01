from pandas import read_excel
from sklearn.cluster import KMeans
from seaborn import displot, catplot, histplot, cubehelix_palette
import openpyxl
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import customtkinter
from customtkinter import filedialog
import threading
matplotlib.use('TkAgg')


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Clustering App")
        customtkinter.set_appearance_mode("light")
        self.configure(fg_color='#ffffff')

        my_font = customtkinter.CTkFont(family="Celias", size=14)

        self.frame = customtkinter.CTkFrame(master=self, fg_color='#ffffff')
        self.frame.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="nsw")

        self.button_file = customtkinter.CTkButton(master=self.frame, text="Choose File", fg_color='#01675A',
                                                   hover_color='#AAC980', text_color='#ffffff',
                                                   command=lambda: boo.set_data(), font=my_font)
        self.button_file.grid(row=0, column=0, padx=20, pady=20)

        self.button_2 = customtkinter.CTkButton(master=self.frame, text="Elbow Method", fg_color='#01675A',
                                                hover_color='#AAC980', text_color='#ffffff', font=my_font,
                                                command=lambda: boo.elbow_method(data))
        self.button_2.grid(row=1, column=0, padx=20, pady=20)

        self.button_3 = customtkinter.CTkButton(master=self.frame, text="Cluster !", fg_color='#01675A',
                                                hover_color='#AAC980', text_color='#ffffff', font=my_font,
                                                command=lambda: boo.clustering_func(data))
        self.button_3.grid(row=2, column=0, padx=20, pady=20)

        self.button_4 = customtkinter.CTkButton(master=self.frame, text="Show Figures", fg_color='#01675A',
                                                hover_color='#AAC980', text_color='#ffffff', font=my_font,
                                                command=lambda: boo.show_clusters(data))
        self.button_4.grid(row=3, column=0, padx=20, pady=20)



class ClusteringClass:
    def __init__(self):
        super().__init__()

    def set_data(self):
        global data
        filename = filedialog.askopenfilename()

        data = read_excel(filename)
        print("Data is selected !")
        return data

    def set_feature_column(self, data):
        grade_mapping = {"SARI ETIKETLI": 0, "MAKUL": 1, "IYI": 2, "HARIKA": 3}
        data['Grade_num'] = data['Grade'].apply(lambda x: grade_mapping[x] if x in grade_mapping else x)

        top_20_categories = data['Category'].value_counts().nlargest(20).index

        data['top20_categories'] = data['Category'].apply(lambda x: x if x in top_20_categories else 'DIGER')

        
        feature_columns = (['Grade_num', 'Perakende']
                            + [col for col in data.columns if 'Brand_' in col or 'Category_' in col])

        y = data[feature_columns]  

        return y

    def set_cluster_count(self):
        
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")

        dialog = customtkinter.CTkInputDialog(text="Enter desired cluster count :", title='Set Cluster Number',
                                              button_fg_color='#01675A', button_hover_color='#AAC980')
  
        n_cluster = int(dialog.get_input())
        return n_cluster

    def clustering_func(self, data):
        def run_clustering():
            x = self.set_feature_column(data)
            kmeans = KMeans(n_clusters=self.set_cluster_count())
            kmeans.fit(x)
            data['Cluster'] = kmeans.labels_
            data['Cluster'] = data['Cluster'].astype('category')
            data.to_excel("output.xlsx")
            print("Clustering completed!")

        thread = threading.Thread(target=run_clustering)
        thread.start()

    def show_clusters(self, data):

        print(matplotlib.get_backend())

        displot(data=data, x='Cluster', multiple="stack", aspect=2, height=5)
        plt.title('Küme Başına Düşen Satış Miktarı')
        plt.xticks(ticks=range(data['Cluster'].nunique()), labels=data['Cluster'].cat.categories)
        plt.xlabel('Kümeler')
        plt.ylabel('Satış Sayısı')

        displot(data=data, y='top20_categories', hue="Cluster", palette='YlGn', multiple="stack", aspect=1.5, height=10)
        plt.title('Kategoriler ve Kümeler Arası Satış Dağılımı')
        plt.ylabel('Kategoriler')
        plt.xlabel('Satış Sayısı')

        # displot(data=data ,x='Perakende',y='Sales Time',hue='Cluster',aspect=2,height=5)
        # plt.title('Sales Price vs Sales Time')/

        catplot(data=data, x='Cluster', y='Perakende', kind='violin', palette='YlGn', aspect=1.5, height=10)
        plt.title('Kümeler Arası Satış Fiyatları Dağılımı')
        plt.xticks(ticks=range(data['Cluster'].nunique()), labels=data['Cluster'].cat.categories)
        plt.xlabel('Kümeler')
        plt.ylabel('Satış Fiyatı')

        plt.figure()
        histplot(data=data, y='Lokasyon', hue='Cluster', palette='YlGn', multiple="stack")
        plt.title('Lokasyonlar ve Kümeler Arası Satış Dağılımı')
        plt.xlabel('Satış Sayısı')
        plt.ylabel('Lokasyonlar')

        plt.figure()
        histplot(data=data, x='Cinsiyet', hue='Cluster', palette='YlGn', multiple="stack")
        plt.title('Cinsiyetler ve Kümeler Arası Satış Dağılımı')
        plt.xlabel('Cinsiyet')
        plt.ylabel('Satış Miktarı')



        plt.show()

    def elbow_method(self, data):

        x = self.set_feature_column(data)
        sse = {}
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, max_iter=1000).fit(x)
            sse[k] = kmeans.inertia_

        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Number of cluster")
        plt.ylabel("SSE")
        plt.show()

boo = ClusteringClass()
app = App()
app.mainloop()
