import pygame
import sys
import copy
import pygame.freetype
import heapq
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gensim.models import FastText
from gensim import corpora
from sklearn.preprocessing import MinMaxScaler

# Inisialisasi Pygame
pygame.init()

# Warna
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)  # Warna untuk kota tujuan

# Ukuran layar
WIDTH, HEIGHT = 1500, 800
SCREEN_SIZE = (WIDTH, HEIGHT)

# Kota-kota dan koneksi antara kota-kota
cities = {
    'A': [(100.0, 100.0), ['E', 'D', 'B']],
    'B': [(300.0, 100.0), ['E', 'A', "C"]],
    'C': [(500.0, 100.0), ['J', 'K', 'F']],
    'D': [(100.0, 300.0), ['G', 'H', 'A']],
    'E': [(300.0, 300.0), ['B', 'H', 'G', 'A']],
    'F': [(500.0, 300.0), ['C', 'H', 'L','I']],
    'G': [(100.0, 500.0), ['D', 'H', 'E']],
    'H': [(300.0, 500.0), ['D', 'E', 'F', 'G']],
    'I': [(500.0, 500.0), ['L', 'K', "F"]],
    'J': [(700.0, 100.0), ['C', 'K', 'M']],
    'K': [(700.0, 300.0), ['I', 'C', 'M', 'O', 'J']],
    'L': [(700.0, 500.0), ['F', 'I', 'O']],
    'M': [(900.0, 100.0), ['J', 'K', 'N', 'P']],
    'N': [(900.0, 300.0), ['M', 'Q', 'R', 'P']],
    'O': [(900.0, 500.0), ['K', 'R', 'L']],
    'P': [(1100.0, 100.0), ['Q', 'M', 'N', 'S']],
    'Q': [(1100.0, 300.0), ['N', 'R', 'P', 'T', 'S']],
    'R': [(1100.0, 500.0), ['Q', 'O', 'N','T']],
    'S': [(1300.0, 100.0), ['P', 'Q', 'T']],
    'T': [(1300.0, 300.0), ['Q', 'S','R']]
}
current_city = 'A'
new_character = 'A'


#Bryan D
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "She sells seashells by the seashore",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood",
    "Peter Piper picked a peck of pickled peppers",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "All that glitters is not gold",
    "Birds of a feather flock together",
    "A picture is worth a thousand words",
    "Actions speak louder than words",
    "The early bird catches the worm",
    "Beauty is in the eye of the beholder",
    "Better late than never"
]

# Buat Corpus nya
tokenized_corpus = [[token.lower() for token in doc.split()] for doc in sentences]
# Buat Dictionary nya
dictionary = corpora.Dictionary(tokenized_corpus)

#Membuat dan Training Fast Text Model
model = FastText(vector_size=100, window=3, min_count=1)
model2 = FastText(vector_size=100, window=3, min_count=1)
model.build_vocab(corpus_iterable=tokenized_corpus)
model.train(corpus_iterable=tokenized_corpus, total_examples=len(tokenized_corpus), epochs=10)
model2.build_vocab(corpus_iterable=tokenized_corpus)
model2.train(corpus_iterable=tokenized_corpus, total_examples=len(tokenized_corpus), epochs=5)
wv = model.wv
wv2 = model2.wv
#Untuk menyimpan model
model.save("fasttext.model")


disabled_city = None
disabled_city2 = None
saved_city_data1 = {}
saved_city_data2 = {}
disabled_relations = {}
disabled_relations2 = {} #player1
# Data cuaca
data = {
    'suhu': [30, 22, 25, 28, 32, 24, 31, 23, 27, 29, 33, 26, 30, 25, 28, 27, 30, 22, 23, 24],
    'kelembaban': [80, 65, 70, 85, 90, 60, 82, 66, 73, 88, 91, 63, 78, 70, 76, 84, 81, 68, 75, 62],
    'tekanan': [1012, 1010, 1013, 1009, 1011, 1014, 1010, 1008, 1012, 1011, 1013, 1015, 1012, 1009, 1011, 1013, 1012, 1010, 1011, 1014],
    'angin': [5, 3, 4, 6, 7, 2, 6, 3, 5, 4, 8, 2, 5, 4, 6, 3, 7, 2, 5, 4],
    'hujan': [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]
}
# Memuat data awal ke dalam DataFrame
df= pd.DataFrame(data)

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 10))

# Normalisasi data
df_normalized = df.copy()
df_normalized[['suhu', 'kelembaban', 'tekanan', 'angin']] = scaler.fit_transform(df[['suhu', 'kelembaban', 'tekanan', 'angin']])

print("Data Awal sebelum Normalisasi:\n", df)
print("\nData setelah Normalisasi:\n", df_normalized)
# Target
y = df_normalized['hujan']

# Fitur
X = df_normalized.drop('hujan', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Inisialisasi model KNN dengan 3 tetangga
knn_model = KNeighborsClassifier(n_neighbors=2)
#knn_model = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn_model2 = KNeighborsClassifier(n_neighbors=1)
#knn_model2 = KNeighborsClassifier(n_neighbors=3, weights='distance')

# Melatih model dengan data pelatihan
knn_model.fit(X_train, y_train)
knn_model2.fit(X_train, y_train)

# # Prediksi pada data pengujian
y_pred = knn_model.predict(X_test)

# # Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi pada data pengujian model 1: {accuracy}")

# # Menampilkan matriks kebingungan
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriks Kebingungan model 1:\n", conf_matrix)

# # Menampilkan laporan klasifikasi
class_report = classification_report(y_test, y_pred)
print("Laporan Klasifikasi model 1:\n", class_report)

# # Prediksi pada data pengujian
y_pred = knn_model2.predict(X_test)

# # Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi pada data pengujian model 2: {accuracy}")

# # Menampilkan matriks kebingungan
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriks Kebingungan model 2:\n", conf_matrix)

# # Menampilkan laporan klasifikasi
class_report = classification_report(y_test, y_pred)
print("Laporan Klasifikasi model 2:\n", class_report)

def tebak_cuaca(screen,player):
    # Menentukan batas atas dan bawah untuk nilai random berdasarkan data asli
    suhu_min, suhu_max = df['suhu'].min(), df['suhu'].max()
    kelembaban_min, kelembaban_max = df['kelembaban'].min(), df['kelembaban'].max()
    tekanan_min, tekanan_max = df['tekanan'].min(), df['tekanan'].max()
    angin_min, angin_max = df['angin'].min(), df['angin'].max()
        
    # Membuat data baru yang dirandom
    X_new = [[
    round(np.random.uniform(suhu_min, suhu_max)),
    round(np.random.uniform(kelembaban_min, kelembaban_max)),
    round(np.random.uniform(tekanan_min, tekanan_max)),
    round(np.random.uniform(angin_min, angin_max))
    ]]
    df_prediksi = pd.DataFrame(X_new, columns=['suhu', 'kelembaban', 'tekanan', 'angin'])
    print("Tebak awal:")
    print(df_prediksi)
    df_prediksi_normalized = df_prediksi.copy()
    df_prediksi_normalized[['suhu', 'kelembaban', 'tekanan' , 'angin']] = scaler.transform(df_prediksi[['suhu', 'kelembaban', 'tekanan', 'angin']])
    
    if(player==1):
        # Menampilkan pertanyaan
        screen.fill(WHITE)
        font_question = pygame.font.SysFont(None, 30)
        font = pygame.freetype.SysFont(None, 18)
        
        pygame.freetype.init()
        # Convert data to string
        data_strings = []
        for key, values in data.items():
            data_strings.append(f"{key}: {', '.join(map(str, values))}")
        
        # Display each line of text in black
        y_offset = 25
        for line in data_strings:
            font.render_to(screen, (250, y_offset), line, (0, 0, 0))
            y_offset += 30  # Move to the next line
        pygame.display.flip()
        suhu, kelembaban, tekanan, angin = X_new[0]
        data_text = font_question.render(f"Data baru (random):\nSuhu: {suhu}\nKelembaban: {kelembaban}\nTekanan: {tekanan}\nAngin: {angin}", True, BLACK)
        question_text = font_question.render("Apakah dari data di atas maka akan hujan atau tidak?", True, BLACK)
        screen.blit(data_text, (20, 20))
        screen.blit(question_text, (100, 200))
        hujan = pygame.Rect(150, 400, 200, 50)
        tidakHujan = pygame.Rect(450, 400, 200, 50)
        draw_button(screen, hujan, BLACK, "Hujan", WHITE)
        draw_button(screen, tidakHujan, BLACK, "Tidak Hujan", WHITE)
        pygame.display.flip()
        # Menggunakan model untuk memprediksi data baru
        prediksi = knn_model.predict(df_prediksi_normalized)
        print("Prediksi:")
        print(prediksi)
        jawaban = ""  # Inisialisasi jawaban
        loop=True
        while loop:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if hujan.collidepoint(mouse_pos):
                        jawaban=1
                        loop=False
                    elif tidakHujan.collidepoint(mouse_pos):
                        jawaban=0
                        loop=False
                    # Update input text di layar
                    screen.fill(WHITE)
                    screen.blit(data_text, (20, 20))
                    screen.blit(question_text, (100, 200))
                    pygame.display.flip()
                    # Display each line of text in black
                    y_offset = 25
                    for line in data_strings:
                        font.render_to(screen, (250, y_offset), line, (0, 0, 0))
                        y_offset += 30  # Move to the next line
                    pygame.display.flip()
                    hujan = pygame.Rect(150, 400, 200, 50)
                    tidakHujan = pygame.Rect(450, 400, 200, 50)
                    draw_button(screen, hujan, BLACK, "Hujan", WHITE)
                    draw_button(screen, tidakHujan, BLACK, "Tidak Hujan", WHITE)
                    pygame.display.flip()
        # Menampilkan hasil tebakan
        hasil= show_result(screen, jawaban, prediksi)
    else:
        prediksi1 = knn_model.predict(df_prediksi_normalized)
        prediksi2 = knn_model2.predict(df_prediksi_normalized)
        if(prediksi1==prediksi2):
            hasil="benar"
        else:
            hasil="salah"
            print("Bot salah")
    
    return hasil

def show_result(screen, jawaban, prediksi):
    screen.fill(WHITE)
    font_result = pygame.font.SysFont(None, 30)
    simpn=''
    if prediksi==1:
        simpn="Hujan"
    else:
        simpn="Tidak Hujan"
    
    if int(jawaban) == prediksi:
        result_text = font_result.render(f"Tebakan Anda benar! Prediksi: {simpn}", True, BLACK)
        hasil="benar"
    else:
        result_text = font_result.render(f"Tebakan Anda salah! Prediksi: {simpn}", True, BLACK)
        hasil="salah"
    screen.blit(result_text, (100, 100))
    pygame.display.flip()
    pygame.time.wait(2000)  # Tampilkan hasil selama 3 detik
    return hasil

def show_resultw2f(screen, jawaban, bnr):
    screen.fill(WHITE)
    font_result = pygame.font.SysFont(None, 30)
    if jawaban == "benar":
        result_text = font_result.render(f"Tebakan Anda benar! Prediksi: {bnr}", True, BLACK)
        hasil="benar"
    else:
        result_text = font_result.render(f"Tebakan Anda salah! Prediksi: {bnr}", True, BLACK)
        hasil="salah"
    screen.blit(result_text, (100, 100))
    
    pygame.display.flip()
    pygame.time.wait(2000)  # Tampilkan hasil selama 3 detik
    return hasil


# Fungsi untuk menggambar kota-kota dan koneksi antar kota
def draw_cities(screen, winner=None, current_city=None, show_exit_button=False):
    for city, (pos, connections) in cities.items():
        color = YELLOW if city == 'T' else RED  # Mengubah warna kota "T" menjadi kuning
        pygame.draw.circle(screen, color, pos, 30)
        font = pygame.font.SysFont(None, 24)
        text = font.render(city, True, BLACK)
        text_rect = text.get_rect(center=pos)
        screen.blit(text, text_rect)
        for connected_city in connections:
            if connected_city in cities:
                pygame.draw.line(screen, BLACK, pos, cities[connected_city][0])
    
    if winner:
        # Menampilkan pop-up window dengan pemenang dan tombol restart/exit
        popup_rect = pygame.Rect(300, 200, 600, 300)
        pygame.draw.rect(screen, WHITE, popup_rect)
        pygame.draw.rect(screen, BLACK, popup_rect, 3)

        font = pygame.font.SysFont(None, 36)
        winner_text = font.render(f"Pemenang: {winner}", True, BLACK)
        winner_text_rect = winner_text.get_rect(center=(600, 270))
        screen.blit(winner_text, winner_text_rect)

        restart_button_rect = pygame.Rect(400, 350, 200, 50)
        exit_button_rect = pygame.Rect(600, 350, 200, 50)
        draw_button(screen, restart_button_rect, BLACK, "Restart", WHITE)
        draw_button(screen, exit_button_rect, BLACK, "Exit", WHITE)

        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if restart_button_rect.collidepoint(mouse_pos):
                        return "restart"
                    elif exit_button_rect.collidepoint(mouse_pos):
                        return "exit"
   
# Fungsi untuk menggambar karakter
def draw_character(screen, current_city, color):
    pygame.draw.circle(screen, color, cities[current_city][0], 10)

# Fungsi DFS
def dfs(graph, start, goal):
    stack = [(start, [start])]  # stack untuk menyimpan (current_node, path)
    visited = set()  # set untuk menyimpan node yang sudah dikunjungi
    
    while stack:
        (vertex, path) = stack.pop()  # ambil node dan jalur dari stack
        if vertex not in visited:
            if vertex == goal:  # jika kita mencapai node tujuan
                return path  # kembalikan jalur yang diambil
            
            visited.add(vertex)  # tambahkan node ke dalam set visited
            for neighbor in graph[vertex][1]:  # iterasi melalui tetangga
                if neighbor not in visited and neighbor in cities:  # jika tetangga belum dikunjungi
                    stack.append((neighbor, path + [neighbor]))  # tambahkan tetangga dan jalurnya ke stack

    return None  # jika tidak ada jalur yang ditemukan
# Fungsi untuk membuat tombol
def draw_button(screen, rect, color, text, text_color):
    pygame.draw.rect(screen, color, rect)
    font = pygame.font.SysFont(None, 24)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)
    
    
def tutupJalan(screen, player, path):
    if player==1:
        screen.fill(WHITE)
        font_question = pygame.font.SysFont(None, 30)
        bisa=[item[0] for item in cities]
        bisa2=[]
        for item in bisa:
            if item!=current_city and item!=new_character and item!='T':
                bisa2.append(item)
        x_offset = 0
        
        for city, (pos, connections) in cities.items():
            color = YELLOW if city == 'T' else RED  # Mengubah warna kota "T" menjadi kuning
            temppos = [0,0]
            temppos[0] = pos[0]/2 + 750
            temppos[1] = pos[1]/2 + 50 
            
            pygame.draw.circle(screen, color, temppos, 30)
            font = pygame.font.SysFont(None, 24)
            text = font.render(city, True, BLACK)
            text_rect = text.get_rect(center=temppos)
            screen.blit(text, text_rect)
            for connected_city in connections:
                if connected_city in cities:
                    temppos2 = [0,0]
                    temppos2[0] = (cities[connected_city][0][0]/2 + 750)
                    temppos2[1] = (cities[connected_city][0][1]/2 + 50)
                    pygame.draw.line(screen, BLACK, temppos, temppos2)
        
        posisi = font_question.render("Posisi Player: " + current_city, True, BLACK)
        screen.blit(posisi, (800,20))
        pygame.display.flip()
        
        posisi2 = font_question.render("Posisi Musuh: " + new_character, True, BLACK)
        screen.blit(posisi2, (1000,20))
        pygame.display.flip()
        
        text = font_question.render("Kota yang bisa dinon-aktifkan" , True, BLACK)
        screen.blit(text, (40, 20))
        pygame.display.flip()
        filtered_city_names = [city for city in cities.keys() if city != 'T' and city !=current_city and city !=new_character]
        city_names = ' '.join(filtered_city_names)
        tampilkota=font_question.render(city_names,True,BLACK)
        screen.blit(tampilkota, (40, 60))
        pygame.display.flip()
        data_text = font_question.render("Masukkan kota yang ingin dinon-aktifkan", True, BLACK)
        screen.blit(data_text, (40, 100))
        pygame.display.flip()
        input_text = ""
        error_message = ""
        pygame.draw.rect(screen, BLACK, (40, 300, 600, 50), 2)
        pygame.display.flip()
        
        waiting_for_input = True
        while waiting_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            input_string = input_text.upper()
                            global disabled_city, disabled_relations2
                            if input_string in cities and input_string!=new_character and input_string!=current_city and input_string!="T":
                                city_to_disable=input_string
                                if disabled_city is not None:
                                    cities[disabled_city[0]] = disabled_city[1]
                                    # Tambahkan hubungan kota yang dinonaktifkan kembali
                                    for city, relations in disabled_relations2.items():
                                        if city in cities:
                                            cities[city][1].extend(relations)
                                    disabled_city = None
                                    disabled_relations2 = {}
                                
                                # Simpan informasi tentang kota yang dinonaktifkan sementara
                                disabled_city = (city_to_disable, cities.pop(city_to_disable))

                                # Hapus semua hubungan yang terkait dengan kota yang dinonaktifkan
                                for city, city_info in cities.items():
                                    if city_to_disable in city_info[1]:
                                        if city not in disabled_relations2:
                                            disabled_relations2[city] = []
                                        disabled_relations2[city].append(city_to_disable)
                                        city_info[1].remove(city_to_disable)
                                waiting_for_input = False
                                print("Kota player hapus:")
                                print(city_to_disable)
                            else:
                                input_text = ""  # Reset input_text jika input tidak valid
                                error_message = "Coba lagi."
                        elif event.key == pygame.K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            input_text += event.unicode
                        # Update input text di layar
                        screen.fill(WHITE)
                        screen.blit(data_text, (40, 100))
                        pygame.draw.rect(screen, BLACK, (40, 300, 600, 50), 2)
                        input_surface = pygame.font.SysFont(None, 36).render(input_text, True, BLACK)
                        screen.blit(input_surface, (110, 310))
                        pygame.display.flip()
                        tampilkota=font_question.render(city_names,True,BLACK)
                        screen.blit(tampilkota, (40, 60))
                        pygame.display.flip()
                        
                        screen.blit(posisi, (800,20))
                        pygame.display.flip()
                        
                        screen.blit(text, (40, 20))
                        screen.blit(posisi2, (1000,20))
                        pygame.display.flip()
        
                        error_font = pygame.font.SysFont(None, 24)
                        error_surface = error_font.render(error_message, True, RED)
                        screen.blit(error_surface, (20, HEIGHT - 80))
                        pygame.display.flip()
                        
                        for city, (pos, connections) in cities.items():
                            color = YELLOW if city == 'T' else RED  # Mengubah warna kota "T" menjadi kuning
                            temppos = [0,0]
                            temppos[0] = pos[0]/2 + 750
                            temppos[1] = pos[1]/2 + 50 
                            
                            pygame.draw.circle(screen, color, temppos, 30)
                            font = pygame.font.SysFont(None, 24)
                            text2 = font.render(city, True, BLACK)
                            text_rect = text2.get_rect(center=temppos)
                            screen.blit(text2, text_rect)
                            for connected_city in connections:
                                if connected_city in cities:
                                    temppos2 = [0,0]
                                    temppos2[0] = (cities[connected_city][0][0]/2 + 750)
                                    temppos2[1] = (cities[connected_city][0][1]/2 + 50)
                                    pygame.draw.line(screen, BLACK, temppos, temppos2)
                        pygame.display.flip()
    

    elif player==2:
        global disabled_city2, disabled_relations
        # Jika ada kota yang sebelumnya dinonaktifkan, tambahkan kembali ke daftar
        if disabled_city2 is not None:
            cities[disabled_city2[0]] = disabled_city2[1]
            # Tambahkan hubungan kota yang dinonaktifkan kembali
            for city, relations in disabled_relations.items():
                if city in cities:
                    cities[city][1].extend(relations)
            disabled_city2 = None
            disabled_relations = {}

        # Buat daftar kota yang tersedia
        available_cities = list(cities.keys())
        print("Available cities before disabling:", available_cities)
        if path:
            city_to_disable = path[1]
        loop=True
        while loop:
            if path:
                print("Masuk ada path")
                if path[1]=="T":
                    while(True):
                        random_city = random.choice(list(cities.keys()))
                        if random_city not in [current_city, new_character, "T"]:
                            print("Disabling city:", random_city)
                            disabled_city2 = (random_city, cities.pop(random_city))
                            city_to_disable = random_city
                            loop = False
                            break
                else: 
                    if(city_to_disable!=new_character and city_to_disable!=current_city and city_to_disable!="T"):
                        print("Masuk if")
                        disabled_city2 = (city_to_disable, cities.pop(city_to_disable))
                        print("City to disable:", city_to_disable)
                        loop=False
                    else:
                        while(True):
                            random_city = random.choice(list(cities.keys()))
                            if random_city not in [current_city, new_character, "T"]:
                                print("Disabling city:", random_city)
                                disabled_city2 = (random_city, cities.pop(random_city))
                                city_to_disable = random_city
                                loop = False
                                break
                    loop = False
            else:
                print("Masuk No Path")
                while(True):
                    random_city = random.choice(list(cities.keys()))
                    if random_city not in [current_city, new_character, "T"]:
                        print("Disabling city:", random_city)
                        disabled_city2 = (random_city, cities.pop(random_city))
                        city_to_disable = random_city
                        loop = False
                        break
        # Hapus semua hubungan yang terkait dengan kota yang dinonaktifkan
        for city, city_info in cities.items():
            if city_to_disable in city_info[1]:
                if city not in disabled_relations:
                    disabled_relations[city] = []
                disabled_relations[city].append(city_to_disable)
                city_info[1].remove(city_to_disable)
    
    else:
        # global disabled_city2, disabled_relations, disabled_city, disabled_relations2
        if disabled_city is not None:
            cities[disabled_city[0]] = disabled_city[1]
            # Tambahkan hubungan kota yang dinonaktifkan kembali
            for city, relations in disabled_relations2.items():
                if city in cities:
                    cities[city][1].extend(relations)
                disabled_city = None
                disabled_relations2 = {}
        if disabled_city2 is not None:
            cities[disabled_city2[0]] = disabled_city2[1]
            # Tambahkan hubungan kota yang dinonaktifkan kembali
            for city, relations in disabled_relations.items():
                if city in cities:
                    cities[city][1].extend(relations)
            disabled_city2 = None
            disabled_relations = {}
    print(cities)  
      
def w2f(screen, player):
    output=""
    random_item = random.choice(list(dictionary.token2id.items()))
    arr=wv.most_similar(random_item[0])
    print("pilihan jawaban")
    print(arr)
    datajwback = list(arr)  # Ensure it's a list if it's not already
    
    random.shuffle(datajwback)
    arrword=[item[0] for item in datajwback]
    arrnilai=[item[1] for item in datajwback]
    
    max=0
    index=0
    masuk=True
    while masuk:
        if index==4:
            masuk=False
        elif arrnilai[index]>max:
            max=arrnilai[index]
            jwbbnr=arrword[index]
        else:
            index=index+1
    print("jawaban benar")
    print(jwbbnr)
    
    if(player==1):
        # Menampilkan pertanyaan
        font = pygame.freetype.SysFont(None, 17)
        screen.fill(WHITE)
        font_question = pygame.font.SysFont(None, 25)
        y_offset = 50
        for sentence in sentences:
            font.render_to(screen, (100, y_offset), sentence, (0, 0, 0))
            y_offset += 30  # Move to the next line
        pygame.display.flip()
        random_item2 = arrword[0]
        random_item3 = arrword[1]
        random_item4 = arrword[2]
        random_item5 = arrword[3]
        soal=random_item
        
        question_text = font_question.render(f"Dari data diatas, maka kata {soal[0]} akan paling mendekati dengan kata: \nA. {random_item2}\nB. {random_item3}\nC. {random_item4}\nD. {random_item5}", True, BLACK)
        screen.blit(question_text, (100, 450))
        pygame.display.flip()
        a = pygame.Rect(150, 570, 200, 50)
        b = pygame.Rect(450, 570, 200, 50)
        c = pygame.Rect(150, 670, 200, 50)
        d = pygame.Rect(450, 670, 200, 50)
        draw_button(screen, a, BLACK, random_item2, WHITE)
        draw_button(screen, b, BLACK, random_item3, WHITE)
        draw_button(screen, c, BLACK, random_item4, WHITE)
        draw_button(screen, d, BLACK, random_item5, WHITE)
        pygame.display.flip()
        
        jawaban = ""  # Inisialisasi jawaban
        loop=True
        while loop:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if a.collidepoint(mouse_pos):
                        jawaban=random_item2
                        loop=False
                    elif b.collidepoint(mouse_pos):
                        jawaban=random_item3
                        loop=False
                    elif c.collidepoint(mouse_pos):
                        jawaban=random_item4
                        loop=False
                    elif d.collidepoint(mouse_pos):
                        jawaban=random_item5
                        loop=False
                    # Update input text di layar
                    screen.fill(WHITE)
                    screen.blit(question_text, (100, 300))
                    y_offset = 50
                    for sentence in sentences:
                        font.render_to(screen, (100, y_offset), sentence, (0, 0, 0))
                        y_offset += 30  # Move to the next line
                    pygame.display.flip()
                    draw_button(screen, a, BLACK, random_item2, WHITE)
                    draw_button(screen, b, BLACK, random_item3, WHITE)
                    draw_button(screen, c, BLACK, random_item4, WHITE)
                    draw_button(screen, d, BLACK, random_item5, WHITE)
                    pygame.display.flip()
                    if jawaban==jwbbnr:
                        output="benar"
                    else:
                        output="salah"
        # Menampilkan tebakan
        show_resultw2f(screen, output, jwbbnr)
        
    else:
        arr2=wv2.most_similar(random_item[0])
        print("hasil model 2")
        print(arr2)
        arrword2=[item[0] for item in arr2]
        arrnilai2=[item[1] for item in arr2]
        countisi=0
        index3=0
        jwbword=[]
        jwbnil=[]
        
        jwbbot=""
        while index3<4:
            count=0
            for item in arrword2:
                if item==arrword[0]:
                    jwbword.append(item)
                    jwbnil.append(arrnilai2[count])
                elif item==arrword[1]:
                    jwbword.append(item)
                    jwbnil.append(arrnilai2[count])
                elif item==arrword[2]:
                    jwbword.append(item)
                    jwbnil.append(arrnilai2[count])
                elif item==arrword[3]:
                    jwbword.append(item)
                    jwbnil.append(arrnilai2[count])
                count=count+1
            index3=index3+1
            
        print("pilihan bot")
        print(jwbword)
        
        for isi in jwbnil:
            if isi>max:
                jwbbotnil=isi
                jwbbot=jwbword[countisi]
            countisi=countisi+1
        if(jwbbnr==jwbbot):
            output="benar"
        else:
            output="salah"
            print("Bot salah")
        print(jwbbot)
    
    return output

def heuristic(node1, node2):
    # Menghitung jarak Euclidean antara dua node
    return ((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5

def a_star(graph, start, goal):
    print("Masuk A-star")
    open_set = []
    heapq.heappush(open_set, (0, start, [start]))
    g_costs = {start: 0}
    goal_coords = graph[goal][0]

    while open_set:
        _, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        for neighbor in graph[current][1]:
            if(neighbor not in cities.keys()):
                continue
            tentative_g_cost = g_costs[current] + heuristic(graph[current][0], graph[neighbor][0])
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                print("Masuk situ")
                g_costs[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic(graph[neighbor][0], goal_coords)
                heapq.heappush(open_set, (f_cost, neighbor, path + [neighbor]))
        print(path)

    return None

def play(screen):
    hitung1=0
    hitung2=0
    countmain=0
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    running = True
    global new_character, current_city
    path = dfs(cities, new_character, 'T')
    pathhps = a_star(cities, current_city, 'T')
    print("Path")
    print(path)
    print("Path Hapus")
    print(pathhps)
    input_text = ""
    error_message = ""
    winner = None  # Pemenang
    data_text = font.render("Kalahkan musuh dan capai kota T secepatnya!", True, BLACK)
    

    while running:
        
        hasil1="benar"
        hasil2="benar"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    next_city = input_text.upper()
                    print("posisi player: ")
                    print(current_city)
                    print("posisi bot:")
                    print(new_character)
                    print("Main")
                    print(countmain)
                    if(next_city in cities[current_city][1] and next_city in cities):
                        if countmain%2==0:
                            hasil1=tebak_cuaca(screen,1)  
                        else:
                            hasil1=w2f(screen,1)  
                        if hasil1=="benar":
                            current_city = next_city
                            if current_city == 'T':
                                winner = "Player 1"  # Jika player 1 mencapai kota "T"
                                if draw_cities(screen, winner) == "restart":
                                    # Reset papan saat tombol restart ditekan
                                    
                                    current_city = 'A'
                                    new_character = 'A'
                                    error_message = ""
                                    winner = None
                                    tutupJalan(screen,3,pathhps)
                                    input_text = ""
                                    path = dfs(cities, new_character, 'T')
                                    pathhps = a_star(cities, current_city, 'T')
                                    break
                                else:
                                    running = False  # Keluar dari permainan jika tombol exit ditekan
                            pathhps = a_star(cities, current_city, 'T')
                            print("Path Hapus:")
                            print(pathhps)
                            print("posisi player pindah:")
                            print(current_city)
                            hitung1=hitung1+1
                            if hitung1==1:
                                tutupJalan(screen,1, pathhps)
                                path = dfs(cities, new_character, 'T')
                                print("Path")
                                print(path)
                                pathhps = a_star(cities, current_city, 'T')
                                print("Path Hapus:")
                                print(pathhps)
                                hitung1=0
                            error_message = ""
                            
                            
                        if countmain%2==0:
                            hasil2=tebak_cuaca(screen,2)
                        else:
                            hasil2=w2f(screen,2)                           
                            
                        countmain=countmain+1
                                         
                        if hasil2=="benar":
                             
                            if path:
                                hitung2=hitung2+1
                                new_character = path[1]
                                if new_character == 'T':
                                    winner = "Player 2"  # Jika player 1 mencapai kota "T"
                                    if draw_cities(screen, winner) == "restart":
                                        # Reset papan saat tombol restart ditekan
                                        new_character = 'A'
                                        current_city = "A"
                                        error_message = ""
                                        input_text = ""
                                        winner = None
                                        tutupJalan(screen,3,pathhps)
                                        path = dfs(cities, new_character, 'T')
                                        pathhps = a_star(cities, current_city, 'T')
                                        break
                                    else:
                                        running = False  # Keluar dari permainan jika tombol exit ditekan
                                print("posisi bot pindah:")
                                print(new_character)
                                path.pop(1)
                                if hitung2==1:
                                    tutupJalan(screen,2,pathhps)
                                    path = dfs(cities, new_character, 'T')
                                    print("Path")
                                    print(path)
                                    pathhps = a_star(cities, current_city, 'T')
                                    print("Path Hapus:")
                                    print(pathhps)
                                    hitung2=0
                                draw_cities(screen, winner, new_character, show_exit_button=True)  # Menggambar kota dengan informasi kota saat ini
                            else:
                                input_text = ""
                                tutupJalan(screen,2,pathhps)
                                continue
                    else:
                        error_message = "Coba lagi."
                    input_text = ""
                    
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode

        screen.fill(WHITE)
        draw_cities(screen, winner)
        draw_character(screen, current_city, BLACK)
        draw_character(screen, new_character, BLUE)
        
        screen.blit(data_text, (20, 20))
        input_surface = font.render("Masukkan nama kota tujuan: " + input_text, True, BLACK)
        screen.blit(input_surface, (20, HEIGHT - 50))
        
        posisi = font.render("Posisi Player: " + current_city +" (Hitam)", True, BLACK)
        screen.blit(posisi, (20, HEIGHT - 100))
        error_font = pygame.font.SysFont(None, 24)
        
        posisi2 = font.render("Posisi Musuh: " + new_character +" (Biru)", True, BLACK)
        screen.blit(posisi2, (300, HEIGHT - 100))
        error_font = pygame.font.SysFont(None, 24)
        
        error_surface = error_font.render(error_message, True, RED)
        screen.blit(error_surface, (20, HEIGHT - 80))

        pygame.display.flip()
        clock.tick(30)
def main():
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("DFS on Cities")
    font = pygame.font.SysFont(None, 24)
    main = pygame.Rect(500, 400, 200, 50)
    keluar = pygame.Rect(800, 400, 200, 50)
    image = pygame.image.load("mario.png")
    background = pygame.image.load('background.png')
    background = pygame.transform.scale(background, (1500,800))
    clock = pygame.time.Clock()

    screen.fill(WHITE)
    screen.blit(background, (0,0))
    screen.blit(image, (675, 150))
    draw_button(screen, main, BLACK, "Main", WHITE)
    draw_button(screen, keluar, BLACK, "Keluar", WHITE)
    pygame.display.flip()   
    loop=True
    while loop:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if main.collidepoint(mouse_pos):
                    play(screen)
                    loop=False
                elif keluar.collidepoint(mouse_pos):
                    exit
                    loop=False
                # Update input text di layar
                screen.fill(WHITE)
                screen.blit(background, (0,0))
                screen.blit(image, (675, 150))
                draw_button(screen, main, BLACK, "Main", WHITE)
                draw_button(screen, keluar, BLACK, "Keluar", WHITE)
                pygame.display.flip()
                    
    screen.fill(WHITE)
    screen.blit(background, (0,0))
    screen.blit(image, (675, 150))
    draw_button(screen, main, BLACK, "Main", WHITE)
    draw_button(screen, keluar, BLACK, "Keluar", WHITE)
    pygame.display.flip()            


if __name__ == "__main__":
    main()
