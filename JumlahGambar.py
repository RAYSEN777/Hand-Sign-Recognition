import os

def hitung_jumlah_gambar(folder_path):
    # Ekstensi gambar yang umum
    ekstensi_gambar = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
    total_gambar = 0

    # Memeriksa apakah direktori/folder tersebut ada
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' tidak ditemukan.")
        return 0

    # os.walk akan menelusuri folder utama dan semua subfolder di dalamnya
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Memeriksa apakah file diakhiri dengan salah satu ekstensi gambar
            if file.lower().endswith(ekstensi_gambar):
                total_gambar += 1

    return total_gambar

# --- CARA PENGGUNAAN ---
# Ganti path di bawah ini sesuai lokasi folder yang ingin Anda hitung
path_folder = r"Dataset/CNN_Images_Cleaned"

jumlah = hitung_jumlah_gambar(path_folder)
print(f"Total jumlah gambar ditemukan: {jumlah}")