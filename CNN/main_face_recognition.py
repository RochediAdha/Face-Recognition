import face_recognition
import cv2
import os


# Fungsi untuk memuat gambar wajah yang dikenal
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    # Iterasi melalui file gambar di folder
    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith((".jpg", ".png", ".jpeg")):  # Cek jika file adalah gambar
            image_path = os.path.join(known_faces_dir, file_name)
            image = face_recognition.load_image_file(image_path)

            # Ekstraksi encoding wajah
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)

            # Gunakan nama file (tanpa ekstensi) sebagai label
            known_face_names.append(os.path.splitext(file_name)[0])

    return known_face_encodings, known_face_names


# Path ke folder wajah dikenal
KNOWN_FACES_DIR = "known_faces"

# Validasi folder
if not os.path.exists(KNOWN_FACES_DIR):
    raise FileNotFoundError(
        f"Folder '{KNOWN_FACES_DIR}' tidak ditemukan. Pastikan folder ini berisi gambar wajah yang dikenal."
    )

# Load wajah yang dikenal
known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)
print("Wajah dikenal berhasil dimuat:", known_face_names)

# Inisialisasi video capture (menggunakan webcam)
video_capture = cv2.VideoCapture(0)

print("Tekan 'q' untuk keluar dari streaming.")

while True:
    # Tangkap frame dari webcam
    ret, frame = video_capture.read()

    # Resize frame untuk mempercepat proses
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Deteksi wajah dalam frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Bandingkan wajah terdeteksi dengan wajah dikenal
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Cari wajah dengan jarak terdekat
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Tampilkan hasil pada frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Skala kembali posisi wajah
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Tambahkan label nama
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
        )
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Tampilkan video feed
    cv2.imshow("Face Recognition", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bersihkan sumber daya
video_capture.release()
cv2.destroyAllWindows()
