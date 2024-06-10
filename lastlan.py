import face_recognition
import cv2
import numpy as np

# 웹캠 #0 (기본 웹캠)에 대한 참조 가져오기
video_capture = cv2.VideoCapture(0)

# 샘플 사진을 로드하고 얼굴 인코딩을 학습합니다.
park_image = face_recognition.load_image_file("park.jpg")
park_face_encoding = face_recognition.face_encodings(park_image)[0]

# 두 번째 샘플 사진을 로드하고 얼굴 인코딩을 학습합니다.
Dicaprio_image = face_recognition.load_image_file("Dicaprio.jpg")
Dicaprio_face_encoding = face_recognition.face_encodings(Dicaprio_image)[0]

# 알려진 얼굴 인코딩 및 이름의 배열 생성
known_face_encodings = [
    park_face_encoding,
    Dicaprio_face_encoding
]
known_face_names = [
    "park",
    "Dicaprio"
]

# 일부 변수를 초기화
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # 단일 프레임의 비디오 가져오기
    ret, frame = video_capture.read()

    # 시간 절약을 위해 매 프레임마다 처리하지 않음
    if process_this_frame:
        # 얼굴 인식 처리를 위해 비디오 프레임을 1/4 크기로 조정
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # 이미지를 BGR 색상(OpenCV 사용)에서 RGB 색상(face_recognition 사용)으로 변환
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # 현재 비디오 프레임에서 모든 얼굴 위치와 얼굴 인코딩 찾기
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_probabilities = []
        for face_encoding in face_encodings:
            # 얼굴이 알려진 얼굴과 일치하는지 확인
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # 얼굴 인코딩의 거리 계산
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # 확률 계산
            probabilities = 1 - face_distances
            probability_text = f"Park: {probabilities[0]:.2f}, Dicaprio: {probabilities[1]:.2f}"

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            face_probabilities.append(probability_text)

    process_this_frame = not process_this_frame

    # 결과 표시
    for (top, right, bottom, left), name, probability_text in zip(face_locations, face_names, face_probabilities):
        # 프레임을 1/4 크기로 축소했으므로 얼굴 위치를 다시 원래 크기로 조정
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 얼굴 주위에 상자 그리기
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 얼굴 아래에 이름 레이블 그리기
        cv2.rectangle(frame, (left, bottom - 50), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 36), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, probability_text, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # 결과 이미지 표시
    cv2.imshow('Video', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 핸들 해제
video_capture.release()
cv2.destroyAllWindows()