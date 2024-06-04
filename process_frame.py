import cv2
import numpy as np
import mediapipe as mp
import time
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line
import  thresholds

class ProcessFrame:
    def __init__(self, flip_frame=False, thresholds=None):

        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame
        # self.thresholds
        self.thresholds = thresholds
        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # line type
        self.linetype = cv2.LINE_AA
        # set radius to draw arc
        self.radius = 20
        # Colors in BGR format.
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (0, 255, 255),
            'light_blue': (102, 204, 255)
        }
        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
            'shoulder': 11,
            'elbow': 13,
            'wrist': 15,
            'hip': 23,
            'knee': 25,
            'ankle': 27,
            'foot': 31
        }
        self.right_features = {
            'shoulder': 12,
            'elbow': 14,
            'wrist': 16,
            'hip': 24,
            'knee': 26,
            'ankle': 28,
            'foot': 32
        }
        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0
        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            'state_seq': [],
            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,
            'DISPLAY_TEXT': np.full((4,), False),
            'COUNT_FRAMES': np.zeros((4,), dtype=np.int64),
            'LOWER_HIPS': False,
            'INCORRECT_POSTURE': False,
            'prev_state': None,
            'curr_state': None,
            'SQUAT_COUNT': 0,
            'IMPROPER_SQUAT': 0,
            'SCORE': 0
        }
        self.FEEDBACK_ID_MAP = {
            0: ('BEND BACKWARDS', 215, (0, 153, 255)),
            1: ('BEND FORWARD', 215, (0, 153, 255)),
            2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
            3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
        }

    def _get_state(self, knee_angle):
        knee = None
        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            knee = 3
        return f's{knee}' if knee else None

    def _update_state_sequence(self, state):
        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')) == 0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2') == 1)):
                self.state_tracker['state_seq'].append(state)
        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']:
                self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):
        if lower_hips_disp:
            draw_text(
                frame,
                'LOWER YOUR HIPS',
                pos=(30, 80),
                text_color=(0, 0, 0),
                font_scale=0.6,
                text_color_bg=(255, 255, 0)
            )
        for idx in np.where(c_frame)[0]:
            draw_text(
                frame,
                dict_maps[idx][0],
                pos=(30, dict_maps[idx][1]),
                text_color=(255, 255, 230),
                font_scale=0.6,
                text_color_bg=dict_maps[idx][2]
            )
        return frame

    def process(self, frame: np.array, pose, exercise):

        if exercise == "squat":
            play_sound = None
            frame_height, frame_width, _ = frame.shape
            # Process the image
            keypoints = pose.process(frame)
            if keypoints.pose_landmarks:
                ps_lm = keypoints.pose_landmarks
                nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width,
                                                   frame_height)
                left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                    get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
                right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                    get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)
                offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)
                if offset_angle > self.thresholds['OFFSET_THRESH']:
                    display_inactivity = False
                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker[
                        'start_inactive_time_front']
                    self.state_tracker['start_inactive_time_front'] = end_time
                    if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0
                        self.state_tracker['IMPROPER_SQUAT'] = 0
                        self.state_tracker['SCORE'] = 0  # Puan sıfırlanır
                        display_inactivity = True
                    cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                    cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                    cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)
                    if self.flip_frame:
                        frame = cv2.flip(frame, 1)
                    if display_inactivity:
                        play_sound = 'reset_counters'
                        self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                        self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                    draw_text(
                        frame,
                        "DOGRU: " + str(self.state_tracker['SQUAT_COUNT']),
                        pos=(int(frame_width * 0.68), 30),
                        text_color=(255, 255, 230),
                        font_scale=0.7,
                        text_color_bg=(18, 185, 0)
                    )
                    draw_text(
                        frame,
                        "YANLIS: " + str(self.state_tracker['IMPROPER_SQUAT']),
                        pos=(int(frame_width * 0.68), 80),
                        text_color=(255, 255, 230),
                        font_scale=0.7,
                        text_color_bg=(221, 0, 0),
                    )
                    draw_text(
                        frame,
                        'CAMERA NOT ALIGNED PROPERLY!!!',
                        pos=(30, frame_height - 60),
                        text_color=(255, 255, 230),
                        font_scale=0.65,
                        text_color_bg=(255, 153, 0),
                    )
                    draw_text(
                        frame,
                        'OFFSET ANGLE: ' + str(offset_angle),
                        pos=(30, frame_height - 30),
                        text_color=(255, 255, 230),
                        font_scale=0.65,
                        text_color_bg=(255, 153, 0),
                    )
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0
                    self.state_tracker['prev_state'] = None
                    self.state_tracker['curr_state'] = None
                else:
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()
                    dist_l_sh_hip = abs(left_foot_coord[1] - left_shldr_coord[1])
                    dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord)[1]
                    shldr_coord = None
                    elbow_coord = None
                    wrist_coord = None
                    hip_coord = None
                    knee_coord = None
                    ankle_coord = None
                    foot_coord = None
                    if dist_l_sh_hip > dist_r_sh_hip:
                        shldr_coord = left_shldr_coord
                        elbow_coord = left_elbow_coord
                        wrist_coord = left_wrist_coord
                        hip_coord = left_hip_coord
                        knee_coord = left_knee_coord
                        ankle_coord = left_ankle_coord
                        foot_coord = left_foot_coord
                        multiplier = -1
                    else:
                        shldr_coord = right_shldr_coord
                        elbow_coord = right_elbow_coord
                        wrist_coord = right_wrist_coord
                        hip_coord = right_hip_coord
                        knee_coord = right_knee_coord
                        ankle_coord = right_ankle_coord
                        foot_coord = right_foot_coord
                        multiplier = 1
                    hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                    cv2.ellipse(frame, hip_coord, (30, 30),
                                angle=0, startAngle=-90, endAngle=-90 + multiplier * hip_vertical_angle,
                                color=self.COLORS['white'], thickness=3, lineType=self.linetype)
                    draw_dotted_line(frame, hip_coord, start=hip_coord[1] - 80, end=hip_coord[1] + 20,
                                     line_color=self.COLORS['blue'])
                    knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                    cv2.ellipse(frame, knee_coord, (20, 20),
                                angle=0, startAngle=-90, endAngle=-90 - multiplier * knee_vertical_angle,
                                color=self.COLORS['white'], thickness=3, lineType=self.linetype)
                    draw_dotted_line(frame, knee_coord, start=knee_coord[1] - 50, end=knee_coord[1] + 20,
                                     line_color=self.COLORS['blue'])
                    ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                    cv2.ellipse(frame, ankle_coord, (30, 30),
                                angle=0, startAngle=-90, endAngle=-90 + multiplier * ankle_vertical_angle,
                                color=self.COLORS['white'], thickness=3, lineType=self.linetype)
                    draw_dotted_line(frame, ankle_coord, start=ankle_coord[1] - 50, end=ankle_coord[1] + 20,
                                     line_color=self.COLORS['blue'])
                    cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                    cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                    cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                    cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                    cv2.line(frame, ankle_coord, knee_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                    cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                    cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                    cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                    cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                    cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                    cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                    cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                    cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1, lineType=self.linetype)
                    current_state = self._get_state(int(knee_vertical_angle))
                    self.state_tracker['curr_state'] = current_state
                    self._update_state_sequence(current_state)
                    if current_state == 's1':
                        if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                            self.state_tracker['SQUAT_COUNT'] += 1
                            play_sound = str(self.state_tracker['SQUAT_COUNT'])
                            # Doğru squat: 10 puan
                            self.state_tracker['SCORE'] += 10
                        elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                            self.state_tracker['IMPROPER_SQUAT'] += 1
                            play_sound = 'incorrect'
                            # Yanlış squat: 0 puan
                            self.state_tracker['SCORE'] += 0
                        elif self.state_tracker['INCORRECT_POSTURE']:
                            self.state_tracker['IMPROPER_SQUAT'] += 1
                            play_sound = 'incorrect'
                            # Yanlış squat: 0 puan
                            self.state_tracker['SCORE'] += 0
                        self.state_tracker['state_seq'] = []
                        self.state_tracker['INCORRECT_POSTURE'] = False
                    else:
                        # ... (bu bölüm önceki kodla aynı) ...

                        # Eğer squat tamamlanmamışsa, 5 puan ver
                        if 's2' in self.state_tracker['state_seq'] and current_state != 's1':
                            self.state_tracker['SCORE'] += 0
                            # ... (bu bölüm önceki kodla aynı) ...

                    # Puanları ekrana yazdır
                    draw_text(
                        frame,
                        "DOGRU: " + str(self.state_tracker['SQUAT_COUNT']),
                        pos=(int(frame_width * 0.68), 30),
                        text_color=(255, 255, 230),
                        font_scale=0.7,
                        text_color_bg=(18, 185, 0)
                    )
                    draw_text(
                        frame,
                        "YANLIS: " + str(self.state_tracker['IMPROPER_SQUAT']),
                        pos=(int(frame_width * 0.68), 80),
                        text_color=(255, 255, 230),
                        font_scale=0.7,
                        text_color_bg=(221, 0, 0),
                    )
                    draw_text(
                        frame,
                        "PUAN: " + str(self.state_tracker['SCORE']),
                        pos=(int(frame_width * 0.68), 130),
                        text_color=(255, 255, 230),
                        font_scale=0.7,
                        text_color_bg=(18, 185, 0)
                    )
            else:
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                display_inactivity = False
                if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    self.state_tracker['SCORE'] = 0  # Puan sıfırlanır
                    display_inactivity = True
                self.state_tracker['start_inactive_time'] = end_time
                draw_text(
                    frame,
                    "DOGRU: " + str(self.state_tracker['SQUAT_COUNT']),
                    pos=(int(frame_width * 0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )
                draw_text(
                    frame,
                    "YANLIS: " + str(self.state_tracker['IMPROPER_SQUAT']),
                    pos=(int(frame_width * 0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                )
                draw_text(
                    frame,
                    "PUAN: " + str(self.state_tracker['SCORE']),
                    pos=(int(frame_width * 0.68), 130),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )
                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0
            pass
        elif exercise == "elbow-plank":

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_drawing = mp.solutions.drawing_utils

            def calculate_angle(a, b, c):
                a = np.array(a)  # First point
                b = np.array(b)  # Mid point
                c = np.array(c)  # End point

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360.0 - angle

                return angle

            # Dirsek plank pozisyonunu kontrol etmek için bir fonksiyon
            def is_elbow_plank(landmarks):
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_side_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
                right_side_angle = calculate_angle(right_shoulder, right_hip, right_ankle)

                # Açının 165 ila 195 derece arasında olup olmadığını kontrol et
                if 165 <= left_side_angle <= 195 and 165 <= right_side_angle <= 195:
                    return True
                return False

            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    landmarks = results.pose_landmarks.landmark

                    if is_elbow_plank(landmarks):
                        cv2.putText(frame, 'Dogru', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, 'Yanlis', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('Elbow Plank Detection', frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

            pass

        elif exercise == "punches":

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_drawing = mp.solutions.drawing_utils

            def calculate_angle(a, b, c):
                a = np.array(a)  # First point
                b = np.array(b)  # Mid point
                c = np.array(c)  # End point

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360.0 - angle

                return angle

            def is_punch(landmarks):
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_punch_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_punch_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Yumruğun doğru olması için dirseklerin neredeyse düz olması gerekir
                if left_punch_angle > 120 or right_punch_angle > 120:
                    return True
                return False

            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    landmarks = results.pose_landmarks.landmark

                    if is_punch(landmarks):
                        cv2.putText(frame, 'Punch: Dogru', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    else:
                        cv2.putText(frame, 'Punch: Yanlis', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)

                cv2.imshow('Punch Detection', frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

            pass

        elif exercise == "leg-curls":

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_drawing = mp.solutions.drawing_utils

            def calculate_angle(a, b, c):
                a = np.array(a)  # First point
                b = np.array(b)  # Mid point
                c = np.array(c)  # End point

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360.0 - angle

                return angle

            def is_leg_curl(landmarks):
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Diz bükülme açısının leg curl için doğru olup olmadığını kontrol ediyoruz
                if (left_knee_angle < 45 or right_knee_angle < 45):
                    return True
                return False

            hareket_tamamlandi = False  # Hareket tamamlandı mı kontrolü için bir durum değişkeni
            cap = cv2.VideoCapture(0)
            score = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    landmarks = results.pose_landmarks.landmark

                    if is_leg_curl(
                            landmarks) and not hareket_tamamlandi:  # Hareket tamamlanmışsa ve daha önce tamamlanmamışsa
                        score += 10
                        hareket_tamamlandi = True  # Hareket tamamlandı durumunu güncelle
                        cv2.putText(frame, 'Leg Curl: Dogru', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    elif not is_leg_curl(landmarks):  # Hareket tamamlanmamışsa ve yanlış yapılıyorsa
                        hareket_tamamlandi = False  # Hareket tamamlandı durumunu sıfırla
                        cv2.putText(frame, 'Leg Curl: Yanlis', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)

                cv2.putText(frame, f'Score: {score}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow('Leg Curl Detection', frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

            pass


        elif exercise == "high_knees":

            mp_pose = mp.solutions.pose

            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            mp_drawing = mp.solutions.drawing_utils

            def is_high_knees(landmarks):

                left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y

                right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y

                # Yüksek dizler eşiği

                high_knees_threshold = 0.7

                if left_hip_y < high_knees_threshold and right_hip_y < high_knees_threshold:
                    return True

                return False

            cap = cv2.VideoCapture(0)

            score = 0

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = pose.process(image)

                if results.pose_landmarks:

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    landmarks = results.pose_landmarks.landmark

                    if is_high_knees(landmarks):  # Yüksek dizler yapıldıysa

                        score += 10  # Puanı artır

                        cv2.putText(frame, 'High Knees: Dogru', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,

                                    cv2.LINE_AA)

                    else:

                        cv2.putText(frame, 'High Knees: Yanlis', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,

                                    cv2.LINE_AA)

                cv2.putText(frame, f'Score: {score}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,

                            cv2.LINE_AA)

                cv2.imshow('High Knees Detection', frame)

                key = cv2.waitKey(1)

                if key == ord('q') or key == 27:  # 'q' tuşuna veya ESC tuşuna basılınca

                    break

            cap.release()

            cv2.destroyAllWindows()

            pass

        return frame, play_sound

