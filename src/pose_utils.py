import math
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
from src import util
from src.model import bodypose_model


class Body(object):
    def __init__(self, model_path):
        self.model = bodypose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

            # extract outputs, resize, and remove padding
            # heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
            heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))  # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
            paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))  # output 0 is PAFs
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += + paf / len(multiplier)

        all_peaks = []
        peak_counter = 0
        refined_points = []
        id_joint = 0
        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]
            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse

            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            if len(peaks_with_score) > 0:
                for p in peaks_with_score:
                    refined_points.append((p, id_joint))
                    id_joint += 1
                    break
            else:
                refined_points.append(((), id_joint))
                id_joint += 1

            # print(peaks_with_score)
            peak_id = range(peak_counter, peak_counter + len(peaks))
            # print(peak_id)
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)

            peak_counter += len(peaks)

        return refined_points


def detect(img, body_estimation):

    coors_d = {}

    candidate = body_estimation(img)
    for can in candidate:
        try:
            if can[0][-1] > 0.3:
                coors_d[str(can[1])] = (can[0][0], can[0][1], can[0][-1])
        except:
            pass
    return coors_d


def angle_between_points( p0, p1, p2):
    a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
    if a * b == 0:
        return -1.0
    return math.acos((a+b-c) / math.sqrt(4*a*b)) * 180 / math.pi


def distance_formula(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def front_back_check(points, insights_data):
    if len(points) < 6:
        insights_data['Main Pose'] = 'Undetermined (Unclear Image)'
        return insights_data
    try:
        if points['2'][0] < points['5'][0]:
            insights_data['Main Pose'] = 'Front Facing Person'
            return insights_data
        else:
            insights_data['Main Pose'] = 'Back Turned Person'
            return insights_data
    except:
        insights_data['Main Pose'] = 'Undetermined (Unclear Image)'
        return insights_data


def stand_sit_jump_check(points, fb_code, insights_data):
    try:
        right_shoulder = points['2']
    except:
        right_shoulder = ()

    try:
        right_pelvis = points['8']
    except:
        right_pelvis = ()

    try:
        right_knee = points['9']
    except:
        right_knee = ()

    try:
        right_ankle = points['10']
    except:
        right_ankle = ()

    try:
        left_shoulder = points['5']
    except:
        left_shoulder = ()

    try:
        left_pelvis = points['11']
    except:
        left_pelvis = ()

    try:
        left_knee = points['12']
    except:
        left_knee = ()

    try:
        left_ankle = points['13']
    except:
        left_ankle = ()

    if fb_code == 'Front Facing Person':
        try:
            right_spk_angle = angle_between_points(right_shoulder, right_pelvis, right_knee)
        except:
            right_spk_angle = -1
        try:
            left_spk_angle = angle_between_points(left_shoulder, left_pelvis, left_knee)
        except:
            left_spk_angle = -1

        try:
            right_pka_angle = angle_between_points(right_pelvis, right_knee, right_ankle)
        except:
            right_pka_angle = -1
        try:
            left_pka_angle = angle_between_points(left_pelvis, left_knee, left_ankle)
        except:
            left_pka_angle = -1

        if right_spk_angle == -1 and left_spk_angle == -1:
            insights_data['Position'] = 'Standing'
            return insights_data
        elif (right_spk_angle != -1 and right_spk_angle <= 100) or (left_spk_angle != -1 and left_spk_angle <= 100):
            insights_data['Position'] = 'Sitting'
            return insights_data
        elif (right_spk_angle != -1 and right_spk_angle > 100) or (left_spk_angle != -1 and left_spk_angle > 100):
            if (right_pka_angle != -1 and right_pka_angle < 100) and (left_pka_angle != -1 and left_pka_angle < 100):
                insights_data['Position'] = 'Jumping'
                return insights_data
            else:
                try:
                    pelvis_dist = distance_formula(left_pelvis, right_pelvis)
                except:
                    pelvis_dist = -1
                try:
                    ankles_dist = distance_formula(left_ankle, right_ankle)
                except:
                    ankles_dist = -1
                if pelvis_dist != -1 and ankles_dist != -1:
                    if ankles_dist/pelvis_dist > 2.5:
                        insights_data['Position'] = 'Jumping'
                        return insights_data
                else:
                    insights_data['Position'] = 'Standing'
                    return insights_data

        insights_data['Position'] = 'Standing'
        return insights_data

    else:
        insights_data['Position'] = 'Could not Determine'
        return insights_data


def straight_side_check(points, fb_code, insights_data):
    if fb_code != 'Undetermined (Unclear Image)':
        if '3' not in points.keys() and '6' not in points.keys():
            insights_data['Detailed Posture'] = 'Not Determined (Unclear Image)'
            return insights_data
        else:
            arms = []
            shoulder_to_shoulder = distance_formula(points['2'], points['5'])
            try:
                right_arm = distance_formula(points['2'], points['3'])
                arms.append(right_arm)
            except:
                pass
            try:
                left_arm = distance_formula(points['5'], points['6'])
                arms.append(left_arm)
            except:
                pass
            if len(arms) > 0:
                arm_length = max(arms)

                if shoulder_to_shoulder/arm_length < 0.9:
                    pose_side = 'Strictly Turned'
                elif shoulder_to_shoulder/arm_length < 1:
                    pose_side = 'Slight Turned'
                else:
                    pose_side = 'Straight Pose'

                if 'Turned' in pose_side:
                    ssl = 0
                    ssr = 0

                    try:
                        if points['2'][-1] > points['5'][-1]:
                            ssl += 1
                        else:
                            ssr += 1

                    except:
                        if '5' not in points.keys():
                            ssl += 1
                        if '2' not in points.keys():
                            ssr += 1

                    try:
                        if points['3'][-1] > points['6'][-1]:
                            ssl += 1
                        else:
                            ssr += 1

                    except:
                        if '6' not in points.keys():
                            ssl += 1
                        if '3' not in points.keys():
                            ssr += 1

                    try:
                        if points['8'][-1] > points['11'][-1]:
                            ssl += 1
                        else:
                            ssr += 1

                    except:
                        if '11' not in points.keys():
                            ssl += 1
                        if '8' not in points.keys():
                            ssr += 1

                    if ssl > ssr:
                        additional_pose = 'Left'
                    else:
                        additional_pose = 'Right'
                    pose_side += f' {additional_pose}'
                insights_data['Detailed Posture'] = pose_side
                return insights_data

            else:
                insights_data['Detailed Posture'] = 'Not Determined (Unclear Image)'
                return insights_data
    else:
        insights_data['Detailed Posture'] = 'Not Determined (Unclear Image)'
        return insights_data


def limbs_data(points, insights_data):
    try:
        right_shoulder = points['2']
    except:
        right_shoulder = ()

    try:
        right_elbow = points['3']
    except:
        right_elbow = ()

    try:
        right_wrist = points['4']
    except:
        right_wrist = ()

    try:
        right_pelvis = points['8']
    except:
        right_pelvis = ()

    try:
        right_knee = points['9']
    except:
        right_knee = ()

    try:
        right_ankle = points['10']
    except:
        right_ankle = ()

    try:
        left_shoulder = points['5']
    except:
        left_shoulder = ()

    try:
        left_elbow = points['6']
    except:
        left_elbow = ()

    try:
        left_wrist = points['7']
    except:
        left_wrist = ()

    try:
        left_pelvis = points['11']
    except:
        left_pelvis = ()

    try:
        left_knee = points['12']
    except:
        left_knee = ()

    try:
        left_ankle = points['13']
    except:
        left_ankle = ()

    try:
        right_sew_angle = angle_between_points(right_shoulder, right_elbow, right_wrist)
        if right_sew_angle < 90:
            insights_data['Right Arm Posture'] = 'Strictly Bent'
        elif right_sew_angle < 125:
            insights_data['Right Arm Posture'] = 'Bent'
        elif right_sew_angle < 160:
            insights_data['Right Arm Posture'] = 'Slightly Bent'
        else:
            insights_data['Right Arm Posture'] = 'Straight'

    except:
        right_sew_angle = -1
        insights_data['Right Arm Posture'] = 'Undetermined (View Blocked)'

    try:
        left_sew_angle = angle_between_points(left_shoulder, left_elbow, left_wrist)
        if left_sew_angle < 90:
            insights_data['Left Arm Posture'] = 'Strictly Bent'
        elif left_sew_angle < 125:
            insights_data['Left Arm Posture'] = 'Bent'
        elif left_sew_angle < 160:
            insights_data['Left Arm Posture'] = 'Slightly Bent'
        else:
            insights_data['Left Arm Posture'] = 'Straight'
    except:
        left_sew_angle = -1
        insights_data['Left Arm Posture'] = 'Undetermined (View Blocked)'

    try:
        right_pka_angle = angle_between_points(right_pelvis, right_knee, right_ankle)
        if right_pka_angle < 90:
            insights_data['Right Knee Posture'] = 'Strictly Bent'
        elif right_pka_angle < 125:
            insights_data['Right Knee Posture'] = 'Bent'
        elif right_pka_angle < 160:
            insights_data['Right Knee Posture'] = 'Slightly Bent'
        else:
            insights_data['Right Knee Posture'] = 'Straight'
    except:
        right_pka_angle = -1
        insights_data['Right Knee Posture'] = 'Undetermined (View Blocked)'

    try:
        left_pka_angle = angle_between_points(left_pelvis, left_knee, left_ankle)
        if left_pka_angle < 90:
            insights_data['Left Knee Posture'] = 'Strictly Bent'
        elif left_pka_angle < 125:
            insights_data['Left Knee Posture'] = 'Bent'
        elif left_pka_angle < 160:
            insights_data['Left Knee Posture'] = 'Slightly Bent'
        else:
            insights_data['Left Knee Posture'] = 'Straight'
    except:
        left_pka_angle = -1
        insights_data['Left Knee Posture'] = 'Undetermined (View Blocked)'

    return insights_data


def draw_pose(img, points, pose_pairs, colors):
    for i in range(len(pose_pairs)):
        canvas = img.copy()
        seq = pose_pairs[i]
        if str(seq[0]) in list(points.keys()) and str(seq[1]) in list(points.keys()):
            x, y = (points[str(seq[0])][1], points[str(seq[1])][1]), (points[str(seq[0])][0], points[str(seq[1])][0])
            mx = np.mean(x)
            my = np.mean(y)
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(my), int(mx)), (int(length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])
            img = cv2.addWeighted(img, 0.2, canvas, 0.8, 0)
        else:
            continue
    return img


def get_insights(points):
    insights_data = {'Main Pose': '', 'Position': '', 'Detailed Posture': '', 'Right Arm Posture': '',
                     'Left Arm Posture': '', 'Right Knee Posture': '', 'Left Knee Posture': ''}

    insights_data = front_back_check(points, insights_data)
    insights_data = stand_sit_jump_check(points, insights_data['Main Pose'], insights_data)
    insights_data = straight_side_check(points, insights_data['Main Pose'], insights_data)
    insights_data = limbs_data(points, insights_data)
    return insights_data
