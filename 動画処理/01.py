
# !pip install opencv-python

import cv2
from google.colab.patches import cv2_imshow
import time
import csv
from datetime import datetime

# --- Python標準関数でRGB→グレースケール変換 ---
def to_grayscale(frame):
    height = len(frame)
    width = len(frame[0])
    gray = [[0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            r, g, b = frame[y][x][2], frame[y][x][1], frame[y][x][0]
            # 人間の視覚感度に合わせて輝度を計算
            gray[y][x] = int(0.299 * r + 0.587 * g + 0.114 * b)
    return gray


def get_diff_map(gray1, gray2, threshold=40):
    h = len(gray1)
    w = len(gray1[0])
    diff_map = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if abs(gray1[y][x] - gray2[y][x]) > threshold:
                diff_map[y][x] = 1
    return diff_map


# def extract_clusters(diff_map, min_cluster_size=80):
def extract_clusters(diff_map, min_cluster_size=100):
    h = len(diff_map)
    w = len(diff_map[0])
    visited = [[False for _ in range(w)] for _ in range(h)]
    clusters = []

    def dfs(y, x, pixels):
        stack = [(y, x)]
        visited[y][x] = True
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if not visited[ny][nx] and diff_map[ny][nx] == 1:
                            visited[ny][nx] = True
                            stack.append((ny, nx))

    for y in range(h):
        for x in range(w):
            if diff_map[y][x] == 1 and not visited[y][x]:
                pixels = []
                dfs(y, x, pixels)
                if len(pixels) >= min_cluster_size:
                    sum_y = sum(p[0] for p in pixels)
                    sum_x = sum(p[1] for p in pixels)
                    cy, cx = sum_y // len(pixels), sum_x // len(pixels)
                    clusters.append((cy, cx))
    return clusters

# --- クラスタ間のベクトル計算 ---
def compute_motion_vectors(prev_clusters, curr_clusters, max_distance=50):
    motions = []
    match_dict = {}
    for i, (py, px) in enumerate(prev_clusters):
        closest = None
        min_dist = float('inf')
        for j, (cy, cx) in enumerate(curr_clusters):
            dist = abs(px - cx) + abs(py - cy)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                closest = (j, cy, cx)
        if closest:
            j, cy, cx = closest
            dy, dx = cy - py, cx - px
            motions.append((dx, dy))
            match_dict[j] = (dx, dy, py, px)
    return motions, match_dict

# 背景移動を除いた相対dxで判定
# def is_flowing_v4(motions, min_avg_rel_dx=1.5, min_motion_count=2):
def is_flowing_v4(motions, min_avg_rel_dx=1.0, min_motion_count=1):
    if not motions:
        return False, []
    all_dx = [dx for dx, dy in motions]
    bg_dx = sum(all_dx) / len(all_dx)
    rel_dx_list = [dx - bg_dx for dx in all_dx]
    true_motion_count = sum(1 for rdx in rel_dx_list if rdx >= min_avg_rel_dx)
    return true_motion_count >= min_motion_count, rel_dx_list

# --- メイン処理（全フレームプレビュー） ---
def run_with_preview_skip(video_path, skip_rate=5):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("動画が読み込めません")
        return

    frame_count = 0
    prev_clusters = []
    first_frame = cv2.resize(first_frame, (320, 240))
    background_gray = to_grayscale(first_frame)
    recent_motion_results = []
    log_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        gray = to_grayscale(frame)
        diff_map = get_diff_map(background_gray, gray)
        curr_clusters = extract_clusters(diff_map, min_cluster_size=80)
        motions, match_dict = compute_motion_vectors(prev_clusters, curr_clusters)

        is_moving, rel_dx_list = is_flowing_v4(motions)
        recent_motion_results.append(1 if is_moving else 0)

        ###### 判定に使うフレーム数 10 フレーム
        if len(recent_motion_results) > 10:
            recent_motion_results.pop(0)
        
        ### 6以上なら、動いている => GOGOGO 判定
        flowing = sum(recent_motion_results) >= 6
        status_label = "GOGOGO" if flowing else "STOP"

        # ★★★ 10履歴がそろったらログ記録
        if len(recent_motion_results) == 10:
            log_data.append([frame_count] + list(recent_motion_results) + [status_label])

        # 間引きフレームだけ表示（Colab高速化）
        if frame_count % skip_rate == 0:
            display_frame = frame.copy()
            color = (0, 0, 255) if flowing else (0, 255, 0)
            text = "GOGOGO" if flowing else "STOP"
            cv2.rectangle(display_frame, (10, 10), (150, 50), color, -1)
            cv2.putText(display_frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            for i, (cy, cx) in enumerate(curr_clusters):
                cv2.circle(display_frame, (cx, cy), 4, (255, 255, 0), -1)
                if i in match_dict:
                    dx = match_dict[i][0]
                    cv2.putText(display_frame, f"dx={dx}", (cx+5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            cv2_imshow(display_frame)
            time.sleep(0.1)

        prev_clusters = curr_clusters
        frame_count += 1

    cap.release()

    # --- CSV出力 ---
    now = datetime.now()
    timestr = now.strftime("%H%M%S")
    csv_path = f"/content/drive/MyDrive/動画処理・解析 2025 07-08/recent_motion_history_label_log_{timestr}.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["frame"] + [f"recent_{i+1}" for i in range(10)] + ["status_label"]
        writer.writerow(header)
        for row in log_data:
            writer.writerow(row)
    print("判定履歴ログを出力しました:", csv_path)

###### 実行
run_with_preview_skip("対象動画パス"）