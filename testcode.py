import os
import matplotlib.pyplot as plt

# 定义情绪种类
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# 统计每个情绪文件夹中的图片数量
emotion_counts = []
for emotion in emotions:
  emotion_dir = os.path.join("data/train", emotion)
  num_files = len(os.listdir(emotion_dir))
  emotion_counts.append(num_files)

# 生成柱状图
plt.bar(emotions, emotion_counts)
plt.xlabel("Emotion")
plt.ylabel("Number of Images")
plt.show()